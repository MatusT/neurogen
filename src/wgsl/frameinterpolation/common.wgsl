// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_common.h: bilinear gather, device-depth to view-space
// conversion, the packed motion-vector-field entry format, and the buffer
// addressing this port needs.
//
// AMD keeps the motion vector fields and the reconstructed depths in textures
// it hits with InterlockedMax/InterlockedMin. WebGPU has no atomics on storage
// textures, so those resources are atomic storage buffers here and every access
// flattens the pixel coordinate itself. Bounds must therefore be checked before
// flattening — a texture drops an out-of-range store, a buffer index does not,
// and a negative row would alias the row above.

// @import ./params.wgsl
// @import ../core/math.wgsl
// @import ../core/pack.wgsl

fn fiInBounds(pos: vec2<i32>, size: vec2<i32>) -> bool {
    return all(pos >= vec2<i32>(0)) && all(pos < size);
}

fn fiFlatIndex(pos: vec2<i32>, size: vec2<i32>) -> u32 {
    return u32(pos.y * size.x + pos.x);
}

fn fiIsUvInside(uv: vec2<f32>) -> bool {
    return all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
}

fn fiMinDividedByMax(v0: f32, v1: f32) -> f32 {
    let m = max(v0, v1);
    return select(0.0, min(v0, v1) / m, m != 0.0);
}

fn fiNormalizedDot3(v0: vec3<f32>, v1: vec3<f32>) -> f32 {
    let maxLength = max(length(v0), length(v1));
    return select(1.0, dot(v0 / maxLength, v1 / maxLength), maxLength > 0.0);
}

//
// BILINEAR GATHER
//

// AMD's BilinearSamplingData. The four offsets are the fixed 2x2 quad, so they
// are recomputed from the index instead of stored.
struct FiBilinear {
    basePos: vec2<i32>,
    weights: vec4<f32>,
}

fn fiBilinear(uv: vec2<f32>, size: vec2<i32>) -> FiBilinear {
    let samplePos = uv * vec2<f32>(size) - vec2<f32>(0.5);
    let frac = fract(samplePos);

    var data: FiBilinear;
    data.basePos = vec2<i32>(floor(samplePos));
    data.weights = vec4<f32>(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y,
    );
    return data;
}

// Quad order (0,0), (1,0), (0,1), (1,1) — the order `weights` is built in.
fn fiBilinearOffset(index: i32) -> vec2<i32> {
    return vec2<i32>(index & 1, index >> 1);
}

// Colour gathered over the taps that land on screen, renormalised by the weight
// they carry, plus that weight sum. A sum of zero means the reprojection landed
// entirely off screen and the colour is meaningless — callers branch on it
// rather than on the colour.
fn fiGatherColor(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<i32>) -> vec4<f32> {
    let bilinear = fiBilinear(uv, size);

    var color = vec3<f32>(0.0);
    var weightSum = 0.0;
    for (var i = 0; i < 4; i++) {
        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, size)) { continue; }

        let weight = bilinear.weights[i];
        color += textureLoad(tex, samplePos, 0).rgb * weight;
        weightSum += weight;
    }

    return vec4<f32>(select(vec3<f32>(0.0), color / weightSum, weightSum != 0.0), weightSum);
}

// The clamp-to-edge equivalent, standing in for AMD's plain linear-sampler
// reads. Off-screen taps repeat the edge texel instead of dropping out, so the
// weights always sum to 1 and no renormalisation is needed.
fn fiSampleColorClamped(tex: texture_2d<f32>, uv: vec2<f32>, size: vec2<i32>) -> vec3<f32> {
    let bilinear = fiBilinear(uv, size);

    var color = vec3<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let samplePos = clamp(bilinear.basePos + fiBilinearOffset(i), vec2<i32>(0), size - vec2<i32>(1));
        color += textureLoad(tex, samplePos, 0).rgb * bilinear.weights[i];
    }

    return color;
}

//
// DEPTH
//

// AMD passes `fDeviceToViewDepth` in as four host-computed constants. They are
// derived here instead, from the projection planes the caller already has to
// supply, so there is one fewer thing for orchestration to get wrong.
//
// A standard (non-inverted, finite) perspective projection gives
// d = A + B/z with A = far/(far-near) and B = -far*near/(far-near), so
// z = B/(d - A). A > 1 for any 0 < near < far and d never exceeds 1, so the
// denominator cannot reach zero for a valid device depth.
fn fiViewSpaceDepth(deviceDepth: f32) -> f32 {
    let range = params.farPlane - params.nearPlane;
    let a = params.farPlane / range;
    let b = -params.farPlane * params.nearPlane / range;

    return b / (deviceDepth - a);
}

// Tangent of the half field of view, horizontal in .x and vertical in .y —
// AMD's fDeviceToViewDepth.zw, i.e. the NDC-to-view-space scale of X and Y at
// unit depth.
fn fiTanHalfFov() -> vec2<f32> {
    let tanHalfFovVertical = tan(params.verticalFovRadians * 0.5);
    let aspect = f32(params.renderSize.x) / f32(params.renderSize.y);

    return vec2<f32>(tanHalfFovVertical * aspect, tanHalfFovVertical);
}

// Unwritten reconstructed-depth cells must read as the far plane: the scatter
// keeps the nearest depth with an atomic min over the float bit patterns, which
// is an ordinary integer min only because device depths are non-negative.
const FI_DEPTH_FAR_SENTINEL: f32 = 1.0;

//
// MOTION VECTOR FIELD
//

// One 32-bit entry per component: priority in the high bits, the f16 vector
// coefficient in the low 16. Ordering the fields this way is what lets a single
// atomic max resolve the scatter — the comparison reaches the coefficient bits
// only when two writers tie on priority.
const MV_FIELD_PRIORITY_LOW_OFFSET: u32 = 16u;
const MV_FIELD_PRIORITY_HIGH_OFFSET: u32 = 21u;
const MV_FIELD_PRIMARY_BIT: u32 = 1u << 31u;
const MV_FIELD_PRIORITY_LOW_MAX: u32 = 31u;
const MV_FIELD_PRIORITY_HIGH_MAX: u32 = 1023u;
const MV_FIELD_COEFFICIENT_MASK: u32 = 0xffffu;

// Which kind of vector an entry carries. Primary entries are a pixel's own
// motion; secondary ones are the trail behind it, stored with their depth
// priority reversed so the atomic max keeps the furthest rather than the
// nearest. WGSL has no enum type, so these are the named alternatives.
const MV_FIELD_SECONDARY: u32 = 0u;
const MV_FIELD_PRIMARY: u32 = 1u;

struct FiVectorFieldEntry {
    motionVector: vec2<f32>,
    highPriorityFactor: f32,
    lowPriorityFactor: f32,
    valid: bool,
    primary: bool,
    velocity: f32,
    negOutside: bool,
    posOutside: bool,
}

fn fiNewVectorFieldEntry() -> FiVectorFieldEntry {
    var entry: FiVectorFieldEntry;
    entry.motionVector = vec2<f32>(0.0);
    entry.highPriorityFactor = 0.0;
    entry.lowPriorityFactor = 0.0;
    entry.valid = false;
    entry.primary = false;
    entry.velocity = 0.0;
    entry.negOutside = false;
    entry.posOutside = false;
    return entry;
}

fn fiPackedEntryIsPrimary(packed: u32) -> bool {
    return (packed & MV_FIELD_PRIMARY_BIT) != 0u;
}

fn fiPackVectorField(kind: u32, highPriority: u32, lowPriority: u32, motionVector: vec2<f32>) -> vec2<u32> {
    let priority = (kind * MV_FIELD_PRIMARY_BIT)
        | ((highPriority & MV_FIELD_PRIORITY_HIGH_MAX) << MV_FIELD_PRIORITY_HIGH_OFFSET)
        | ((lowPriority & MV_FIELD_PRIORITY_LOW_MAX) << MV_FIELD_PRIORITY_LOW_OFFSET);

    return vec2<u32>(
        priority | (ffxF32ToF16(motionVector.x) & MV_FIELD_COEFFICIENT_MASK),
        priority | (ffxF32ToF16(motionVector.y) & MV_FIELD_COEFFICIENT_MASK),
    );
}

fn fiUnpackVectorField(packed: vec2<u32>) -> FiVectorFieldEntry {
    var entry = fiNewVectorFieldEntry();

    entry.highPriorityFactor = f32((packed.x >> MV_FIELD_PRIORITY_HIGH_OFFSET) & MV_FIELD_PRIORITY_HIGH_MAX)
        / f32(MV_FIELD_PRIORITY_HIGH_MAX);
    entry.lowPriorityFactor = f32((packed.x >> MV_FIELD_PRIORITY_LOW_OFFSET) & MV_FIELD_PRIORITY_LOW_MAX)
        / f32(MV_FIELD_PRIORITY_LOW_MAX);

    entry.primary = fiPackedEntryIsPrimary(packed.x);
    // A cleared cell is all zeroes, and every scattered entry carries a
    // non-zero high priority, so this distinguishes "written" from "empty".
    entry.valid = entry.highPriorityFactor > 0.0;

    // Secondary vectors are stored with their depth priority reversed so that
    // the atomic max keeps the *furthest* of them; undo that on read.
    if (entry.valid && !entry.primary) {
        entry.highPriorityFactor = 1.0 - entry.highPriorityFactor;
    }

    entry.motionVector = vec2<f32>(
        unpack2x16float(packed.x & MV_FIELD_COEFFICIENT_MASK).x,
        unpack2x16float(packed.y & MV_FIELD_COEFFICIENT_MASK).x,
    );
    return entry;
}

// The reprojection tests every consumer of an entry needs. Both are taken at
// half the frame interval in each direction, because a field entry holds the
// half-frame vector.
fn fiAnnotateReprojection(entry: ptr<function, FiVectorFieldEntry>, uv: vec2<f32>) {
    let motionVector = (*entry).motionVector;
    (*entry).negOutside = !fiIsUvInside(uv - motionVector);
    (*entry).posOutside = !fiIsUvInside(uv + motionVector);
    (*entry).velocity = length(motionVector);
}

//
// COLOUR
//

const REC709_LUMA_WEIGHTS = vec3<f32>(0.2126, 0.7152, 0.0722);
const REC2020_LUMA_WEIGHTS = vec3<f32>(0.2627, 0.678, 0.0593);

const TRANSFER_FUNCTION_LINEAR_LDR: u32 = 0u;
const TRANSFER_FUNCTION_PQ: u32 = 1u;
const TRANSFER_FUNCTION_SCRGB: u32 = 2u;

// PQ encodes absolute nits with a 10000 nit peak; scRGB is scaled so 1.0 is
// 80 nits. Both are renormalised against the display's own peak luminance.
const PQ_PEAK_NITS: f32 = 10000.0;
const SCRGB_NITS_PER_UNIT: f32 = 80.0;

fn fiRawRgbToLinear(rawRgb: vec3<f32>) -> vec3<f32> {
    if (params.backbufferTransferFunction == TRANSFER_FUNCTION_PQ) {
        return ffxLinearFromPQ(rawRgb) * (PQ_PEAK_NITS / params.minMaxLuminance.y);
    }

    if (params.backbufferTransferFunction == TRANSFER_FUNCTION_SCRGB) {
        let range = (params.minMaxLuminance.y - params.minMaxLuminance.x) / SCRGB_NITS_PER_UNIT;
        return (rawRgb - vec3<f32>(params.minMaxLuminance.x / SCRGB_NITS_PER_UNIT)) / vec3<f32>(range);
    }

    return ffxLinearFromSrgb(rawRgb);
}

fn fiRawRgbToLuminance(rawRgb: vec3<f32>) -> f32 {
    let linearRgb = fiRawRgbToLinear(rawRgb);
    // PQ decodes to Rec.2020 primaries; the other two stay Rec.709.
    if (params.backbufferTransferFunction == TRANSFER_FUNCTION_PQ) {
        return dot(linearRgb, REC2020_LUMA_WEIGHTS);
    }

    return dot(linearRgb, REC709_LUMA_WEIGHTS);
}

//
// INPAINTING PYRAMID
//

// AMD builds 13 mips; v1 builds 4 (design note: reduced scope). Mip 0 is half
// the render resolution, so mip `level` is renderSize >> (level + 1).
const FI_INPAINTING_MIP_COUNT: i32 = 4;

fn fiPyramidMipSize(level: i32) -> vec2<i32> {
    return max(params.renderSize >> vec2<u32>(u32(level) + 1u), vec2<i32>(1));
}

// All four mips share one buffer; each starts where the previous one ended.
fn fiPyramidMipOffset(level: i32) -> u32 {
    var offset = 0u;
    for (var i = 0; i < level; i++) {
        let size = fiPyramidMipSize(i);
        offset += u32(size.x * size.y);
    }
    return offset;
}

// The mip this dispatch writes. Clamped rather than trusted: orchestration
// chooses the level, and one past the end of the table would address memory
// belonging to no mip and silently produce wrong pixels rather than fail.
fn fiInpaintingMipLevel() -> i32 {
    // Clamp in u32 space first: converting an out-of-range mip level to i32
    // before clamping is a bit-reinterpretation, so a large value can land
    // negative and slip past a min() taken after the cast.
    return i32(min(params.inpaintingMipLevel, u32(FI_INPAINTING_MIP_COUNT - 1)));
}
