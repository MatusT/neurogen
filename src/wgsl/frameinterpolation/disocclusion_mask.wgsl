// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_disocclusion_mask.h: for each pixel of the frame being
// interpolated, decide whether that pixel's surface is actually visible in the
// previous frame (.x) and in the current frame (.y).
//
// A pixel that fails both is content neither source frame contains — it was
// hidden behind something that has since moved. Nothing can be warped into it,
// so the blend hands it to inpainting instead.
//
// Two of the three formulas in AMD's file are dead code: `ComputeSampleDepthClip`
// is defined but never called, and `ComputeDepthClip` computes a resolution
// factor and a power it never uses. Only the live path is ported.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution. Must run after
// game_motion_vector_field.wgsl and reconstruct_and_dilate.wgsl.

// @import ./common.wgsl

@group(0) @binding(1) var<storage, read> reconstructedDepthInterpolated: array<u32>;
@group(0) @binding(2) var<storage, read> reconstructedDepthPrevious: array<u32>;
@group(0) @binding(3) var<storage, read> dilatedDepth: array<f32>;
@group(0) @binding(4) var<storage, read> gameMotionVectorField: array<u32>;
// .x: visible in the previous frame. .y: visible in the current frame. Both
// binarised to exactly 0 or 1.
@group(0) @binding(5) var<storage, read_write> disocclusionMask: array<vec2<f32>>;

// "Minimum Triangle Separation for Correct Z-Buffer Occlusion", section 4: the
// smallest view-space gap at which two surfaces are reliably distinguishable
// rather than z-fighting artefacts of the same surface.
const KSEP: f32 = 1.37e-05;
const DEPTH_BILINEAR_WEIGHT_THRESHOLD: f32 = FI_EPSILON;

// Which frame's depth a comparison is against.
const ESTIMATE_PREVIOUS_FRAME: u32 = 0u;
const ESTIMATE_CURRENT_FRAME: u32 = 1u;

fn loadEstimatedDepth(estimate: u32, pos: vec2<i32>) -> f32 {
    let index = fiFlatIndex(pos, params.renderSize);
    if (estimate == ESTIMATE_PREVIOUS_FRAME) {
        return bitcast<f32>(reconstructedDepthPrevious[index]);
    }

    return dilatedDepth[index];
}

fn loadGameFieldMv(uv: vec2<f32>) -> FiVectorFieldEntry {
    let pos = vec2<i32>(uv * vec2<f32>(params.renderSize));
    if (!fiInBounds(pos, params.renderSize)) { return FiVectorFieldEntry(); }

    let index = fiFlatIndex(pos, params.renderSize) * 2u;
    var entry = fiUnpackVectorField(vec2<u32>(gameMotionVectorField[index], gameMotionVectorField[index + 1u]));
    fiAnnotateReprojection(&entry, uv);
    return entry;
}

// The field of view's contribution to the separation threshold. AMD derives it
// as the ratio of the view-space distances to a corner and to the centre of the
// screen at some plane depth; both scale linearly with that depth, so it cancels
// and the ratio is this constant.
fn kFov() -> f32 {
    let tanHalfFov = fiTanHalfFov();
    return sqrt(dot(tanHalfFov, tanHalfFov) + 1.0);
}

// The fraction of this pixel's reprojected footprint that lands on the *same*
// surface it started on. Zero means every part of the footprint is now behind
// something else, i.e. fully disoccluded.
fn computeDepthClip(estimate: u32, uvSample: vec2<f32>, currentDepth: f32) -> f32 {
    let currentDepthViewSpace = fiViewSpaceDepth(currentDepth);
    let bilinear = fiBilinear(uvSample, params.renderSize);
    // AMD names this half the viewport width; it is the full diagonal.
    let viewportDiagonal = length(vec2<f32>(params.renderSize));

    var sameSurface = 0.0;
    var weightSum = 0.0;
    for (var i = 0; i < 4; i++) {
        let weight = bilinear.weights[i];
        if (weight <= DEPTH_BILINEAR_WEIGHT_THRESHOLD) { continue; }

        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, params.renderSize)) { continue; }

        let estimatedDepthViewSpace = fiViewSpaceDepth(loadEstimatedDepth(estimate, samplePos));
        let depthDifference = currentDepthViewSpace - estimatedDepthViewSpace;
        // Only something *in front* of this pixel can occlude it; a tap behind
        // it says nothing and is left out of the weight sum entirely.
        if (depthDifference <= 0.0) { continue; }

        let requiredSeparation = KSEP * kFov() * viewportDiagonal * min(currentDepthViewSpace, estimatedDepthViewSpace);
        sameSurface += f32(requiredSeparation >= depthDifference) * weight;
        weightSum += weight;
    }

    // Nothing in front anywhere: not occluded, so nothing is clipped.
    if (weightSum <= 0.0) { return 0.0; }

    return saturate(1.0 - sameSurface / weightSum);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (!fiInBounds(pos, params.renderSize)) { return; }

    let index = fiFlatIndex(pos, params.renderSize);
    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(params.renderSize);
    let depth = bitcast<f32>(reconstructedDepthInterpolated[index]);
    let gameMv = loadGameFieldMv(uv);

    let visibleInPrevious = 1.0 - computeDepthClip(ESTIMATE_PREVIOUS_FRAME, uv + gameMv.motionVector, depth);
    let visibleInCurrent = 1.0 - computeDepthClip(ESTIMATE_CURRENT_FRAME, uv - gameMv.motionVector, depth);

    // Binarised: the blend only ever asks whether a source is fully usable, and
    // a partial answer here would leak a soft edge into the colour.
    var mask = vec2<f32>(vec2<f32>(visibleInPrevious, visibleInCurrent) >= vec2<f32>(FI_EPSILON));

    // A reprojection that leaves the frame has no depth to compare against, so
    // the test above reports a disocclusion that is really just an edge. Treat
    // it as visible and let the bounds checks in the blend handle it.
    let sourceMotionVector = gameMv.motionVector * 2.0;
    let previousPos = vec2<i32>((uv + sourceMotionVector) * vec2<f32>(params.renderSize));
    mask.x = saturate(mask.x + f32(!fiInBounds(previousPos, params.renderSize)));

    let currentPos = vec2<i32>((uv - sourceMotionVector) * vec2<f32>(params.renderSize));
    mask.y = saturate(mask.y + f32(!fiInBounds(currentPos, params.renderSize)));

    disocclusionMask[index] = mask;
}
