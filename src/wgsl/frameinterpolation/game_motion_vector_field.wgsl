// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_game_motion_vector_field.h and
// ffx_frameinterpolation_reconstruct_previous_depth.h. Both scatter from the
// same dilated inputs into disjoint targets, so they run as one dispatch here
// rather than AMD's two.
//
// Every pixel pushes its half-frame motion vector into the interpolated frame's
// grid, and the same pixel's depth into the interpolated frame's depth. Both are
// scatters, not gathers: the destination is where the pixel *will be* midway
// through the frame, which no destination pixel can work out on its own. The
// collisions that follow are resolved by an atomic on the priority-packed entry
// — nearest surface wins the motion vector, nearest depth wins the depth.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution. Must run after
// setup.wgsl and reconstruct_and_dilate.wgsl.

// @import ./common.wgsl

@group(0) @binding(1) var<storage, read> dilatedDepth: array<f32>;
@group(0) @binding(2) var<storage, read> dilatedMotionVectors: array<vec2<f32>>;
@group(0) @binding(3) var currentColor: texture_2d<f32>;
@group(0) @binding(4) var previousColor: texture_2d<f32>;
// Two interleaved entries per pixel, x then y — see setup.wgsl.
@group(0) @binding(5) var<storage, read_write> gameMotionVectorField: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> reconstructedDepthInterpolated: array<atomic<u32>>;

const RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD: f32 = FI_EPSILON;
// Once four primary vectors have been passed over, the trail is running
// through content that already has real motion and stops contributing.
const SECONDARY_MAX_PRIMARY_HITS: u32 = 3u;
// Diagonal of the UV square: no secondary trail can usefully be longer.
const SECONDARY_MAX_UV_DISTANCE: f32 = 0.70710678;
// Keeps the luma ratio meaningful where both frames are near black.
const LUMA_BLACK_FLOOR: f32 = 0.001;
// AMD's depth compression exponent — a cube root, so distant surfaces still
// separate from each other instead of all saturating to the same priority.
const DEPTH_PRIORITY_EXPONENT: f32 = 0.33;

// Depth priority, near surfaces highest. AMD writes this as `1 - d / (1 + d)`,
// which is `1 / (1 + d)`. Never zero, because zero is what an unwritten cell
// reads back as and would make a real vector indistinguishable from an empty
// one.
fn priorityFromViewSpaceDepth(viewSpaceDepth: f32) -> u32 {
    let compressed = pow(viewSpaceDepth, DEPTH_PRIORITY_EXPONENT);
    return max(1u, u32(f32(MV_FIELD_PRIORITY_HIGH_MAX) / (1.0 + compressed)));
}

// Returns the entry that was already there, so the caller can see whether it
// just wrote over a primary vector.
fn updateGameField(pos: vec2<i32>, packed: vec2<u32>) -> u32 {
    let index = fiFlatIndex(pos, params.renderSize) * 2u;
    let previousX = atomicMax(&gameMotionVectorField[index], packed.x);
    let previousY = atomicMax(&gameMotionVectorField[index + 1u], packed.y);

    return max(previousX, previousY);
}

// How well this pixel's colour survives its own reprojection — a vector that
// lands on a differently lit pixel is the weaker candidate when two entries tie
// on depth.
fn colorAgreementPriority(uv: vec2<f32>, motionVector: vec2<f32>) -> u32 {
    let reprojectedUv = uv + motionVector;
    let previousLuma = LUMA_BLACK_FLOOR + fiRawRgbToLuminance(fiSampleColorClamped(previousColor, reprojectedUv, params.renderSize));
    let currentLuma = LUMA_BLACK_FLOOR + fiRawRgbToLuminance(fiSampleColorClamped(currentColor, uv, params.renderSize));

    let agreement = fiMinDividedByMax(previousLuma, currentLuma);
    return u32(round(agreement * f32(MV_FIELD_PRIORITY_LOW_MAX))) * u32(fiIsUvInside(reprojectedUv));
}

// Secondary vectors trail backwards along the motion from the primary landing
// site. They exist to give the regions a moving object is about to uncover
// something better than nothing to fall back on, so their depth priority is
// inverted: the atomic keeps the furthest of them, which is the background.
fn writeSecondaryVectors(interpolatedUv: vec2<f32>, motionVector: vec2<f32>, halfMotionVector: vec2<f32>, highPriorityPrimary: u32) {
    let stepScale = length(1.0 / vec2<f32>(params.renderSize));
    let stepDirection = normalize(motionVector);
    let breakDistance = min(length(halfMotionVector), SECONDARY_MAX_UV_DISTANCE);
    let highPriority = max(1u, MV_FIELD_PRIORITY_HIGH_MAX - highPriorityPrimary);

    var primaryHits = 0u;
    for (var distance = stepScale; distance <= breakDistance; distance += stepScale) {
        let secondaryUv = interpolatedUv - stepDirection * distance;

        // Trailing further from the frame centre than towards it means the
        // vector is heading off screen, where it can never be read.
        let towardsCentre = normalize(vec2<f32>(0.5) - secondaryUv);
        let lowPriority = u32(max(0.0, dot(towardsCentre, stepDirection)) * f32(MV_FIELD_PRIORITY_LOW_MAX));
        let packed = fiPackVectorField(MV_FIELD_SECONDARY, highPriority, lowPriority, halfMotionVector);

        // Only the first bilinear tap: a secondary vector is a hint, and
        // spreading it over the full quad would let it outvote real vectors.
        let samplePos = fiBilinear(secondaryUv, params.renderSize).basePos;
        if (!fiInBounds(samplePos, params.renderSize)) { return; }

        primaryHits += u32(fiPackedEntryIsPrimary(updateGameField(samplePos, packed)));
        if (primaryHits > SECONDARY_MAX_PRIMARY_HITS) { return; }
    }
}

// The interpolated frame's depth, by the same scatter the motion vectors use:
// half the motion vector, nearest depth wins.
fn reconstructInterpolatedDepth(uv: vec2<f32>, depth: f32, halfMotionVector: vec2<f32>) {
    let bilinear = fiBilinear(uv + halfMotionVector, params.renderSize);

    for (var i = 0; i < 4; i++) {
        if (bilinear.weights[i] <= RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD) { continue; }

        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, params.renderSize)) { continue; }

        atomicMin(&reconstructedDepthInterpolated[fiFlatIndex(samplePos, params.renderSize)], bitcast<u32>(depth));
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (!fiInBounds(pos, params.renderSize)) { return; }

    let index = fiFlatIndex(pos, params.renderSize);
    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(params.renderSize);
    let depth = dilatedDepth[index];
    let motionVector = dilatedMotionVectors[index];

    // The field holds half-frame vectors throughout: the interpolated frame
    // sits midway, so a consumer reaches the previous frame by adding the
    // entry and the current frame by subtracting it.
    let halfMotionVector = motionVector * 0.5;
    let interpolatedUv = uv + halfMotionVector;

    let highPriority = priorityFromViewSpaceDepth(fiViewSpaceDepth(depth));
    let lowPriority = colorAgreementPriority(uv, motionVector);
    let packedPrimary = fiPackVectorField(MV_FIELD_PRIMARY, highPriority, lowPriority, halfMotionVector);

    let bilinear = fiBilinear(interpolatedUv, params.renderSize);
    for (var i = 0; i < 4; i++) {
        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, params.renderSize)) { continue; }

        updateGameField(samplePos, packedPrimary);
    }

    reconstructInterpolatedDepth(uv, depth, halfMotionVector);

    // normalize() of a zero vector is undefined, and a stationary pixel has no
    // trail to leave behind anyway.
    if (length(halfMotionVector * vec2<f32>(params.renderSize)) > FI_EPSILON) {
        writeSecondaryVectors(interpolatedUv, motionVector, halfMotionVector, highPriority);
    }
}
