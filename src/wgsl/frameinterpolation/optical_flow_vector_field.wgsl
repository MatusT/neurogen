// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_optical_flow_vector_field.h: smooth Task 2's raw
// optical flow over a 3x3 neighbourhood, then scatter it into the interpolated
// frame as priority-packed field entries in the same format the game motion
// vectors use.
//
// This is the pass that adapts Task 2's frozen output to the blend's input.
// The blend samples a *scattered* field — vectors moved to where their content
// will be midway through the frame — which is not what Task 2 produces, and the
// entry format carries a priority the raw vectors have no equivalent of.
//
// Runs at the optical-flow grid resolution, one cell per 8x8 block of render
// pixels, which is where Task 2 leaves it and where the blend samples it.
//
// Dispatch: (ceil(ceil(W/8)/8), ceil(ceil(H/8)/8)). Must run after setup.wgsl.

// @import ./common.wgsl

// Task 2's frozen output: whole render-resolution pixels, current frame ->
// previous frame, one entry per 8x8 block. See ../opticalflow/params.wgsl.
@group(0) @binding(1) var<storage, read> opticalFlowVectorField: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read> opticalFlowValidity: array<u32>;
@group(0) @binding(3) var currentColor: texture_2d<f32>;
@group(0) @binding(4) var previousColor: texture_2d<f32>;
// Two interleaved entries per cell, x then y — see setup.wgsl.
@group(0) @binding(5) var<storage, read_write> opticalFlowMotionVectorField: array<atomic<u32>>;

// Beyond this a block match is far more likely to be a false one than real
// motion, so it stops contributing to its neighbourhood's average.
const MAX_TRUSTED_VELOCITY_PIXELS: f32 = 512.0;
// Below this the block is stationary and its direction is meaningless.
const MIN_MOVING_VELOCITY_PIXELS: f32 = 1.0;
// Motion across this fraction of the frame earns full priority.
const FULL_PRIORITY_VELOCITY_FRACTION: f32 = 0.05;
// Sharpens the agreement weighting so a tap pointing the neighbourhood's way
// dominates one that merely does not disagree.
const DIRECTION_AGREEMENT_EXPONENT: f32 = 1.25;
const LUMA_BLACK_FLOOR: f32 = 0.001;

fn loadFlowUv(pos: vec2<i32>, gridSize: vec2<i32>) -> vec2<f32> {
    let index = fiFlatIndex(pos, gridSize);
    // An invalid cell's vector is arbitrary, so it must not be averaged in.
    // AMD loads a confidence factor at exactly this point and then never
    // applies it; Task 2's validity flag is applied here instead.
    if (opticalFlowValidity[index] == 0u) { return vec2<f32>(0.0); }

    return vec2<f32>(opticalFlowVectorField[index]) / vec2<f32>(params.renderSize);
}

// Long vectors are downweighted rather than rejected, so a genuinely fast
// region still agrees with itself while an isolated wild match is outvoted.
fn velocityWeight(motionVector: vec2<f32>) -> f32 {
    let velocity = length(motionVector * vec2<f32>(params.renderSize));
    return max(0.0, MAX_TRUSTED_VELOCITY_PIXELS - velocity) * f32(velocity > MIN_MOVING_VELOCITY_PIXELS);
}

// Two weighted passes over the 3x3: the first finds the neighbourhood's
// dominant direction, the second keeps only the taps that agree with it.
fn smoothFlow(pos: vec2<i32>, gridSize: vec2<i32>) -> vec2<f32> {
    var average = vec2<f32>(0.0);
    var averageWeightSum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let samplePos = pos + vec2<i32>(x, y);
            if (!fiInBounds(samplePos, gridSize)) { continue; }

            let motionVector = loadFlowUv(samplePos, gridSize);
            let weight = velocityWeight(motionVector);
            average += motionVector * weight;
            averageWeightSum += weight;
        }
    }

    if (averageWeightSum <= 0.0) { return vec2<f32>(0.0); }
    average /= averageWeightSum;

    var flow = vec2<f32>(0.0);
    var weightSum = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let samplePos = pos + vec2<i32>(x, y);
            if (!fiInBounds(samplePos, gridSize)) { continue; }

            let motionVector = loadFlowUv(samplePos, gridSize);
            // AMD clamps after raising the dot product to a fractional power,
            // where a tap pointing the opposite way has already produced a
            // negative base and an undefined result. Clamping first is the
            // same intent without the undefined step.
            let agreement = pow(max(0.0, dot(average, motionVector)), DIRECTION_AGREEMENT_EXPONENT);
            let weight = agreement * velocityWeight(motionVector);
            flow += motionVector * weight;
            weightSum += weight;
        }
    }

    if (weightSum <= FI_EPSILON) { return vec2<f32>(0.0); }
    return flow / weightSum;
}

// Unlike the game vectors, an optical flow vector carries no depth, so its
// priority comes from speed alone: fast content is what the flow is good at and
// what the game vectors most often miss.
fn velocityPriority(motionVector: vec2<f32>) -> u32 {
    let velocity = length(motionVector * vec2<f32>(params.renderSize));
    let fullPriorityVelocity = length(vec2<f32>(params.renderSize) * FULL_PRIORITY_VELOCITY_FRACTION);

    return u32(velocity > MIN_MOVING_VELOCITY_PIXELS)
        * u32(saturate(velocity / fullPriorityVelocity) * f32(MV_FIELD_PRIORITY_HIGH_MAX));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    let gridSize = fiOpticalFlowGridSize();
    if (!fiInBounds(pos, gridSize)) { return; }

    let motionVector = smoothFlow(pos, gridSize);
    let highPriority = velocityPriority(motionVector);
    if (highPriority == 0u) { return; }

    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(gridSize);
    let reprojectedUv = uv + motionVector;
    let previousLuma = LUMA_BLACK_FLOOR + fiRawRgbToLuminance(fiSampleColorClamped(previousColor, reprojectedUv, params.renderSize));
    let currentLuma = LUMA_BLACK_FLOOR + fiRawRgbToLuminance(fiSampleColorClamped(currentColor, uv, params.renderSize));

    let lowPriority = u32(round(fiMinDividedByMax(previousLuma, currentLuma) * f32(MV_FIELD_PRIORITY_LOW_MAX)))
        * u32(fiIsUvInside(reprojectedUv));

    let halfMotionVector = motionVector * 0.5;
    let packed = fiPackVectorField(MV_FIELD_PRIMARY, highPriority, lowPriority, halfMotionVector);

    let bilinear = fiBilinear(uv + halfMotionVector, gridSize);
    for (var i = 0; i < 4; i++) {
        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, gridSize)) { continue; }

        let index = fiFlatIndex(samplePos, gridSize) * 2u;
        atomicMax(&opticalFlowMotionVectorField[index], packed.x);
        atomicMax(&opticalFlowMotionVectorField[index + 1u], packed.y);
    }
}
