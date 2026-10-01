// SDKCore hydra_match_txt(false)_dmatch(none), 0x4c5e0.
// Compare derivative-expanded luminance intervals after symmetric half-motion warps.
// Outputs an optical matching error and records which source pixels were used.
// The r8unorm target clamps negative interval errors to zero; stored error 0.5
// corresponds to zero optical confidence in resolve.wgsl.
struct MatchParams {
    currentUvScale: f32,
    previousUvScale: f32,
    // Retained for the host uniform layout; this variant does not use matrices.
    unusedPreviousTransform: mat4x4<f32>,
    unusedCurrentTransform: mat4x4<f32>,
}

@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(2) var halfMotion: texture_2d<f32>;
// Multiple matched pixels can claim the same source texel; atomics avoid write races.
@group(0) @binding(3) var<storage, read_write> usedPixels: array<atomic<u32>>;
@group(1) @binding(0) var previousLuma: texture_2d<f32>;
@group(1) @binding(1) var currentLuma: texture_2d<f32>;
@group(3) @binding(0) var<uniform> params: MatchParams;

fn markUsed(pixel: vec2<i32>, size: vec2<i32>, frameBit: u32) {
    if (all(pixel >= vec2(0)) && all(pixel < size)) {
        atomicOr(&usedPixels[u32(pixel.y * size.x + pixel.x)], frameBit);
    }
}

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    // The motion texture has one texel per 8x8 luminance block. Motion itself
    // is measured in luminance pixels, so it can offset the source pixels here.
    let lumaSize = vec2<f32>(textureDimensions(previousLuma));
    let pixel = uv * lumaSize;
    let flowUv = (pixel * 0.125 / vec2<f32>(textureDimensions(halfMotion))) * params.currentUvScale;
    let motion = textureSample(halfMotion, linearSampler, flowUv).xyz;
    let previousPixel = pixel - motion.xy;
    let currentPixel = pixel + motion.xy;
    let previousValue = textureSample(
        previousLuma, linearSampler, previousPixel / lumaSize * params.previousUvScale,
    ).r;
    let currentValue = textureSample(
        currentLuma, linearSampler, currentPixel / lumaSize * params.currentUvScale,
    ).r;

    // Derivatives estimate nearby brightness values. Comparing each sample to
    // the other frame's range tolerates small sampling shifts across an edge.
    let previousMin = min(
        previousValue,
        min(previousValue + dpdx(previousValue), previousValue + dpdy(previousValue)),
    );
    let previousMax = max(
        previousValue,
        max(previousValue + dpdx(previousValue), previousValue + dpdy(previousValue)),
    );
    let currentMin = min(
        currentValue,
        min(currentValue + dpdx(currentValue), currentValue + dpdy(currentValue)),
    );
    let currentMax = max(
        currentValue,
        max(currentValue + dpdx(currentValue), currentValue + dpdy(currentValue)),
    );
    let intervalError = min(
        max(previousValue - currentMax, currentMin - previousValue),
        max(currentValue - previousMax, previousMin - currentValue),
    );

    // Invalid seeds force rejection. A valid match claims both source pixels
    // only below the recovered error threshold, before the output's 0.5 scale.
    let error = select(1.0, 40.0 * intervalError, motion.z >= 1.0);
    if (error < 1.0) {
        markUsed(vec2<i32>(previousPixel), vec2<i32>(lumaSize), 1u);
        markUsed(vec2<i32>(currentPixel), vec2<i32>(lumaSize), 2u);
    }

    return error * 0.5;
}
