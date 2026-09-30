// SDKCore hydra_match_txt(false)_dmatch(none), 0x4c5e0.
// Compare derivative-expanded luminance intervals after symmetric half-motion warps.
struct MatchParams {
    currentUvScale: f32,
    previousUvScale: f32,
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

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    let lumaSize = vec2<f32>(textureDimensions(previousLuma));
    let pixel = uv * lumaSize;
    let flowUv = (pixel * 0.125 / vec2<f32>(textureDimensions(halfMotion))) * params.currentUvScale;
    let motion = textureSample(halfMotion, linearSampler, flowUv).xyz;
    let previousPixel = pixel - motion.xy;
    let currentPixel = pixel + motion.xy;
    let a = textureSample(previousLuma, linearSampler, previousPixel / lumaSize * params.previousUvScale).r;
    let b = textureSample(currentLuma, linearSampler, currentPixel / lumaSize * params.currentUvScale).r;
    let aMin = min(a, min(a + dpdx(a), a + dpdy(a)));
    let aMax = max(a, max(a + dpdx(a), a + dpdy(a)));
    let bMin = min(b, min(b + dpdx(b), b + dpdy(b)));
    let bMax = max(b, max(b + dpdx(b), b + dpdy(b)));
    let intervalError = min(max(a - bMax, bMin - a), max(b - aMax, aMin - b));
    let error = select(1.0, 40.0 * intervalError, motion.z >= 1.0);
    if (error < 1.0) {
        markUsed(vec2<i32>(previousPixel), vec2<i32>(lumaSize), 1u);
        markUsed(vec2<i32>(currentPixel), vec2<i32>(lumaSize), 2u);
    }
    return error * 0.5;
}
