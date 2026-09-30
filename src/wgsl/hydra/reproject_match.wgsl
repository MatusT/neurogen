// SDKCore hydra_reproject_match, 0x43fac.
struct Scales { currentUvScale: f32, previousUvScale: f32 }
@group(0) @binding(0) var lumaSampler: sampler;
@group(0) @binding(1) var offsetSampler: sampler;
@group(0) @binding(2) var previousOffset: texture_2d<f32>;
@group(0) @binding(3) var currentOffset: texture_2d<f32>;
@group(1) @binding(0) var previousLuma: texture_2d<f32>;
@group(1) @binding(1) var currentLuma: texture_2d<f32>;
@group(3) @binding(0) var<uniform> scales: Scales;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    let previousUv = (uv + textureSample(previousOffset, offsetSampler, uv).xy) * scales.previousUvScale;
    let currentUv = (uv + textureSample(currentOffset, offsetSampler, uv).xy) * scales.currentUvScale;
    let a = textureSample(previousLuma, lumaSampler, previousUv).r;
    let b = textureSample(currentLuma, lumaSampler, currentUv).r;
    let aMin = min(a, min(a + dpdx(a), a + dpdy(a)));
    let aMax = max(a, max(a + dpdx(a), a + dpdy(a)));
    let bMin = min(b, min(b + dpdx(b), b + dpdy(b)));
    let bMax = max(b, max(b + dpdx(b), b + dpdy(b)));
    return min(max(a - bMax, bMin - a), max(b - aMax, aMin - b)) * 40.0 * 0.5;
}
