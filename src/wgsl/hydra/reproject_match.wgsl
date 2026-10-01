// SDKCore hydra_reproject_match, 0x43fac.
// Score the camera-reprojected midpoint using the same luminance interval test
// as optical matching. This error is blurred before resolve turns it into a
// reprojection weight; stored error 0.5 corresponds to zero confidence.
struct Scales {
    currentUvScale: f32,
    previousUvScale: f32,
}

@group(0) @binding(0) var lumaSampler: sampler;
@group(0) @binding(1) var offsetSampler: sampler;
@group(0) @binding(2) var previousOffset: texture_2d<f32>;
@group(0) @binding(3) var currentOffset: texture_2d<f32>;
@group(1) @binding(0) var previousLuma: texture_2d<f32>;
@group(1) @binding(1) var currentLuma: texture_2d<f32>;
@group(3) @binding(0) var<uniform> scales: Scales;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    // Mesh offsets point from the midpoint UV back to each real frame.
    let previousUv = (uv + textureSample(previousOffset, offsetSampler, uv).xy) * scales.previousUvScale;
    let currentUv = (uv + textureSample(currentOffset, offsetSampler, uv).xy) * scales.currentUvScale;
    let previousValue = textureSample(previousLuma, lumaSampler, previousUv).r;
    let currentValue = textureSample(currentLuma, lumaSampler, currentUv).r;

    // Expand each sample into a local brightness range to tolerate small
    // sampling shifts. A sample inside the other range gives nonpositive error.
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

    // The r8unorm render target clamps negative errors to zero (a good match).
    return intervalError * 40.0 * 0.5;
}
