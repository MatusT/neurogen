// SDKCore hydra_reproject_blur, 0x44850: separable 0.3/0.4/0.3 filter.
// Smooth reprojection error horizontally, then vertically, before resolve uses
// it as a blend weight. Linear filtering makes the 1.2-texel taps fractional.
struct BlurParams {
    horizontal: u32,
    mipLevel: i32,
}

@group(0) @binding(0) var linearSampler: sampler;
@group(1) @binding(0) var matchingError: texture_2d<f32>;
@group(3) @binding(0) var<uniform> params: BlurParams;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    let tapOffset = select(vec2(0.0, 1.2), vec2(1.2, 0.0), params.horizontal != 0u);
    let offset = tapOffset / vec2<f32>(textureDimensions(matchingError, params.mipLevel));
    let level = f32(params.mipLevel);
    return 0.4 * textureSampleLevel(matchingError, linearSampler, uv, level).r +
        0.3 * textureSampleLevel(matchingError, linearSampler, uv + offset, level).r +
        0.3 * textureSampleLevel(matchingError, linearSampler, uv - offset, level).r;
}
