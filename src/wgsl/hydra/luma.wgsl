// SDKCore compute_luma_no_ui, 0x3dd2c.
// Reduce scene color to one brightness channel for motion search and matching.
// The host renders this at half resolution, then builds its luminance pyramid.
@group(0) @binding(0) var linearSampler: sampler;
@group(1) @binding(0) var sceneColor: texture_2d<f32>;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    // Preserve the recovered RGB weights for the linear LDR input contract.
    let color = textureSample(sceneColor, linearSampler, uv).rgb;
    return dot(color, vec3(0.299, 0.587, 0.114));
}
