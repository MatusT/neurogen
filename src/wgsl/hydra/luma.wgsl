// SDKCore compute_luma_no_ui, 0x3dd2c.
@group(0) @binding(0) var linearSampler: sampler;
@group(1) @binding(0) var sceneColor: texture_2d<f32>;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    return dot(textureSample(sceneColor, linearSampler, uv).rgb, vec3(0.299, 0.587, 0.114));
}
