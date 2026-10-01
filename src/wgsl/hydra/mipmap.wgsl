// Render-pass replacement for native image blits. The host binds one source mip
// as level 0 and renders into the next smaller mip. Linear sampling filters luma,
// matching errors, and unused-pixel masks; depth uses depth_input.wgsl instead.
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(source, linearSampler, uv, 0.0);
}
