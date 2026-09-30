// Float-depth equivalent of split_zs_blit + depth_mipmap_conversion (average).
// Interpolation does not consume stencil. The first stage retains D24 quantization.
struct Params { quantize: u32 }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

fn loadDepth(p: vec2<i32>) -> f32 {
    let depth = textureLoad(source, clamp(p, vec2(0), vec2<i32>(textureDimensions(source)) - 1), 0).x;
    if (params.quantize != 0u) {
        return floor(clamp(depth, 0.0, 1.0) * 16777215.0) / 16777215.0;
    }
    return depth;
}

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    let p = vec2<i32>(floor(uv * vec2<f32>(textureDimensions(source)) - 0.5));
    return (loadDepth(p) + loadDepth(p + vec2(1, 0)) +
            loadDepth(p + vec2(0, 1)) + loadDepth(p + vec2(1, 1))) * 0.25;
}
