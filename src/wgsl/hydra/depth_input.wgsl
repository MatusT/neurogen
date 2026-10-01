// Float-depth equivalent of split_zs_blit + depth_mipmap_conversion (average).
// Produces the depth pyramid sampled by the camera-reprojection mesh.
// Interpolation does not consume stencil. Only the input stage quantizes to D24;
// subsequent levels average the already converted float depth.
struct Params {
    quantize: u32,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const D24_MAX: f32 = 16777215.0; // 2^24 - 1.

fn loadDepth(pixel: vec2<i32>) -> f32 {
    // Repeat edge texels when the reduction footprint extends outside the image.
    let sourceSize = vec2<i32>(textureDimensions(source));
    let depth = textureLoad(source, clamp(pixel, vec2(0), sourceSize - 1), 0).x;
    if (params.quantize != 0u) {
        return floor(clamp(depth, 0.0, 1.0) * D24_MAX) / D24_MAX;
    }
    return depth;
}

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) f32 {
    // Average the 2x2 source footprint around this destination pixel's center.
    let pixel = vec2<i32>(floor(uv * vec2<f32>(textureDimensions(source)) - 0.5));
    return (
        loadDepth(pixel) + loadDepth(pixel + vec2(1, 0)) +
        loadDepth(pixel + vec2(0, 1)) + loadDepth(pixel + vec2(1, 1))
    ) * 0.25;
}
