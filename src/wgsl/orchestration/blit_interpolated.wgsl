// Original work, not a port of AMD code.
//
// Copies the frame interpolation module's output buffer into the texture the
// library hands back. The module writes `array<vec4<f32>>` rather than a
// texture because WebGPU has no atomics on storage textures and the inpainting
// pyramid has to be both read and written, so something has to make the
// crossing.
//
// A copyBufferToTexture would be the obvious way and does not work in general:
// a multi-row copy needs bytesPerRow to be a multiple of 256, the buffer's rows
// are tightly packed at 16 bytes per pixel, and the shaders index it as
// `y * renderSize.x + x` with no padding to spare. That restricts the direct
// copy to widths that are a multiple of 16. This pass has no such constraint.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution, after final_blend.

struct BlitParams {
    renderSize: vec2<i32>,
}

@group(0) @binding(0) var<uniform> params: BlitParams;
@group(0) @binding(1) var<storage, read> interpolatedColor: array<vec4<f32>>;
// The format is fixed here because WGSL bakes it into the type. rgba16float
// holds the module's HDR range, which the 8-bit formats would clip.
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (any(pos >= params.renderSize)) { return; }

    let index = u32(pos.y * params.renderSize.x + pos.x);
    textureStore(outputTexture, pos, interpolatedColor[index]);
}
