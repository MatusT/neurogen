// Expand atomic match flags into the recovered algorithm's unused-pixel masks.
// match.wgsl sets bit 0 for previous-frame texels and bit 1 for current-frame
// texels. Here 1 means unused and 0 means claimed by a reliable optical match.
// Resolve samples filtered mips of these masks to weight the unwarped fallback.
@group(0) @binding(0) var<storage, read> usedPixels: array<u32>;
@group(0) @binding(1) var previousUsage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var currentUsage: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let size = textureDimensions(previousUsage);
    if (any(invocation.xy >= size)) {
        return;
    }

    let flags = usedPixels[invocation.y * size.x + invocation.x];
    let pixel = vec2<i32>(invocation.xy);
    textureStore(previousUsage, pixel, vec4(select(1.0, 0.0, (flags & 1u) != 0u)));
    textureStore(currentUsage, pixel, vec4(select(1.0, 0.0, (flags & 2u) != 0u)));
}
