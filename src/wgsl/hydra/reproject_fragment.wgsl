// SDKCore hydra_reproject_distort fragment stage, 0x4337c.
// Rasterization interpolates the mesh's source UV offsets across the midpoint
// image. Matching and resolve recover the source UV as midpoint UV + offset.
@fragment
fn main(@location(0) sourceOffset: vec2<f32>) -> @location(0) vec2<f32> {
    return sourceOffset;
}
