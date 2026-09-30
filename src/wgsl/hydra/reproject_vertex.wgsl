// SDKCore hydra_reproject_distort vertex stage, 0x434b4.
struct Mesh { cells: vec4<u32> }
struct Layer { index: u32 }
struct Transforms { previousToCurrent: mat4x4<f32>, currentToPrevious: mat4x4<f32> }
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) sourceOffset: vec2<f32> }
@group(1) @binding(0) var<uniform> transforms: Transforms;
@group(2) @binding(0) var sceneDepth: texture_2d<f32>;
@group(3) @binding(0) var<uniform> layer: Layer;
@group(3) @binding(1) var<uniform> mesh: Mesh;

@vertex fn main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    let grid = vec2(vertex % (mesh.cells.x + 1u), vertex / (mesh.cells.x + 1u));
    let uv = vec2<f32>(grid) / vec2<f32>(mesh.cells.xy);
    let depthSize = vec2<i32>(textureDimensions(sceneDepth));
    let depth = textureLoad(sceneDepth, clamp(vec2<i32>(uv * vec2<f32>(depthSize)), vec2(0), depthSize - 1), 0).r;
    var transform = transforms.previousToCurrent;
    if (layer.index != 0u) { transform = transforms.currentToPrevious; }
    let projected = transform * vec4(uv * 2.0 - 1.0, depth, 1.0);
    let offset = clamp(uv - (projected.xy * (0.5 / projected.w) + 0.5), vec2(-0.25), vec2(0.25)) * 0.5;
    let interior = (grid > vec2(0u)) & (grid < mesh.cells.xy);
    let midpointUv = select(uv, uv - offset, interior);
    return VertexOutput(vec4(midpointUv.x * 2.0 - 1.0, 1.0 - midpointUv.y * 2.0, 0.0, 1.0), offset);
}
