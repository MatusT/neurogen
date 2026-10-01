// SDKCore hydra_reproject_distort vertex stage, 0x434b4.
// Warp a depth-sampled grid halfway toward the other frame. Drawing one mesh per
// frame produces the two midpoint-to-source UV offset textures used by resolve.
struct Mesh {
    cells: vec4<u32>, // XY: grid cell counts; ZW: retained uniform padding.
}

struct Layer {
    index: u32, // 0: previous frame; 1: current frame.
}

struct Transforms {
    previousToCurrent: mat4x4<f32>,
    currentToPrevious: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) sourceOffset: vec2<f32>,
}

@group(1) @binding(0) var<uniform> transforms: Transforms;
@group(2) @binding(0) var sceneDepth: texture_2d<f32>;
@group(3) @binding(0) var<uniform> layer: Layer;
@group(3) @binding(1) var<uniform> mesh: Mesh;

@vertex
fn main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    // The indexed mesh has one more vertex than cells along each axis.
    let grid = vec2(vertex % (mesh.cells.x + 1u), vertex / (mesh.cells.x + 1u));
    let uv = vec2<f32>(grid) / vec2<f32>(mesh.cells.xy);
    let depthSize = vec2<i32>(textureDimensions(sceneDepth));
    let depthPixel = clamp(vec2<i32>(uv * vec2<f32>(depthSize)), vec2(0), depthSize - 1);
    let depth = textureLoad(sceneDepth, depthPixel, 0).r;

    // The host converts the supplied matrices to the shader's Y-down convention.
    var transform = transforms.previousToCurrent;
    if (layer.index != 0u) {
        transform = transforms.currentToPrevious;
    }
    let projected = transform * vec4(uv * 2.0 - 1.0, depth, 1.0);
    let projectedUv = projected.xy * (0.5 / projected.w) + 0.5;

    // Limit full-frame displacement before taking half for t=0.5. The sign
    // makes the stored offset point from the midpoint back to this source frame.
    let sourceOffset = clamp(uv - projectedUv, vec2(-0.25), vec2(0.25)) * 0.5;

    // Pin each boundary coordinate to the viewport edge to keep mesh coverage.
    let interior = (grid > vec2(0u)) & (grid < mesh.cells.xy);
    let midpointUv = select(uv, uv - sourceOffset, interior);

    // Convert back to WebGPU's Y-up clip coordinates for rasterization.
    return VertexOutput(
        vec4(midpointUv.x * 2.0 - 1.0, 1.0 - midpointUv.y * 2.0, 0.0, 1.0),
        sourceOffset,
    );
}
