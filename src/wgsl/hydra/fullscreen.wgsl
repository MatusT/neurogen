// SDKCore 0x3cec0: full-screen triangle. UV stays top-left; clip Y is up.
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn main(@builtin(vertex_index) vertex: u32) -> VertexOutput {
    let corners = array(vec2(-1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    let corner = corners[vertex];
    return VertexOutput(vec4(corner.x, -corner.y, 0.0, 1.0), corner * 0.5 + 0.5);
}
