// Puts a frame on the canvas. Needed because the interpolated frame is
// rgba16float — a format no ordinary canvas can be configured as — so it has to
// be sampled into the swap chain rather than copied. The real frames go through
// the same pass so that toggling changes only which texture is read.

const SHADER = /* wgsl */ `
@group(0) @binding(0) var source: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  // One oversized triangle covering the frame, no vertex buffer.
  let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
  return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

// The scene, and therefore the module's whole graph, works in linear light —
// TransferFunction.LinearLdr. The swap chain is displayed as sRGB.
const DISPLAY_GAMMA = 2.2;

@fragment
fn fragmentMain(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let linear = textureLoad(source, vec2<i32>(position.xy), 0).rgb;
  return vec4<f32>(pow(linear, vec3<f32>(1.0 / DISPLAY_GAMMA)), 1.0);
}
`;

export class Presenter {
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;

    const module = device.createShaderModule({ label: "demo-present", code: SHADER });
    this.pipeline = device.createRenderPipeline({
      label: "demo-present",
      layout: "auto",
      vertex: { module, entryPoint: "vertexMain" },
      fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
    });
  }

  // `source` must be the same size as the canvas: the fragment reads it by
  // pixel coordinate rather than sampling, so there is no filtering to get
  // wrong and no size mismatch to hide.
  present(target: GPUTextureView, source: GPUTextureView): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: source }],
    });

    const encoder = this.device.createCommandEncoder({ label: "demo-present" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: "clear", clearValue: [0, 0, 0, 1], storeOp: "store" }],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
