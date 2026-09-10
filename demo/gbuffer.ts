// The demo's renderer: one multi-target pass that produces exactly the four
// inputs FrameGenerator asks for — this frame's colour, device depth and motion
// vectors, plus last frame's colour, kept by double-buffering the colour target.
//
// The motion vectors are the scene's own: the vertex stage projects each vertex
// twice, with this frame's model matrix and the previous frame's, and the
// fragment differences the two in pixels. Nothing is estimated from the image.

import {
  CUBE_FLOATS_PER_VERTEX,
  CUBE_VERTEX_COUNT,
  cubeVertices,
  FLOATS_PER_INSTANCE,
  packInstances,
  type Instance,
} from "./scene.js";
import { perspective, type Mat4, type Projection } from "./math.js";

// Half floats are plenty for colour, and the interpolated frame the library
// hands back is rgba16float too, so presentation reads one format.
const COLOR_FORMAT: GPUTextureFormat = "rgba16float";
// Depth travels as colour because the module samples it as texture_2d<f32>.
// Full floats: device depth crowds towards 1 near the far plane, where f16
// steps are coarse enough to make the disocclusion mask's view-space
// comparisons noise.
const DEPTH_COLOR_FORMAT: GPUTextureFormat = "r32float";
const MOTION_FORMAT: GPUTextureFormat = "rg16float";
const DEPTH_ATTACHMENT_FORMAT: GPUTextureFormat = "depth24plus";

const BYTES_PER_F32 = 4;
// mat4x4 + mat4x4 + vec3 + f32.
const INSTANCE_BYTES = FLOATS_PER_INSTANCE * BYTES_PER_F32;
const MAX_INSTANCES = 16;
// mat4x4<f32> + vec2<f32>, padded to the struct's 16-byte alignment.
const CAMERA_BYTES = 80;

const FAR_PLANE_DEPTH = 1;

const SHADER = /* wgsl */ `
struct Camera {
  viewProjection: mat4x4<f32>,
  viewport: vec2<f32>,
};

struct Instance {
  model: mat4x4<f32>,
  previousModel: mat4x4<f32>,
  color: vec3<f32>,
  patternScale: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

struct Varyings {
  @builtin(position) position: vec4<f32>,
  @location(0) clip: vec4<f32>,
  @location(1) previousClip: vec4<f32>,
  @location(2) normal: vec3<f32>,
  @location(3) local: vec3<f32>,
  @location(4) @interpolate(flat) instance: u32,
};

@vertex
fn vertexMain(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @builtin(instance_index) instance: u32,
) -> Varyings {
  let body = instances[instance];
  let clip = camera.viewProjection * body.model * vec4<f32>(position, 1.0);

  var out: Varyings;
  out.position = clip;
  // Carried as a varying as well: the fragment needs the perspective-correct
  // clip position of both frames to difference them in screen space.
  out.clip = clip;
  out.previousClip = camera.viewProjection * body.previousModel * vec4<f32>(position, 1.0);
  out.normal = (body.model * vec4<f32>(normal, 0.0)).xyz;
  out.local = position;
  out.instance = instance;

  return out;
}

struct Targets {
  @location(0) color: vec4<f32>,
  @location(1) depth: vec4<f32>,
  @location(2) motion: vec4<f32>,
};

const LIGHT_DIRECTION = vec3<f32>(0.4014, 0.6221, 0.6723);
const AMBIENT = 0.35;
const DETAIL_FLOOR = 0.72;

// Per-cell value with no periodicity, fixed to object space so it travels with
// the body rather than swimming across it. Optical flow block matching needs
// detail that is stable frame to frame to lock onto.
fn hash(cell: vec3<f32>) -> f32 {
  return fract(sin(dot(cell, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

// The render target's own pixels, y down — the space the module's motion vector
// contract is stated in.
fn toPixel(clip: vec4<f32>) -> vec2<f32> {
  let ndc = clip.xy / clip.w;
  return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * camera.viewport;
}

@fragment
fn fragmentMain(vary: Varyings) -> Targets {
  let body = instances[vary.instance];
  let lit = AMBIENT + (1.0 - AMBIENT) * max(dot(normalize(vary.normal), LIGHT_DIRECTION), 0.0);
  let detail = mix(DETAIL_FLOOR, 1.0, hash(floor(vary.local * body.patternScale)));

  var out: Targets;
  out.color = vec4<f32>(body.color * lit * detail, 1.0);
  // Device depth of whichever surface survived the depth test.
  out.depth = vec4<f32>(vary.position.z, 0.0, 0.0, 1.0);
  // From this pixel to where its surface was in the previous frame.
  out.motion = vec4<f32>(toPixel(vary.previousClip) - toPixel(vary.clip), 0.0, 1.0);

  return out;
}
`;

function renderTarget(
  device: GPUDevice,
  size: readonly [number, number],
  format: GPUTextureFormat,
  label: string,
): GPUTexture {
  return device.createTexture({
    label,
    size: [size[0], size[1]],
    format,
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });
}

export class GBuffer {
  private readonly device: GPUDevice;
  private readonly size: readonly [number, number];
  private readonly pipeline: GPURenderPipeline;
  private readonly camera: GPUBuffer;
  private readonly instances: GPUBuffer;
  private readonly vertices: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly colors: readonly [GPUTexture, GPUTexture];
  private readonly depth: GPUTexture;
  private readonly motion: GPUTexture;
  private readonly depthAttachment: GPUTexture;
  private current = 0;

  constructor(device: GPUDevice, size: readonly [number, number], projection: Projection) {
    this.device = device;
    this.size = size;

    const module = device.createShaderModule({ label: "demo-gbuffer", code: SHADER });
    this.pipeline = device.createRenderPipeline({
      label: "demo-gbuffer",
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vertexMain",
        buffers: [
          {
            arrayStride: CUBE_FLOATS_PER_VERTEX * BYTES_PER_F32,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 3 * BYTES_PER_F32, format: "float32x3" },
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fragmentMain",
        targets: [{ format: COLOR_FORMAT }, { format: DEPTH_COLOR_FORMAT }, { format: MOTION_FORMAT }],
      },
      // Opaque bodies behind a depth test look the same either way, and no
      // winding convention can then be got wrong.
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: DEPTH_ATTACHMENT_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const vertices = cubeVertices();
    this.vertices = device.createBuffer({
      label: "demo-cube",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertices, 0, vertices);

    this.camera = device.createBuffer({
      label: "demo-camera",
      size: CAMERA_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.camera, 0, this.packCamera(perspective(projection)));

    this.instances = device.createBuffer({
      label: "demo-instances",
      size: MAX_INSTANCES * INSTANCE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camera } },
        { binding: 1, resource: { buffer: this.instances } },
      ],
    });

    this.colors = [
      renderTarget(device, size, COLOR_FORMAT, "demo-color-a"),
      renderTarget(device, size, COLOR_FORMAT, "demo-color-b"),
    ];
    this.depth = renderTarget(device, size, DEPTH_COLOR_FORMAT, "demo-depth");
    this.motion = renderTarget(device, size, MOTION_FORMAT, "demo-motion");
    this.depthAttachment = device.createTexture({
      label: "demo-depth-attachment",
      size: [size[0], size[1]],
      format: DEPTH_ATTACHMENT_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  // Promotes the colour target to `previousColor` and draws the next real frame
  // into `currentColor`. Depth and motion vectors describe only the new frame,
  // which is all the module's contract asks for, so they are not doubled.
  renderFrame(instances: readonly Instance[]): void {
    if (instances.length > MAX_INSTANCES) {
      throw new Error(`scene has ${instances.length} instances, past the ${MAX_INSTANCES} allocated`);
    }

    this.current ^= 1;
    this.device.queue.writeBuffer(this.instances, 0, packInstances(instances));

    const encoder = this.device.createCommandEncoder({ label: "demo-gbuffer" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        this.attachment(this.colors[this.current], [0, 0, 0, 1]),
        // Cleared to the far plane: a pixel no geometry covered must not read
        // back as the near plane, which the disocclusion mask would take for a
        // surface right in front of the camera.
        this.attachment(this.depth, [FAR_PLANE_DEPTH, 0, 0, 1]),
        this.attachment(this.motion, [0, 0, 0, 1]),
      ],
      depthStencilAttachment: {
        view: this.depthAttachment.createView(),
        depthClearValue: FAR_PLANE_DEPTH,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertices);
    pass.draw(CUBE_VERTEX_COUNT, instances.length);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  get currentColor(): GPUTexture {
    return this.colors[this.current];
  }

  get previousColor(): GPUTexture {
    return this.colors[this.current ^ 1];
  }

  // Exactly FrameGenerator's input set, in the encodings its contract names.
  frameInputs(): {
    currentColor: GPUTextureView;
    previousColor: GPUTextureView;
    depth: GPUTextureView;
    motionVectors: GPUTextureView;
  } {
    return {
      currentColor: this.currentColor.createView(),
      previousColor: this.previousColor.createView(),
      depth: this.depth.createView(),
      motionVectors: this.motion.createView(),
    };
  }

  destroy(): void {
    for (const color of this.colors) {
      color.destroy();
    }
    this.depth.destroy();
    this.motion.destroy();
    this.depthAttachment.destroy();
    this.camera.destroy();
    this.instances.destroy();
    this.vertices.destroy();
  }

  private attachment(texture: GPUTexture, clearValue: readonly number[]): GPURenderPassColorAttachment {
    return {
      view: texture.createView(),
      clearValue: [...clearValue],
      loadOp: "clear",
      storeOp: "store",
    };
  }

  private packCamera(viewProjection: Mat4): Float32Array<ArrayBuffer> {
    const packed = new Float32Array(CAMERA_BYTES / BYTES_PER_F32);
    packed.set(viewProjection, 0);
    packed.set(this.size, 16);

    return packed;
  }
}
