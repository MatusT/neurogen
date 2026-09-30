import fullscreen from "../wgsl/generated/hydra/fullscreen.wgsl.js";
import luma from "../wgsl/generated/hydra/luma.wgsl.js";
import search from "../wgsl/generated/hydra/search.wgsl.js";
import match from "../wgsl/generated/hydra/match.wgsl.js";
import usageMasks from "../wgsl/generated/hydra/usage_masks.wgsl.js";
import reprojectVertex from "../wgsl/generated/hydra/reproject_vertex.wgsl.js";
import reprojectFragment from "../wgsl/generated/hydra/reproject_fragment.wgsl.js";
import reprojectMatch from "../wgsl/generated/hydra/reproject_match.wgsl.js";
import reprojectBlur from "../wgsl/generated/hydra/reproject_blur.wgsl.js";
import resolve from "../wgsl/generated/hydra/resolve.wgsl.js";
import mipmap from "../wgsl/generated/hydra/mipmap.wgsl.js";
import depthInput from "../wgsl/generated/hydra/depth_input.wgsl.js";
import type { FrameGeneratorConfig, FrameGeneratorOptions } from "./FrameGenerator.js";
import { TransferFunction } from "./params.js";

export interface HydraFrameInputs {
  previousColor: GPUTextureView;
  currentColor: GPUTextureView;
  // Single-sampled float texture containing WebGPU device depth in R, [0, 1].
  depth: GPUTextureView;
  previousDepth?: GPUTextureView;
  // Column-major transforms between WebGPU clip spaces. Supply both or neither.
  // Omission means a stationary camera, not automatic camera estimation.
  previousToCurrentClip?: readonly number[] | Float32Array;
  currentToPreviousClip?: readonly number[] | Float32Array;
  resetHistory?: boolean;
}

type Pipeline = GPUComputePipeline | GPURenderPipeline;
type Bindings = readonly (readonly [number, GPUBindingResource])[];
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const BLOCK = 8;

/** Recovered SDKCore BASIC/Hydra interpolation, supplied-camera-transform path. */
export class HydraFrameGenerator {
  readonly device: GPUDevice;
  private resources: (GPUTexture | GPUBuffer)[] = [];
  private pipelines = new Map<string, Pipeline>();
  private uniforms = new Map<string, GPUBuffer>();
  private linear!: GPUSampler;
  private nearest!: GPUSampler;
  private colors!: [GPUTexture, GPUTexture];
  private depths!: [GPUTexture, GPUTexture];
  private flows: GPUTexture[] = [];
  private seed!: GPUTexture;
  private masks!: [GPUTexture, GPUTexture];
  private usedPixels!: GPUBuffer;
  private motionError!: GPUTexture;
  private reprojection!: [GPUTexture, GPUTexture];
  private reprojectionError!: GPUTexture;
  private blurScratch!: GPUTexture;
  private filteredError!: GPUTexture;
  private output!: GPUTexture;
  private indices!: GPUBuffer;
  private indexCount = 0;
  private meshSize: readonly [number, number] = [0, 0];
  private depthIndex = 0;
  private hasDepth = false;
  private pending: HydraFrameInputs | null = null;
  private configured = false;

  constructor({ device }: FrameGeneratorOptions) { this.device = device; }

  configure(config: FrameGeneratorConfig): void {
    const { renderWidth: width, renderHeight: height } = config;
    for (const size of [width, height]) {
      if (!Number.isInteger(size) || size < 16 || size > this.device.limits.maxTextureDimension2D) {
        throw new Error("Hydra dimensions must be integers in [16, maxTextureDimension2D]");
      }
    }
    if ((config.transferFunction ?? TransferFunction.LinearLdr) !== TransferFunction.LinearLdr) {
      throw new Error("Hydra currently supports linear LDR color only");
    }
    const lw = Math.max(8, Math.floor(width / 2));
    const lh = Math.max(8, Math.floor(height / 2));
    if (lw * lh * 4 > this.device.limits.maxStorageBufferBindingSize) {
      throw new Error("Hydra usage mask exceeds maxStorageBufferBindingSize");
    }
    this.destroy();
    this.linear = this.device.createSampler({ minFilter: "linear", magFilter: "linear", mipmapFilter: "linear" });
    this.nearest = this.device.createSampler({});
    const levels = Math.min(5, Math.floor(Math.log2(Math.min(lw, lh))) - 2);
    this.meshSize = [Math.ceil(lw / BLOCK), Math.ceil(lh / BLOCK)];
    this.colors = [0, 1].map(i => this.texture(lw, lh, "r8unorm", `luma-${i}`, levels)) as [GPUTexture, GPUTexture];
    this.depths = [0, 1].map(i => this.texture(lw, lh, "r32float", `depth-${i}`, levels)) as [GPUTexture, GPUTexture];
    this.flows = Array.from({ length: levels }, (_, level) => this.texture(
      Math.ceil(Math.max(1, lw >> level) / BLOCK), Math.ceil(Math.max(1, lh >> level) / BLOCK),
      "rgba16float", `flow-${level}`, 1, true));
    this.seed = this.texture(1, 1, "rgba16float", "flow-seed");
    this.masks = [0, 1].map(i => this.texture(lw, lh, "rgba8unorm", `usage-${i}`, 3, true)) as [GPUTexture, GPUTexture];
    this.usedPixels = this.device.createBuffer({ label: "hydra-used-pixels", size: lw * lh * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.resources.push(this.usedPixels);
    this.motionError = this.texture(lw, lh, "r8unorm", "motion-error", 3);
    this.reprojection = [0, 1].map(i => this.texture(lw, lh, "rg16float", `reprojection-${i}`)) as [GPUTexture, GPUTexture];
    this.reprojectionError = this.texture(lw, lh, "r8unorm", "reprojection-error", 3);
    this.blurScratch = this.texture(lw, lh, "r8unorm", "blur-scratch");
    this.filteredError = this.texture(lw, lh, "r8unorm", "filtered-error", 3);
    this.output = this.texture(width, height, "rgba16float", "output");
    this.renderPipeline("luma", luma, "r8unorm");
    this.renderPipeline("depth", depthInput, "r32float");
    this.renderPipeline("match", match, "r8unorm");
    this.renderPipeline("reproject", reprojectFragment, "rg16float", reprojectVertex);
    this.renderPipeline("reproject-match", reprojectMatch, "r8unorm");
    this.renderPipeline("blur", reprojectBlur, "r8unorm");
    this.renderPipeline("resolve", resolve, "rgba16float");
    for (const format of ["r8unorm", "rgba8unorm"] as const) this.renderPipeline(`mip-${format}`, mipmap, format);
    this.pipelines.set("search", this.device.createComputePipeline({
      label: "hydra-search", layout: "auto", compute: { module: this.device.createShaderModule({ code: search }), entryPoint: "main" },
    }));
    this.pipelines.set("usage-masks", this.device.createComputePipeline({
      label: "hydra-usage-masks", layout: "auto", compute: { module: this.device.createShaderModule({ code: usageMasks }), entryPoint: "main" },
    }));
    const indices: number[] = [];
    const [gx, gy] = this.meshSize;
    for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
      const a = y * (gx + 1) + x, b = a + 1, c = a + gx + 1, d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
    this.indexCount = indices.length;
    this.indices = this.device.createBuffer({ label: "hydra-mesh", size: indices.length * 4, usage: GPUBufferUsage.INDEX, mappedAtCreation: true });
    new Uint32Array(this.indices.getMappedRange()).set(indices);
    this.indices.unmap();
    this.resources.push(this.indices);
    this.uniform("mesh", new Uint32Array([gx, gy, 0, 0]));
    this.uniform("layer-0", new Uint32Array(4));
    this.uniform("layer-1", new Uint32Array([1, 0, 0, 0]));
    this.uniform("quantize", new Uint32Array([1, 0, 0, 0]));
    this.uniform("reduce", new Uint32Array(4));
    this.uniform("blur-x", new Int32Array([1, 0, 0, 0]));
    this.uniform("blur-y", new Int32Array(4));
    this.uniform("scale", new Float32Array([1, 1, 0, 0]));
    // The recovered match uniform retains two unused matrices at offset 16.
    this.uniform("match", new Float32Array([1, 1, 0, 0, ...IDENTITY, ...IDENTITY]));
    this.uniform("transforms", new Float32Array([...IDENTITY, ...IDENTITY]));
    const searchData = new ArrayBuffer(16);
    new Uint32Array(searchData).set([0, 1]);
    new Float32Array(searchData).set([1, 1], 2);
    this.uniform("search", new Uint8Array(searchData));
    this.configured = true;
  }

  // Bracket the entire prepare submission, including intervening passes and
  // buffer clears. Both indices are optional, as in WebGPU timestampWrites.
  prepare(inputs: HydraFrameInputs, timestampWrites?: GPUComputePassTimestampWrites): void {
    this.requireConfigured();
    if (this.pending) throw new Error("prepare() called twice without dispatch()");
    const { previousToCurrentClip: forward, currentToPreviousClip: backward } = inputs;
    if (Boolean(forward) !== Boolean(backward)) throw new Error("Supply both camera transforms or neither");
    for (const matrix of [forward, backward]) {
      if (matrix && (matrix.length !== 16 || !Array.from(matrix).every(Number.isFinite))) {
        throw new Error("Camera transforms must contain 16 finite column-major values");
      }
    }
    const transforms = new Float32Array([...(forward ?? IDENTITY), ...(backward ?? IDENTITY)]);
    // Convert clip-up matrices to the recovered shaders' UV-down clip convention.
    for (let m = 0; m < 2; m++) for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      transforms[m * 16 + c * 4 + r] *= (c === 1 ? -1 : 1) * (r === 1 ? -1 : 1);
    }
    this.device.queue.writeBuffer(this.uniforms.get("transforms")!, 0, transforms);
    if (inputs.resetHistory) this.hasDepth = false;
    const encoder = this.device.createCommandEncoder({ label: "hydra-prepare" });
    for (const [i, input] of [inputs.previousColor, inputs.currentColor].entries()) {
      this.draw(encoder, "luma", this.view(this.colors[i]), [[[0, this.linear]], [[0, input]]], false,
        i === 0 && timestampWrites?.beginningOfPassWriteIndex !== undefined
          ? { querySet: timestampWrites.querySet, beginningOfPassWriteIndex: timestampWrites.beginningOfPassWriteIndex } : undefined);
      this.mips(encoder, this.colors[i]);
    }
    this.depthIndex ^= 1;
    this.depthPyramid(encoder, inputs.depth, this.depths[this.depthIndex]);
    if (inputs.previousDepth || !this.hasDepth) this.depthPyramid(encoder, inputs.previousDepth ?? inputs.depth, this.depths[this.depthIndex ^ 1]);
    this.clear(encoder, this.seed, [0, 0, 1, 0]);
    const compute = this.pipelines.get("search")! as GPUComputePipeline;
    for (let level = this.flows.length - 1; level >= 0; level--) {
      const flow = this.flows[level];
      const pass = encoder.beginComputePass({ label: `hydra-search-${level}` });
      pass.setPipeline(compute);
      this.bind(pass, compute, [[[0, this.linear]], [
        [0, this.view(this.colors[0], level)], [1, this.view(this.colors[1], level)],
        [2, this.view(this.flows[level + 1] ?? this.seed)], [3, this.view(flow)],
      ], [], [[0, this.ub("search")]]]);
      pass.dispatchWorkgroups(flow.width, flow.height);
      pass.end();
    }
    encoder.clearBuffer(this.usedPixels);
    this.draw(encoder, "match", this.view(this.motionError), [
      [[0, this.linear], [2, this.view(this.flows[0])], [3, { buffer: this.usedPixels }]],
      [[0, this.view(this.colors[0])], [1, this.view(this.colors[1])]], [], [[0, this.ub("match")]],
    ]);
    const maskPipeline = this.pipelines.get("usage-masks")! as GPUComputePipeline;
    const maskPass = encoder.beginComputePass({ label: "hydra-usage-masks" });
    maskPass.setPipeline(maskPipeline);
    this.bind(maskPass, maskPipeline, [[[0, { buffer: this.usedPixels }],
      [1, this.view(this.masks[0])], [2, this.view(this.masks[1])]]]);
    maskPass.dispatchWorkgroups(Math.ceil(this.masks[0].width / 8), Math.ceil(this.masks[0].height / 8));
    maskPass.end();
    for (const texture of [...this.masks, this.motionError]) this.mips(encoder, texture);
    for (const i of [0, 1]) {
      this.draw(encoder, "reproject", this.view(this.reprojection[i]), [[], [[0, this.ub("transforms")]],
        [[0, this.view(this.depths[this.depthIndex ^ (i === 0 ? 1 : 0)], Math.min(3, this.depths[0].mipLevelCount - 1))]],
        [[0, this.ub(`layer-${i}`)], [1, this.ub("mesh")]],
      ], true);
    }
    this.draw(encoder, "reproject-match", this.view(this.reprojectionError), [
      [[0, this.linear], [1, this.linear], [2, this.view(this.reprojection[0])], [3, this.view(this.reprojection[1])]],
      [[0, this.view(this.colors[0])], [1, this.view(this.colors[1])]], [], [[0, this.ub("scale")]],
    ]);
    this.draw(encoder, "blur", this.view(this.blurScratch), [[[0, this.linear]], [[0, this.view(this.reprojectionError)]], [], [[0, this.ub("blur-x")]]]);
    this.draw(encoder, "blur", this.view(this.filteredError), [[[0, this.linear]], [[0, this.view(this.blurScratch)]], [], [[0, this.ub("blur-y")]]]);
    this.mips(encoder, this.filteredError, timestampWrites?.endOfPassWriteIndex !== undefined
      ? { querySet: timestampWrites.querySet, endOfPassWriteIndex: timestampWrites.endOfPassWriteIndex } : undefined);
    this.device.queue.submit([encoder.finish()]);
    this.hasDepth = true;
    this.pending = inputs;
  }

  dispatch(timestampWrites?: GPUComputePassTimestampWrites): GPUTexture {
    this.requireConfigured();
    if (!this.pending) throw new Error("dispatch() called before prepare()");
    const encoder = this.device.createCommandEncoder({ label: "hydra-resolve" });
    this.draw(encoder, "resolve", this.view(this.output), [
      [[0, this.linear], [1, this.linear], [2, this.nearest], [3, this.view(this.flows[0])],
        [4, this.motionError.createView()], [5, this.masks[0].createView()], [6, this.masks[1].createView()],
        [7, this.filteredError.createView()], [8, this.view(this.reprojection[0])], [9, this.view(this.reprojection[1])]],
      [[0, this.pending.previousColor], [1, this.pending.currentColor]], [], [[0, this.ub("scale")]],
    ], false, timestampWrites);
    this.device.queue.submit([encoder.finish()]);
    this.pending = null;
    return this.output;
  }

  destroy(): void {
    for (const resource of this.resources) resource.destroy();
    this.resources = [];
    this.uniforms.clear();
    this.pipelines.clear();
    this.flows = [];
    this.pending = null;
    this.hasDepth = false;
    this.depthIndex = 0;
    this.configured = false;
  }

  private requireConfigured(): void {
    if (!this.configured) throw new Error("configure() must be called before prepare() or dispatch()");
  }
  private view(texture: GPUTexture, level = 0): GPUTextureView {
    return texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
  }
  private ub(name: string): GPUBufferBinding { return { buffer: this.uniforms.get(name)! }; }
  private uniform(name: string, data: ArrayBufferView<ArrayBuffer>): void {
    const buffer = this.device.createBuffer({ label: `hydra-${name}`, size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data);
    this.resources.push(buffer);
    this.uniforms.set(name, buffer);
  }
  private texture(width: number, height: number, format: GPUTextureFormat, label: string, mipLevelCount = 1, storage = false): GPUTexture {
    const texture = this.device.createTexture({ label: `hydra-${label}`, size: [width, height], format, mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC |
        (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
    });
    this.resources.push(texture);
    return texture;
  }
  private renderPipeline(name: string, fragment: string, format: GPUTextureFormat, vertex = fullscreen): void {
    this.pipelines.set(name, this.device.createRenderPipeline({ label: `hydra-${name}`, layout: "auto",
      vertex: { module: this.device.createShaderModule({ code: vertex }), entryPoint: "main" },
      fragment: { module: this.device.createShaderModule({ code: fragment }), entryPoint: "main", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    }));
  }
  private bind(pass: GPURenderPassEncoder | GPUComputePassEncoder, pipeline: Pipeline, groups: readonly Bindings[]): void {
    groups.forEach((entries, group) => pass.setBindGroup(group, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(group), entries: entries.map(([binding, resource]) => ({ binding, resource })),
    })));
  }
  private draw(encoder: GPUCommandEncoder, name: string, target: GPUTextureView, groups: readonly Bindings[], mesh = false,
    timestampWrites?: GPURenderPassTimestampWrites): void {
    const pipeline = this.pipelines.get(name)! as GPURenderPipeline;
    const pass = encoder.beginRenderPass({ label: `hydra-${name}`, timestampWrites,
      colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
    pass.setPipeline(pipeline);
    this.bind(pass, pipeline, groups);
    if (mesh) { pass.setIndexBuffer(this.indices, "uint32"); pass.drawIndexed(this.indexCount); }
    else pass.draw(3);
    pass.end();
  }
  private clear(encoder: GPUCommandEncoder, texture: GPUTexture, clearValue: GPUColor): void {
    encoder.beginRenderPass({ colorAttachments: [{ view: this.view(texture), loadOp: "clear", storeOp: "store", clearValue }] }).end();
  }
  private mips(encoder: GPUCommandEncoder, texture: GPUTexture, finalTimestampWrites?: GPURenderPassTimestampWrites): void {
    for (let level = 1; level < texture.mipLevelCount; level++) this.draw(encoder, `mip-${texture.format}`, this.view(texture, level), [
      [[0, this.view(texture, level - 1)], [1, this.linear]],
    ], false, level === texture.mipLevelCount - 1 ? finalTimestampWrites : undefined);
  }
  private depthPyramid(encoder: GPUCommandEncoder, input: GPUTextureView, target: GPUTexture): void {
    for (let level = 0; level < target.mipLevelCount; level++) this.draw(encoder, "depth", this.view(target, level), [
      [[0, level === 0 ? input : this.view(target, level - 1)], [1, this.ub(level === 0 ? "quantize" : "reduce")]],
    ]);
  }
}
