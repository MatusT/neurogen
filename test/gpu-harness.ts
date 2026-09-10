// Real (not mocked) WebGPU device for vitest, via the `webgpu` package's
// Dawn bindings — lets shader passes be verified by actually running them,
// not just by naga syntax validation.
import { create, globals } from "webgpu";

// The `webgpu` package ships runtime constants (GPUBufferUsage etc) that
// @webgpu/types only provides as compile-time types. WGSL/orchestration code
// written against the ambient browser globals needs them on globalThis to
// run under Node; harmless to redo across files since the source objects
// are stable singletons.
Object.assign(globalThis, globals);

// mappedAtCreation buffers, and copyBufferToBuffer's size argument, both
// require 4-byte-aligned sizes.
const BUFFER_ALIGNMENT = 4;
function alignUp(size: number): number {
  return Math.ceil(size / BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT;
}

// Per-resource GC retention (tying a buffer/pipeline's lifetime to `device`
// via a property) turned out to be structurally incomplete: `device` itself
// is part of the public API, so any handle a caller creates directly from
// it — a shader module, a queue reference, anything the harness doesn't
// wrap — has no such property and can still be collected out from under a
// live handle, crashing Dawn's Node bindings. Root at module scope instead,
// so any live handle survives for the process's lifetime regardless of what
// a caller holds or how they got it. Both `gpu` and `device` need their own
// root: Dawn's device wrapper runs its own destructor (calling
// device.Destroy()) when *it* is collected, independent of whether the
// owning `gpu` is still alive — rooting `gpu` alone doesn't keep `device`'s
// wrapper reachable. `dispose()` is an explicit opt-out for tests that want
// to release both early.
const rootedGpuResources = new Set<unknown>();

export interface GpuHarness {
  device: GPUDevice;
  createComputePipeline(wgslSource: string, entryPoint?: string): GPUComputePipeline;
  createStorageBuffer(data: ArrayBufferView, extraUsage?: GPUBufferUsageFlags): GPUBuffer;
  createUniformBuffer(data: ArrayBufferView): GPUBuffer;
  // rgba32float only — a texel format mismatch between this raw float upload
  // and a narrower target format (e.g. rgba8unorm) reinterprets the bytes
  // instead of converting them, silently producing wrong pixel values.
  createTexture(width: number, height: number, data: Float32Array<ArrayBuffer>): GPUTexture;
  // Throws if the dispatch produced a WebGPU validation error (e.g. a
  // uniform buffer bound where the shader expects storage) — otherwise a
  // dispatch that did nothing can still read back as a plausible-looking
  // zeroed buffer and falsely pass a numeric fixture.
  dispatch(
    pipeline: GPUComputePipeline,
    bindings: GPUBindingResource[],
    workgroupCounts: readonly [number, number?, number?],
  ): Promise<void>;
  readBuffer(buffer: GPUBuffer, byteLength: number): Promise<ArrayBuffer>;
  // Releases this harness's root on its GPU instance. Optional — tests that
  // never call it just keep the instance alive until the process exits.
  dispose(): void;
}

// Device creation is slow (~seconds) — call once per test file (e.g. in
// `beforeAll`) and share the returned harness across that file's tests.
export async function createGpuHarness(): Promise<GpuHarness> {
  const gpu = create([]);
  rootedGpuResources.add(gpu);
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error("no WebGPU adapter available in this environment");
    }
    const device = await adapter.requestDevice();
    rootedGpuResources.add(device);
    return buildHarness(gpu, device);
  } catch (err) {
    // Nothing was returned for the caller to eventually dispose() — release
    // the root ourselves, or a failed createGpuHarness() call leaks a `gpu`
    // for the rest of the process.
    rootedGpuResources.delete(gpu);
    throw err;
  }
}

function buildHarness(gpu: unknown, device: GPUDevice): GpuHarness {
  function createComputePipeline(wgslSource: string, entryPoint = "main"): GPUComputePipeline {
    const module = device.createShaderModule({ code: wgslSource });
    return device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
  }

  function createBufferWithData(data: ArrayBufferView, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = device.createBuffer({ size: alignUp(data.byteLength), usage, mappedAtCreation: true });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    );
    buffer.unmap();
    return buffer;
  }

  function createStorageBuffer(data: ArrayBufferView, extraUsage: GPUBufferUsageFlags = 0): GPUBuffer {
    return createBufferWithData(
      data,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
    );
  }

  function createUniformBuffer(data: ArrayBufferView): GPUBuffer {
    return createBufferWithData(data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  // `data` is tightly packed rgba32float, row-major, 16 bytes/texel.
  function createTexture(width: number, height: number, data: Float32Array<ArrayBuffer>): GPUTexture {
    const texture = device.createTexture({
      size: [width, height],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture },
      data,
      { bytesPerRow: width * 16, rowsPerImage: height },
      { width, height },
    );
    return texture;
  }

  async function dispatch(
    pipeline: GPUComputePipeline,
    bindings: GPUBindingResource[],
    workgroupCounts: readonly [number, number?, number?],
  ): Promise<void> {
    device.pushErrorScope("validation");
    // A synchronous throw anywhere in the try (e.g. an invalid workgroup
    // count) must still pop the scope, or it's left dangling and silently
    // swallows an unrelated later dispatch's validation error instead.
    try {
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((resource, binding) => ({ binding, resource })),
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...workgroupCounts);
      pass.end();
      device.queue.submit([encoder.finish()]);
    } finally {
      const error = await device.popErrorScope();
      if (error) {
        throw new Error(`WebGPU validation error during dispatch: ${error.message}`);
      }
    }
  }

  async function readBuffer(buffer: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({
      size: alignUp(byteLength),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(buffer, 0, readback, 0, alignUp(byteLength));
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const copy = readback.getMappedRange().slice(0, byteLength);
      readback.unmap();
      return copy;
    } finally {
      readback.destroy();
    }
  }

  function dispose(): void {
    rootedGpuResources.delete(gpu);
    rootedGpuResources.delete(device);
    device.destroy();
  }

  return {
    device,
    createComputePipeline,
    createStorageBuffer,
    createUniformBuffer,
    createTexture,
    dispatch,
    readBuffer,
    dispose,
  };
}
