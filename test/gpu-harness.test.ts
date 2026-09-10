import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";

describe("gpu-harness", () => {
  let harness: GpuHarness;

  beforeAll(async () => {
    harness = await createGpuHarness();
  });

  it("dispatches a compute shader and reads back real results", async () => {
    const pipeline = harness.createComputePipeline(`
      @group(0) @binding(0) var<storage, read_write> out: array<f32>;
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        out[id.x] = f32(id.x) * 2.0;
      }
    `);
    const outBuffer = harness.createStorageBuffer(new Float32Array(4));

    await harness.dispatch(pipeline, [{ buffer: outBuffer }], [4]);

    const result = new Float32Array(await harness.readBuffer(outBuffer, 16));
    expect(Array.from(result)).toEqual([0, 2, 4, 6]);
  });

  it("reads uniform buffer input into a compute pass", async () => {
    const pipeline = harness.createComputePipeline(`
      struct Params { scale: f32 }
      @group(0) @binding(0) var<uniform> params: Params;
      @group(0) @binding(1) var<storage, read_write> out: array<f32>;
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        out[id.x] = f32(id.x) * params.scale;
      }
    `);
    const paramsBuffer = harness.createUniformBuffer(new Float32Array([10]));
    const outBuffer = harness.createStorageBuffer(new Float32Array(3));

    await harness.dispatch(pipeline, [{ buffer: paramsBuffer }, { buffer: outBuffer }], [3]);

    const result = new Float32Array(await harness.readBuffer(outBuffer, 12));
    expect(Array.from(result)).toEqual([0, 10, 20]);
  });

  it("throws on a validation error instead of silently resolving", async () => {
    const pipeline = harness.createComputePipeline(`
      @group(0) @binding(0) var<storage, read_write> out: array<f32>;
      @compute @workgroup_size(1)
      fn main() { out[0] = 1.0; }
    `);
    // Bound as a uniform buffer where the shader's bind group layout (from
    // "auto") expects storage — a real validation error, not a JS-side one.
    const wrongKindBuffer = harness.createUniformBuffer(new Float32Array([0]));

    await expect(harness.dispatch(pipeline, [{ buffer: wrongKindBuffer }], [1])).rejects.toThrow(
      /validation error/i,
    );
  });

  it("round-trips buffer sizes that aren't 4-byte aligned", async () => {
    // mappedAtCreation buffers (upload) and copyBufferToBuffer (readback)
    // both require 4-byte-aligned sizes under the hood; 3 bytes exercises
    // both without a shader in the loop, since WGSL runtime-sized storage
    // arrays derive their length from the bound buffer's size and would
    // make an odd byte count ambiguous to dispatch against directly.
    const data = new Uint8Array([1, 2, 3]);
    const buffer = harness.createStorageBuffer(data);

    const result = await harness.readBuffer(buffer, 3);

    expect(result.byteLength).toBe(3);
    expect(new Uint8Array(result)).toEqual(data);
  });

  it("doesn't leak a dangling error scope when dispatch throws synchronously", async () => {
    const pipeline = harness.createComputePipeline(`
      @group(0) @binding(0) var<storage, read_write> out: array<f32>;
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) { out[id.x] = f32(id.x); }
    `);
    const outBuffer = harness.createStorageBuffer(new Float32Array(4));

    // Invalid workgroup count throws before the encoder is ever submitted —
    // pushErrorScope was already called, so popErrorScope must still run in
    // a finally or the scope dangles and swallows the NEXT call's error.
    await expect(harness.dispatch(pipeline, [{ buffer: outBuffer }], [-1])).rejects.toThrow();

    // An unrelated, valid dispatch right after must not inherit that
    // leftover scope. Its own GPU computation would succeed regardless of
    // whether the scope leaked — that only affects error *reporting*, not
    // execution — so this alone wouldn't catch a reintroduced leak.
    await harness.dispatch(pipeline, [{ buffer: outBuffer }], [4]);
    const result = new Float32Array(await harness.readBuffer(outBuffer, 16));
    expect(Array.from(result)).toEqual([0, 1, 2, 3]);

    // What actually proves the stack is balanced: a pop with nothing of
    // ours left on it rejects (empty-stack is a spec-defined error). If the
    // failed dispatch's scope had leaked, this would resolve instead,
    // popping that leftover scope.
    await expect(harness.device.popErrorScope()).rejects.toThrow();
  });

  it("survives forced GC when only a directly-created device resource escapes", async () => {
    // Not in the DOM/ES2022 lib TS is configured against; only present at
    // runtime when node was launched with --expose-gc.
    const forceGc = (globalThis as { gc?: () => void }).gc;
    if (!forceGc) {
      throw new Error("run with --expose-gc (see vitest.config.ts poolOptions)");
    }

    // A fresh harness, not the shared one from beforeAll — it must go fully
    // out of scope so only the returned buffer keeps anything reachable.
    const mapReadBuffer = await (async () => {
      const scoped = await createGpuHarness();
      // Bypasses every harness factory, same as a caller reaching for
      // device.createBuffer directly for a usage the harness doesn't wrap.
      return scoped.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    })();

    forceGc();
    await new Promise((r) => setTimeout(r, 50));
    forceGc();
    await new Promise((r) => setTimeout(r, 50));
    forceGc();

    // If the owning device's wrapper was collected, this rejects with
    // AbortError instead of resolving.
    await expect(mapReadBuffer.mapAsync(GPUMapMode.READ)).resolves.toBeUndefined();
    mapReadBuffer.unmap();
  });
});
