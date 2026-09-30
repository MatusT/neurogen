import { describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { decodeGpuTiming, GpuTimer } from "../scripts/quality/gpu-timer.js";
import { createInterpolator } from "../src/index.js";
import { GBuffer } from "../demo/gbuffer.js";
import { demoScene, PROJECTION, projectionFor } from "../demo/scene.js";
import { readColor } from "../demo/verify.js";

Object.assign(globalThis, globals);
const gpu = create([]);

describe("GPU timestamps", () => {
  it("preserves nanosecond differences at large epochs and excludes the submission gap", () => {
    const epoch = 2n ** 60n;
    expect(decodeGpuTiming(new BigUint64Array([
      epoch, epoch + 1_000_001n, epoch + 100_000_000n, epoch + 102_000_000n,
    ]))).toEqual({ prepareMs: 1.000001, dispatchMs: 2, totalMs: 3.000001 });
    expect(decodeGpuTiming(new BigUint64Array([1n, 1n, 2n, 2n])).totalMs).toBe(0);
    expect(() => decodeGpuTiming(new BigUint64Array([2n, 1n, 3n, 4n]))).toThrow(/Invalid GPU/);
    expect(() => decodeGpuTiming(new BigUint64Array(3))).toThrow(/Invalid GPU/);
  });

  for (const backend of ["fsr3", "hydra"] as const) it(`times ${backend} without changing output, and reuses query slots`, async context => {
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");
    if (!adapter.features.has("timestamp-query")) { context.skip(); return; }
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
    device.pushErrorScope("validation");
    const timer = new GpuTimer(device);
    const size = [64, 64] as const;
    const projection = projectionFor(size);
    const real = new GBuffer(device, size, projection);
    const timed = createInterpolator({ device, backend });
    const plain = createInterpolator({ device, backend });
    try {
      for (const generator of [timed, plain]) generator.configure({ ...PROJECTION, renderWidth: size[0], renderHeight: size[1] });
      real.renderFrame(demoScene(projection, 0));
      for (let frame = 1; frame <= 2; frame++) {
        real.renderFrame(demoScene(projection, frame));
        const inputs = real.frameInputs();
        timed.prepare(inputs, timer.prepareWrites);
        const output = timed.dispatch(timer.dispatchWrites);
        const timing = await timer.read();
        expect(timing.prepareMs).toBeGreaterThanOrEqual(0);
        expect(timing.dispatchMs).toBeGreaterThanOrEqual(0);
        expect(timing.totalMs).toBeGreaterThan(0);
        plain.prepare(inputs);
        const expected = plain.dispatch();
        expect(await readColor(device, output)).toEqual(await readColor(device, expected));
      }
      expect(await device.popErrorScope()).toBeNull();
    } finally {
      await device.queue.onSubmittedWorkDone();
      timed.destroy(); plain.destroy(); real.destroy(); timer.destroy(); device.destroy();
    }
  });
});
