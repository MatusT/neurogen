import { describe, expect, it, vi } from "vitest";
import { summarizeTimings, timeInterpolation } from "../scripts/quality/benchmark.js";
import { imageMetrics, crossfade } from "../scripts/quality/metrics.js";
import { createGpuHarness } from "./gpu-harness.js";
import { readColor } from "../demo/verify.js";

describe("comparison benchmark", () => {
  it("summarizes unsorted samples with an even median and nearest-rank p95", () => {
    const samples = [100, ...Array.from({ length: 19 }, (_, i) => i + 1)];
    expect(summarizeTimings(samples)).toEqual({ samples: 20, meanMs: 14.5, medianMs: 10.5, p95Ms: 19 });
    expect(samples[0]).toBe(100);
    expect(summarizeTimings([3, 1, 2]).medianMs).toBe(2);
    expect(summarizeTimings([4])).toEqual({ samples: 1, meanMs: 4, medianMs: 4, p95Ms: 4 });
  });

  it("rejects missing or invalid timings", () => {
    for (const samples of [[], [NaN], [Infinity], [-1]]) {
      expect(() => summarizeTimings(samples)).toThrow(/Benchmark timings/);
    }
    // A timestamp clock may quantize a very short phase to zero.
    expect(summarizeTimings([0]).medianMs).toBe(0);
  });

  it("excludes earlier queue work and includes asynchronous GPU completion", async () => {
    let clock = 0;
    let waits = 0;
    const now = vi.spyOn(performance, "now").mockImplementation(() => clock);
    const queue = { onSubmittedWorkDone: async () => {
      await Promise.resolve();
      clock += waits++ === 0 ? 100 : 7;
      return undefined;
    } };
    try {
      const elapsed = await timeInterpolation(queue, () => {
        expect(clock).toBe(100);
        clock += 2;
      });
      expect(elapsed).toBe(9);
    } finally { now.mockRestore(); }
  });
});

describe("quality metrics", () => {
  it("measures RGB error without diluting moving pixels with static background", () => {
    const previous = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1]);
    const current = new Float32Array([1, 1, 1, 1, 0, 0, 0, 1]);
    const truth = new Float32Array([1, 1, 1, 0, 0, 0, 0, 0]);
    const measured = imageMetrics(crossfade(previous, current), truth, previous, current);
    expect(measured.mae).toBe(0.25);
    expect(measured.rmse).toBeCloseTo(Math.sqrt(0.125));
    expect(measured.changedPixels).toBe(1);
    expect(measured.changedRegionMae).toBe(0.5);
    expect(imageMetrics(truth, truth, truth, truth)).toMatchObject({ mae: 0, psnr: null, changedRegionMae: null });
  });

  it("rejects nonfinite GPU results rather than reporting a plausible score", () => {
    const valid = new Float32Array([0, 0, 0, 1]);
    expect(() => imageMetrics(new Float32Array([NaN, 0, 0, 1]), valid, valid, valid)).toThrow(/Nonfinite/);
  });

  it("retains NaN and infinities when decoding GPU half-float readback", async () => {
    const gpu = await createGpuHarness();
    const texture = gpu.device.createTexture({ size: [1, 1], format: "rgba16float",
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    try {
      gpu.device.queue.writeTexture({ texture }, new Uint16Array([0x7c00, 0xfc00, 0x7e00, 0x3c00]), { bytesPerRow: 8 }, [1, 1]);
      const decoded = await readColor(gpu.device, texture);
      expect([...decoded]).toEqual([Infinity, -Infinity, NaN, 1]);
    } finally { texture.destroy(); gpu.dispose(); }
  });
});
