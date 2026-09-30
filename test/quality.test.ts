import { describe, expect, it } from "vitest";
import { imageMetrics, crossfade } from "../scripts/quality/metrics.js";
import { createGpuHarness } from "./gpu-harness.js";
import { readColor } from "../demo/verify.js";

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
