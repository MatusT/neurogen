import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import { createInterpolator, HydraFrameGenerator, type HydraFrameInputs } from "../src/index.js";

let gpu: GpuHarness;
beforeAll(async () => { gpu = await createGpuHarness(); }, 30000);
const config = (width = 128, height = 96) => ({ renderWidth: width, renderHeight: height, nearPlane: 0.1, farPlane: 100, verticalFovRadians: 1 });

function color(width: number, height: number, pixel: (x: number, y: number) => number[]): GPUTexture {
  const texture = gpu.device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const data = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data.set(pixel(x, y), (y * width + x) * 4);
  gpu.device.queue.writeTexture({ texture }, data, { bytesPerRow: width * 4 }, [width, height]);
  return texture;
}

async function read(texture: GPUTexture): Promise<Float32Array> {
  const pitch = Math.ceil(texture.width * 8 / 256) * 256;
  const buffer = gpu.device.createBuffer({ size: pitch * texture.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = gpu.device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: pitch }, [texture.width, texture.height]);
  gpu.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const values = new Uint16Array(buffer.getMappedRange());
  const out = new Float32Array(texture.width * texture.height * 4);
  for (let y = 0; y < texture.height; y++) for (let x = 0; x < texture.width * 4; x++) {
    const bits = values[y * pitch / 2 + x], exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    out[y * texture.width * 4 + x] = (bits & 32768 ? -1 : 1) * (exponent === 0 ? mantissa * 2 ** -24 : exponent === 31 ? NaN : (mantissa + 1024) * 2 ** (exponent - 25));
  }
  buffer.unmap(); buffer.destroy();
  return out;
}

function depthTexture(width: number, height: number, value = 0.5): GPUTexture {
  const texture = gpu.device.createTexture({ size: [width, height], format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  gpu.device.queue.writeTexture({ texture }, new Float32Array(width * height).fill(value),
    { bytesPerRow: width * 4 }, [width, height]);
  return texture;
}

async function generate(previous: GPUTexture, current: GPUTexture, camera: Partial<HydraFrameInputs> = {}): Promise<Float32Array> {
  const generator = createInterpolator({ device: gpu.device, backend: "hydra" });
  const depth = depthTexture(current.width, current.height);
  gpu.device.pushErrorScope("validation");
  try {
    generator.configure(config(current.width, current.height));
    generator.prepare({ previousColor: previous.createView(), currentColor: current.createView(), depth: depth.createView(), ...camera });
    const out = await read(generator.dispatch());
    const error = await gpu.device.popErrorScope();
    expect(error?.message).toBeUndefined();
    return out;
  } finally { generator.destroy(); depth.destroy(); }
}

describe("Hydra interpolation", () => {
  it("preserves an identical asymmetric image and opaque alpha at odd dimensions", async () => {
    const width = 131, height = 99;
    const pixel = (x: number, y: number) => [x < 50 ? 204 : 51, y < 30 ? 179 : 26, 77, 255];
    const input = color(width, height, pixel);
    try {
      const result = await generate(input, input);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        for (let c = 0; c < 4; c++) expect(result[(y * width + x) * 4 + c]).toBeCloseTo(pixel(x, y)[c] / 255, 2);
      }
    } finally { input.destroy(); }
  }, 30000);

  it.each([[8, 0], [-8, 0], [0, 8], [0, -8]])("moves a textured object (%i, %i) to its midpoint", async (dx, dy) => {
    const width = 192, height = 128;
    const pixel = (time: number) => (x: number, y: number) => {
      x -= dx * time;
      y -= dy * time;
      if (x < 48 || x >= 112 || y < 24 || y >= 104) return [0, 0, 0, 255];
      const v = (x * 29 + y * 43) % 128 + 127;
      return [v, v, v, 255];
    };
    const previous = color(width, height, pixel(0)), current = color(width, height, pixel(1));
    try {
      const result = await generate(previous, current);
      let error = 0, baseline = 0;
      for (let y = 40; y < 88; y++) for (let x = 64; x < 96; x++) {
        const target = pixel(0.5)(x, y)[0] / 255;
        error += Math.abs(result[(y * width + x) * 4] - target);
        baseline += Math.abs((pixel(0)(x, y)[0] + pixel(1)(x, y)[0]) / 510 - target);
      }
      expect([...result].every(Number.isFinite)).toBe(true);
      expect(error).toBeLessThan(baseline * 0.8);
    } finally { previous.destroy(); current.destroy(); }
  }, 30000);

  it.each([[16, 0], [0, 16]])("reprojects camera motion (%i, %i) using float depth and clip-space matrices", async (dx, dy) => {
    const width = 192, height = 128;
    const pixel = (x: number, y: number) => {
      const value = ((x * 29 + y * 43) % 128 + 128) % 128 + 80;
      return [value, value, value, 255];
    };
    const previous = color(width, height, pixel);
    const current = color(width, height, (x, y) => pixel(x - dx, y - dy));
    const forward = [1, 0, 0, 0, 0, 1, 0, 0, 4 * dx / width, -4 * dy / height, 1, 0, 0, 0, 0, 1];
    const backward = [...forward];
    backward[8] *= -1;
    backward[9] *= -1;
    try {
      const result = await generate(previous, current, { previousToCurrentClip: forward, currentToPreviousClip: backward });
      let error = 0;
      for (let y = 32; y < height - 32; y++) for (let x = 32; x < width - 32; x++) {
        error += Math.abs(result[(y * width + x) * 4] - pixel(x - dx / 2, y - dy / 2)[0] / 255);
      }
      expect(error / ((width - 64) * (height - 64))).toBeLessThan(0.01);
    } finally { previous.destroy(); current.destroy(); }
  }, 30000);

  it("resets retained depth and validates the prepare/dispatch lifecycle", async () => {
    const width = 64, height = 64;
    const input = color(width, height, (x, y) => [x * 4, y * 4, 0, 255]);
    const oldDepth = depthTexture(width, height, 0.1), newDepth = depthTexture(width, height, 0.8);
    const generator = new HydraFrameGenerator({ device: gpu.device });
    const forward = [1, 0, 0, 0, 0, 1, 0, 0, 0.25, 0, 1, 0, 0, 0, 0, 1];
    const backward = [...forward]; backward[8] = -forward[8];
    const inputs = { previousColor: input.createView(), currentColor: input.createView(), depth: newDepth.createView(),
      previousToCurrentClip: forward, currentToPreviousClip: backward };
    gpu.device.pushErrorScope("validation");
    try {
      expect(() => generator.dispatch()).toThrow(/configure/);
      generator.configure(config(width, height));
      expect(() => generator.dispatch()).toThrow(/prepare/);
      expect(() => generator.prepare({ ...inputs, currentToPreviousClip: undefined })).toThrow(/both/);
      generator.prepare({ ...inputs, depth: oldDepth.createView() });
      expect(() => generator.prepare(inputs)).toThrow(/twice/);
      generator.dispatch();
      generator.prepare({ ...inputs, resetHistory: true });
      const reset = await read(generator.dispatch());
      generator.configure(config(width, height));
      generator.prepare(inputs);
      const fresh = await read(generator.dispatch());
      expect(reset).toEqual(fresh);
      generator.destroy();
      expect(() => generator.prepare(inputs)).toThrow(/configure/);
      expect((await gpu.device.popErrorScope())?.message).toBeUndefined();
    } finally { generator.destroy(); input.destroy(); oldDepth.destroy(); newDepth.destroy(); }
  }, 30000);
});
