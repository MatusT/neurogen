import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import setupWgsl from "../src/wgsl/generated/frameinterpolation/setup.wgsl.js";
import reconstructAndDilateWgsl from "../src/wgsl/generated/frameinterpolation/reconstruct_and_dilate.wgsl.js";

// Mirrors FrameInterpolationParams in src/wgsl/frameinterpolation/params.wgsl.
// Padded to 48 bytes; the struct itself is 40.
const PARAMS_BYTE_LENGTH = 48;

interface ParamsOverrides {
  renderSize: [number, number];
  nearPlane?: number;
  farPlane?: number;
  verticalFovRadians?: number;
  backbufferTransferFunction?: number;
  minMaxLuminance?: [number, number];
  inpaintingMipLevel?: number;
  reset?: number;
}

function frameInterpolationParams(overrides: ParamsOverrides): Uint8Array {
  const bytes = new Uint8Array(PARAMS_BYTE_LENGTH);
  const view = new DataView(bytes.buffer);
  view.setInt32(0, overrides.renderSize[0], true);
  view.setInt32(4, overrides.renderSize[1], true);
  view.setFloat32(8, overrides.nearPlane ?? 0.1, true);
  view.setFloat32(12, overrides.farPlane ?? 100, true);
  view.setFloat32(16, overrides.verticalFovRadians ?? Math.PI / 3, true);
  view.setUint32(20, overrides.backbufferTransferFunction ?? 0, true);
  view.setFloat32(24, overrides.minMaxLuminance?.[0] ?? 0, true);
  view.setFloat32(28, overrides.minMaxLuminance?.[1] ?? 1000, true);
  view.setUint32(32, overrides.inpaintingMipLevel ?? 0, true);
  view.setUint32(36, overrides.reset ?? 0, true);
  return bytes;
}

// The float bit pattern setup.wgsl clears the reconstructed depths to.
const FAR_SENTINEL_BITS = new Uint32Array(new Float32Array([1]).buffer)[0];

function groupCount(size: number, workgroupSize = 8): number {
  return Math.ceil(size / workgroupSize);
}

function opticalFlowGridSize([width, height]: [number, number]): [number, number] {
  return [Math.ceil(width / 8), Math.ceil(height / 8)];
}

// Packs a per-pixel scalar into the .x channel of an rgba32float texture, the
// only format the harness uploads.
function scalarTexture(
  harness: GpuHarness,
  [width, height]: [number, number],
  valueAt: (x: number, y: number) => number,
): GPUTexture {
  const texels = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      texels[(y * width + x) * 4] = valueAt(x, y);
    }
  }
  return harness.createTexture(width, height, texels);
}

function vec2Texture(
  harness: GpuHarness,
  [width, height]: [number, number],
  valueAt: (x: number, y: number) => [number, number],
): GPUTexture {
  const texels = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [vx, vy] = valueAt(x, y);
      texels[(y * width + x) * 4] = vx;
      texels[(y * width + x) * 4 + 1] = vy;
    }
  }
  return harness.createTexture(width, height, texels);
}

describe("frame interpolation", () => {
  let harness: GpuHarness;

  beforeAll(async () => {
    harness = await createGpuHarness();
  });

  describe("setup", () => {
    async function runSetup(
      renderSize: [number, number],
      sceneChanged: number,
      startingFrameIndex: number,
      reset = 0,
    ) {
      const [width, height] = renderSize;
      const pixels = width * height;
      const [ofWidth, ofHeight] = opticalFlowGridSize(renderSize);

      const paramsBuffer = harness.createUniformBuffer(frameInterpolationParams({ renderSize, reset }));
      const gameField = harness.createStorageBuffer(new Uint32Array(pixels * 2).fill(0xdeadbeef));
      const opticalFlowField = harness.createStorageBuffer(
        new Uint32Array(ofWidth * ofHeight * 2).fill(0xdeadbeef),
      );
      const depthPrevious = harness.createStorageBuffer(new Uint32Array(pixels).fill(0));
      const depthInterpolated = harness.createStorageBuffer(new Uint32Array(pixels).fill(0));
      const sceneChange = harness.createStorageBuffer(new Uint32Array([sceneChanged]));
      const state = harness.createStorageBuffer(new Uint32Array([startingFrameIndex]));

      const pipeline = harness.createComputePipeline(setupWgsl);
      await harness.dispatch(
        pipeline,
        [
          { buffer: paramsBuffer },
          { buffer: gameField },
          { buffer: opticalFlowField },
          { buffer: depthPrevious },
          { buffer: depthInterpolated },
          { buffer: sceneChange },
          { buffer: state },
        ],
        [groupCount(width), groupCount(height)],
      );

      return {
        gameField: new Uint32Array(await harness.readBuffer(gameField, pixels * 8)),
        opticalFlowField: new Uint32Array(
          await harness.readBuffer(opticalFlowField, ofWidth * ofHeight * 8),
        ),
        depthPrevious: new Uint32Array(await harness.readBuffer(depthPrevious, pixels * 4)),
        depthInterpolated: new Uint32Array(await harness.readBuffer(depthInterpolated, pixels * 4)),
        frameIndex: new Uint32Array(await harness.readBuffer(state, 4))[0],
      };
    }

    it("clears every scattered-into resource and advances the frame counter", async () => {
      const result = await runSetup([9, 5], 0, 7);

      expect(Array.from(result.gameField).every((entry) => entry === 0)).toBe(true);
      expect(Array.from(result.opticalFlowField).every((entry) => entry === 0)).toBe(true);
      expect(Array.from(result.depthPrevious).every((bits) => bits === FAR_SENTINEL_BITS)).toBe(true);
      expect(Array.from(result.depthInterpolated).every((bits) => bits === FAR_SENTINEL_BITS)).toBe(
        true,
      );
      expect(result.frameIndex).toBe(8);
    });

    it("resets the frame counter on a scene change and on a host reset", async () => {
      expect((await runSetup([8, 8], 1, 7)).frameIndex).toBe(0);
      expect((await runSetup([8, 8], 0, 7, 1)).frameIndex).toBe(0);
    });
  });

  describe("reconstruct and dilate", () => {
    async function runReconstructAndDilate(
      renderSize: [number, number],
      depthAt: (x: number, y: number) => number,
      motionVectorAt: (x: number, y: number) => [number, number],
    ) {
      const [width, height] = renderSize;
      const pixels = width * height;

      const paramsBuffer = harness.createUniformBuffer(frameInterpolationParams({ renderSize }));
      const depthTexture = scalarTexture(harness, renderSize, depthAt);
      const motionVectorTexture = vec2Texture(harness, renderSize, motionVectorAt);
      const dilatedDepth = harness.createStorageBuffer(new Float32Array(pixels));
      const dilatedMotionVectors = harness.createStorageBuffer(new Float32Array(pixels * 2));
      const depthPrevious = harness.createStorageBuffer(
        new Uint32Array(pixels).fill(FAR_SENTINEL_BITS),
      );

      const pipeline = harness.createComputePipeline(reconstructAndDilateWgsl);
      await harness.dispatch(
        pipeline,
        [
          { buffer: paramsBuffer },
          depthTexture.createView(),
          motionVectorTexture.createView(),
          { buffer: dilatedDepth },
          { buffer: dilatedMotionVectors },
          { buffer: depthPrevious },
        ],
        [groupCount(width), groupCount(height)],
      );

      return {
        dilatedDepth: new Float32Array(await harness.readBuffer(dilatedDepth, pixels * 4)),
        dilatedMotionVectors: new Float32Array(
          await harness.readBuffer(dilatedMotionVectors, pixels * 8),
        ),
        depthPrevious: new Float32Array(await harness.readBuffer(depthPrevious, pixels * 4)),
      };
    }

    it("dilates depth and motion vectors to the nearest surface in a 3x3 window", async () => {
      const renderSize: [number, number] = [5, 5];
      // A single near pixel at (2,2) against a far background; it should claim
      // its whole 3x3 neighbourhood, and drag its own motion vector with it.
      const result = await runReconstructAndDilate(
        renderSize,
        (x, y) => (x === 2 && y === 2 ? 0.2 : 0.8),
        (x, y) => (x === 2 && y === 2 ? [3, 0] : [0, 0]),
      );

      for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
          const index = y * 5 + x;
          const touchesNearPixel = Math.abs(x - 2) <= 1 && Math.abs(y - 2) <= 1;
          expect(result.dilatedDepth[index]).toBeCloseTo(touchesNearPixel ? 0.2 : 0.8, 6);
          expect(result.dilatedMotionVectors[index * 2]).toBeCloseTo(
            touchesNearPixel ? 3 / 5 : 0,
            6,
          );
        }
      }
    });

    it("leaves the reconstructed depth at its sentinel when nothing moves", async () => {
      const result = await runReconstructAndDilate([4, 4], () => 0.5, () => [0, 0]);

      expect(Array.from(result.depthPrevious).every((depth) => depth === 1)).toBe(true);
    });

    it("scatters depth to the previous-frame position a whole-pixel motion lands on", async () => {
      const renderSize: [number, number] = [7, 3];
      // A whole-pixel translation puts the reprojection exactly on a texel
      // centre, so one bilinear tap carries all the weight and the landing
      // position is exact. Flat depth keeps the dilation an identity, leaving
      // the two columns nothing reprojects into still holding the sentinel.
      const result = await runReconstructAndDilate(renderSize, () => 0.5, () => [2, 0]);

      for (let y = 0; y < 3; y++) {
        for (let x = 0; x < 7; x++) {
          expect(result.depthPrevious[y * 7 + x]).toBeCloseTo(x < 2 ? 1 : 0.5, 5);
        }
      }
    });

    it("keeps the nearest depth when two surfaces scatter into the same pixel", async () => {
      const renderSize: [number, number] = [6, 1];
      // Pixel 0 (far) moves right by 3; pixel 2 (near) moves right by 1. Both
      // land on pixel 3, and the nearest one has to win the atomic min.
      const result = await runReconstructAndDilate(
        renderSize,
        (x) => (x === 2 ? 0.25 : 0.75),
        (x) => (x === 0 ? [3, 0] : [1, 0]),
      );

      expect(result.depthPrevious[3]).toBeCloseTo(0.25, 5);
    });
  });
});
