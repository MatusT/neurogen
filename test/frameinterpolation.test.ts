import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import setupWgsl from "../src/wgsl/generated/frameinterpolation/setup.wgsl.js";
import reconstructAndDilateWgsl from "../src/wgsl/generated/frameinterpolation/reconstruct_and_dilate.wgsl.js";
import gameMotionVectorFieldWgsl from "../src/wgsl/generated/frameinterpolation/game_motion_vector_field.wgsl.js";
import opticalFlowVectorFieldWgsl from "../src/wgsl/generated/frameinterpolation/optical_flow_vector_field.wgsl.js";
import disocclusionMaskWgsl from "../src/wgsl/generated/frameinterpolation/disocclusion_mask.wgsl.js";
import preliminaryBlendWgsl from "../src/wgsl/generated/frameinterpolation/preliminary_blend.wgsl.js";
import inpaintingPyramidWgsl from "../src/wgsl/generated/frameinterpolation/inpainting_pyramid.wgsl.js";
import finalBlendWgsl from "../src/wgsl/generated/frameinterpolation/final_blend.wgsl.js";

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

function frameInterpolationParams(overrides: ParamsOverrides): Uint8Array<ArrayBuffer> {
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

// Decodes one motion-vector-field cell, mirroring fiUnpackVectorField in
// src/wgsl/frameinterpolation/common.wgsl.
function unpackHalf(bits: number): number {
  const sign = bits >>> 15 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  return sign * (mantissa + 1024) * 2 ** (exponent - 25);
}

interface VectorFieldEntry {
  motionVector: [number, number];
  highPriority: number;
  lowPriority: number;
  primary: boolean;
  valid: boolean;
}

function unpackVectorField(field: Uint32Array, index: number): VectorFieldEntry {
  const packedX = field[index * 2];
  const packedY = field[index * 2 + 1];
  const highPriority = (packedX >>> 21) & 0x3ff;
  return {
    motionVector: [unpackHalf(packedX & 0xffff), unpackHalf(packedY & 0xffff)],
    highPriority,
    lowPriority: (packedX >>> 16) & 0x1f,
    primary: (packedX & 0x80000000) !== 0,
    valid: highPriority > 0,
  };
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

function colorTexture(
  harness: GpuHarness,
  [width, height]: [number, number],
  valueAt: (x: number, y: number) => [number, number, number],
): GPUTexture {
  const texels = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = valueAt(x, y);
      texels[(y * width + x) * 4] = r;
      texels[(y * width + x) * 4 + 1] = g;
      texels[(y * width + x) * 4 + 2] = b;
      texels[(y * width + x) * 4 + 3] = 1;
    }
  }
  return harness.createTexture(width, height, texels);
}

// Mirrors fiPyramidMipSize / fiPyramidMipOffset in common.wgsl.
const INPAINTING_MIP_COUNT = 4;

function pyramidTexelCount([width, height]: [number, number]): number {
  let total = 0;
  for (let level = 0; level < INPAINTING_MIP_COUNT; level++) {
    total += Math.max(width >> (level + 1), 1) * Math.max(height >> (level + 1), 1);
  }
  return total;
}

interface Frame {
  currentColorAt: (x: number, y: number) => [number, number, number];
  previousColorAt: (x: number, y: number) => [number, number, number];
  depthAt: (x: number, y: number) => number;
  motionVectorPixelsAt: (x: number, y: number) => [number, number];
  opticalFlowPixelsAt?: (x: number, y: number) => [number, number];
  sceneChanged?: boolean;
  reset?: boolean;
}

// The whole module wired up in the dispatch order its passes document, holding
// the temporal resources across calls so a sequence of frames exercises the
// same state a real `prepare()` loop would.
class Pipeline {
  private readonly paramsPerMip: GPUBuffer[];
  private readonly sceneChange: GPUBuffer;
  private readonly buffers: Record<string, GPUBuffer>;
  private readonly pipelines: Record<string, GPUComputePipeline>;

  constructor(
    private readonly harness: GpuHarness,
    private readonly renderSize: [number, number],
  ) {
    const [width, height] = renderSize;
    const pixels = width * height;
    const [gridWidth, gridHeight] = opticalFlowGridSize(renderSize);

    this.paramsPerMip = Array.from({ length: INPAINTING_MIP_COUNT }, (_, inpaintingMipLevel) =>
      harness.createUniformBuffer(frameInterpolationParams({ renderSize, inpaintingMipLevel })),
    );
    this.sceneChange = harness.createStorageBuffer(new Uint32Array(1));
    this.buffers = {
      gameField: harness.createStorageBuffer(new Uint32Array(pixels * 2)),
      opticalFlowField: harness.createStorageBuffer(new Uint32Array(gridWidth * gridHeight * 2)),
      depthPrevious: harness.createStorageBuffer(new Uint32Array(pixels)),
      depthInterpolated: harness.createStorageBuffer(new Uint32Array(pixels)),
      state: harness.createStorageBuffer(new Uint32Array(1)),
      dilatedDepth: harness.createStorageBuffer(new Float32Array(pixels)),
      dilatedMotionVectors: harness.createStorageBuffer(new Float32Array(pixels * 2)),
      disocclusionMask: harness.createStorageBuffer(new Float32Array(pixels * 2)),
      preliminaryColor: harness.createStorageBuffer(new Float32Array(pixels * 4)),
      blendWeight: harness.createStorageBuffer(new Float32Array(pixels)),
      inpaintingPyramid: harness.createStorageBuffer(
        new Float32Array(pyramidTexelCount(renderSize) * 4),
      ),
      interpolatedColor: harness.createStorageBuffer(new Float32Array(pixels * 4)),
    };
    this.pipelines = {
      setup: harness.createComputePipeline(setupWgsl),
      reconstructAndDilate: harness.createComputePipeline(reconstructAndDilateWgsl),
      gameMotionVectorField: harness.createComputePipeline(gameMotionVectorFieldWgsl),
      opticalFlowVectorField: harness.createComputePipeline(opticalFlowVectorFieldWgsl),
      disocclusionMask: harness.createComputePipeline(disocclusionMaskWgsl),
      preliminaryBlend: harness.createComputePipeline(preliminaryBlendWgsl),
      inpaintingPyramid: harness.createComputePipeline(inpaintingPyramidWgsl),
      finalBlend: harness.createComputePipeline(finalBlendWgsl),
    };
  }

  async run(frame: Frame): Promise<void> {
    const [width, height] = this.renderSize;
    const [gridWidth, gridHeight] = opticalFlowGridSize(this.renderSize);
    const cells = gridWidth * gridHeight;
    const { device } = this.harness;
    const b = this.buffers;

    device.queue.writeBuffer(
      this.sceneChange,
      0,
      new Uint32Array([frame.sceneChanged ? 1 : 0]),
    );
    const params = frameInterpolationParams({
      renderSize: this.renderSize,
      reset: frame.reset ? 1 : 0,
    });
    device.queue.writeBuffer(this.paramsPerMip[0], 0, params);

    const currentColor = colorTexture(this.harness, this.renderSize, frame.currentColorAt).createView();
    const previousColor = colorTexture(this.harness, this.renderSize, frame.previousColorAt).createView();
    const depth = scalarTexture(this.harness, this.renderSize, frame.depthAt).createView();
    const motionVectors = vec2Texture(this.harness, this.renderSize, frame.motionVectorPixelsAt).createView();

    const flowData = new Int32Array(cells * 2);
    const validityData = new Uint32Array(cells);
    for (let y = 0; y < gridHeight; y++) {
      for (let x = 0; x < gridWidth; x++) {
        const [vx, vy] = frame.opticalFlowPixelsAt?.(x, y) ?? [0, 0];
        flowData[(y * gridWidth + x) * 2] = vx;
        flowData[(y * gridWidth + x) * 2 + 1] = vy;
        validityData[y * gridWidth + x] = 1;
      }
    }
    const opticalFlow = this.harness.createStorageBuffer(flowData);
    const opticalFlowValidity = this.harness.createStorageBuffer(validityData);

    const base = { buffer: this.paramsPerMip[0] };
    const frameGroups: [number, number] = [groupCount(width), groupCount(height)];

    await this.harness.dispatch(
      this.pipelines.setup,
      [base, { buffer: b.gameField }, { buffer: b.opticalFlowField }, { buffer: b.depthPrevious },
        { buffer: b.depthInterpolated }, { buffer: this.sceneChange }, { buffer: b.state }],
      frameGroups,
    );
    await this.harness.dispatch(
      this.pipelines.reconstructAndDilate,
      [base, depth, motionVectors, { buffer: b.dilatedDepth }, { buffer: b.dilatedMotionVectors },
        { buffer: b.depthPrevious }],
      frameGroups,
    );
    await this.harness.dispatch(
      this.pipelines.gameMotionVectorField,
      [base, { buffer: b.dilatedDepth }, { buffer: b.dilatedMotionVectors }, currentColor,
        previousColor, { buffer: b.gameField }, { buffer: b.depthInterpolated }],
      frameGroups,
    );
    await this.harness.dispatch(
      this.pipelines.opticalFlowVectorField,
      [base, { buffer: opticalFlow }, { buffer: opticalFlowValidity }, currentColor, previousColor,
        { buffer: b.opticalFlowField }],
      [groupCount(gridWidth), groupCount(gridHeight)],
    );
    await this.harness.dispatch(
      this.pipelines.disocclusionMask,
      [base, { buffer: b.depthInterpolated }, { buffer: b.depthPrevious }, { buffer: b.dilatedDepth },
        { buffer: b.gameField }, { buffer: b.disocclusionMask }],
      frameGroups,
    );
    await this.harness.dispatch(
      this.pipelines.preliminaryBlend,
      [base, currentColor, previousColor, { buffer: b.gameField }, { buffer: b.opticalFlowField },
        { buffer: b.disocclusionMask }, { buffer: b.preliminaryColor }, { buffer: b.blendWeight },
        { buffer: b.state }],
      frameGroups,
    );
    for (let level = 0; level < INPAINTING_MIP_COUNT; level++) {
      await this.harness.dispatch(
        this.pipelines.inpaintingPyramid,
        [{ buffer: this.paramsPerMip[level] }, { buffer: b.preliminaryColor },
          { buffer: b.blendWeight }, { buffer: b.inpaintingPyramid }],
        [
          groupCount(Math.max(width >> (level + 1), 1)),
          groupCount(Math.max(height >> (level + 1), 1)),
        ],
      );
    }
    await this.harness.dispatch(
      this.pipelines.finalBlend,
      [base, currentColor, { buffer: b.preliminaryColor }, { buffer: b.blendWeight },
        { buffer: b.inpaintingPyramid }, { buffer: b.interpolatedColor }, { buffer: b.state }],
      frameGroups,
    );
  }

  async read(name: string, elements: number): Promise<Float32Array> {
    return new Float32Array(await this.harness.readBuffer(this.buffers[name], elements * 4));
  }

  async interpolatedColor(): Promise<Float32Array> {
    const [width, height] = this.renderSize;
    return this.read("interpolatedColor", width * height * 4);
  }
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

    it("scatters a stationary pixel's depth onto its own location", async () => {
      const result = await runReconstructAndDilate([4, 4], () => 0.5, () => [0, 0]);

      // Sub-pixel motion is snapped to zero but still written. Leaving the far
      // sentinel here instead would make a stationary occluder invisible to the
      // disocclusion pass, which reads this buffer to ask what stood in front
      // of a pixel in the previous frame.
      expect(Array.from(result.depthPrevious).every((depth) => depth === 0.5)).toBe(true);
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

  describe("game motion vector field", () => {
    async function runGameMotionVectorField(
      renderSize: [number, number],
      depthAt: (x: number, y: number) => number,
      motionVectorPixelsAt: (x: number, y: number) => [number, number],
    ) {
      const [width, height] = renderSize;
      const pixels = width * height;

      const dilatedDepthData = new Float32Array(pixels);
      const dilatedMotionVectorData = new Float32Array(pixels * 2);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = y * width + x;
          dilatedDepthData[index] = depthAt(x, y);
          const [vx, vy] = motionVectorPixelsAt(x, y);
          dilatedMotionVectorData[index * 2] = vx / width;
          dilatedMotionVectorData[index * 2 + 1] = vy / height;
        }
      }

      const paramsBuffer = harness.createUniformBuffer(frameInterpolationParams({ renderSize }));
      const flatColor = scalarTexture(harness, renderSize, () => 0.5);
      const dilatedDepth = harness.createStorageBuffer(dilatedDepthData);
      const dilatedMotionVectors = harness.createStorageBuffer(dilatedMotionVectorData);
      const gameField = harness.createStorageBuffer(new Uint32Array(pixels * 2));
      const depthInterpolated = harness.createStorageBuffer(
        new Uint32Array(pixels).fill(FAR_SENTINEL_BITS),
      );

      const pipeline = harness.createComputePipeline(gameMotionVectorFieldWgsl);
      await harness.dispatch(
        pipeline,
        [
          { buffer: paramsBuffer },
          { buffer: dilatedDepth },
          { buffer: dilatedMotionVectors },
          flatColor.createView(),
          flatColor.createView(),
          { buffer: gameField },
          { buffer: depthInterpolated },
        ],
        [groupCount(width), groupCount(height)],
      );

      return {
        gameField: new Uint32Array(await harness.readBuffer(gameField, pixels * 8)),
        depthInterpolated: new Float32Array(await harness.readBuffer(depthInterpolated, pixels * 4)),
      };
    }

    it("fills every cell with a stationary primary vector when nothing moves", async () => {
      const renderSize: [number, number] = [8, 4];
      const result = await runGameMotionVectorField(renderSize, () => 0.5, () => [0, 0]);

      for (let index = 0; index < 8 * 4; index++) {
        const entry = unpackVectorField(result.gameField, index);
        expect(entry.valid).toBe(true);
        expect(entry.primary).toBe(true);
        expect(entry.motionVector).toEqual([0, 0]);
        // Identical current and previous colour: the luma ratio is exactly 1.
        expect(entry.lowPriority).toBe(31);
      }
    });

    it("gives a nearer surface a higher depth priority than a farther one", async () => {
      const renderSize: [number, number] = [8, 4];
      const result = await runGameMotionVectorField(
        renderSize,
        (x) => (x < 4 ? 0.2 : 0.9),
        () => [0, 0],
      );

      const near = unpackVectorField(result.gameField, 1).highPriority;
      const far = unpackVectorField(result.gameField, 6).highPriority;
      expect(near).toBeGreaterThan(far);
      expect(far).toBeGreaterThan(0);
    });

    it("scatters half the motion vector to the interpolated frame's position", async () => {
      const renderSize: [number, number] = [8, 4];
      // 4px of motion, so entries land 2px right of their source and the two
      // leftmost columns have nothing reaching them. The half vector is 0.25 in
      // UV, exactly representable as a half float.
      const result = await runGameMotionVectorField(renderSize, () => 0.5, () => [4, 0]);

      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
          const entry = unpackVectorField(result.gameField, y * 8 + x);
          expect(entry.valid).toBe(x >= 2);
          if (!entry.valid) continue;

          expect(entry.motionVector[0]).toBeCloseTo(0.25, 6);
          expect(entry.motionVector[1]).toBeCloseTo(0, 6);
        }
      }
    });

    it("reconstructs the interpolated frame's depth at the half-vector position", async () => {
      const renderSize: [number, number] = [8, 4];
      const result = await runGameMotionVectorField(renderSize, () => 0.5, () => [4, 0]);

      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
          expect(result.depthInterpolated[y * 8 + x]).toBeCloseTo(x < 2 ? 1 : 0.5, 5);
        }
      }
    });
  });

  describe("optical flow vector field", () => {
    async function runOpticalFlowVectorField(
      renderSize: [number, number],
      flowPixelsAt: (x: number, y: number) => [number, number],
      validAt: (x: number, y: number) => number,
    ) {
      const [width, height] = renderSize;
      const [gridWidth, gridHeight] = opticalFlowGridSize(renderSize);
      const cells = gridWidth * gridHeight;

      const flowData = new Int32Array(cells * 2);
      const validityData = new Uint32Array(cells);
      for (let y = 0; y < gridHeight; y++) {
        for (let x = 0; x < gridWidth; x++) {
          const [vx, vy] = flowPixelsAt(x, y);
          flowData[(y * gridWidth + x) * 2] = vx;
          flowData[(y * gridWidth + x) * 2 + 1] = vy;
          validityData[y * gridWidth + x] = validAt(x, y);
        }
      }

      const paramsBuffer = harness.createUniformBuffer(frameInterpolationParams({ renderSize }));
      const flatColor = scalarTexture(harness, renderSize, () => 0.5);
      const flow = harness.createStorageBuffer(flowData);
      const validity = harness.createStorageBuffer(validityData);
      const field = harness.createStorageBuffer(new Uint32Array(cells * 2));

      const pipeline = harness.createComputePipeline(opticalFlowVectorFieldWgsl);
      await harness.dispatch(
        pipeline,
        [
          { buffer: paramsBuffer },
          { buffer: flow },
          { buffer: validity },
          flatColor.createView(),
          flatColor.createView(),
          { buffer: field },
        ],
        [groupCount(gridWidth), groupCount(gridHeight)],
      );

      return new Uint32Array(await harness.readBuffer(field, cells * 8));
    }

    it("scatters a uniform flow as a full-priority primary vector", async () => {
      const renderSize: [number, number] = [64, 32];
      // 16px of flow across an 8x4 cell grid: the half vector is 0.125 in UV,
      // which is one whole cell, so entries land exactly one cell to the right.
      const field = await runOpticalFlowVectorField(renderSize, () => [16, 0], () => 1);

      const entry = unpackVectorField(field, 1 * 8 + 4);
      expect(entry.primary).toBe(true);
      expect(entry.highPriority).toBe(1023);
      expect(entry.motionVector[0]).toBeCloseTo(0.125, 6);
      expect(entry.motionVector[1]).toBeCloseTo(0, 6);

      // Nothing reprojects into the leftmost column.
      expect(unpackVectorField(field, 1 * 8 + 0).valid).toBe(false);
    });

    it("ignores cells Task 2 marked invalid", async () => {
      const renderSize: [number, number] = [64, 32];
      const field = await runOpticalFlowVectorField(renderSize, () => [16, 0], () => 0);

      expect(Array.from(field).every((entry) => entry === 0)).toBe(true);
    });
  });

  describe("disocclusion mask", () => {
    // A stationary, valid, primary field entry — the only shape these fixtures
    // need, so the half-float coefficients are both zero.
    const STATIONARY_PRIMARY = (0x80000000 | (512 << 21) | (31 << 16)) >>> 0;

    async function runDisocclusionMask(
      renderSize: [number, number],
      interpolatedDepthAt: (x: number, y: number) => number,
      previousDepthAt: (x: number, y: number) => number,
    ) {
      const [width, height] = renderSize;
      const pixels = width * height;

      const interpolatedDepthData = new Float32Array(pixels);
      const previousDepthData = new Float32Array(pixels);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          interpolatedDepthData[y * width + x] = interpolatedDepthAt(x, y);
          previousDepthData[y * width + x] = previousDepthAt(x, y);
        }
      }

      const paramsBuffer = harness.createUniformBuffer(frameInterpolationParams({ renderSize }));
      const depthInterpolated = harness.createStorageBuffer(interpolatedDepthData);
      const depthPrevious = harness.createStorageBuffer(previousDepthData);
      // The current frame's estimate is the dilated depth; keeping it equal to
      // the interpolated depth isolates the previous-frame channel.
      const dilatedDepth = harness.createStorageBuffer(interpolatedDepthData);
      const gameField = harness.createStorageBuffer(
        new Uint32Array(pixels * 2).fill(STATIONARY_PRIMARY),
      );
      const mask = harness.createStorageBuffer(new Float32Array(pixels * 2));

      const pipeline = harness.createComputePipeline(disocclusionMaskWgsl);
      await harness.dispatch(
        pipeline,
        [
          { buffer: paramsBuffer },
          { buffer: depthInterpolated },
          { buffer: depthPrevious },
          { buffer: dilatedDepth },
          { buffer: gameField },
          { buffer: mask },
        ],
        [groupCount(width), groupCount(height)],
      );

      return new Float32Array(await harness.readBuffer(mask, pixels * 8));
    }

    it("reports both frames visible where the depths agree", async () => {
      const mask = await runDisocclusionMask([8, 4], () => 0.5, () => 0.5);

      expect(Array.from(mask).every((component) => component === 1)).toBe(true);
    });

    it("marks a pixel hidden behind nearer geometry in the previous frame", async () => {
      const renderSize: [number, number] = [8, 4];
      // The right half of the previous frame held something much nearer, so the
      // surface this pixel wants to sample was occluded there.
      const mask = await runDisocclusionMask(renderSize, () => 0.5, (x) => (x < 4 ? 0.5 : 0.1));

      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 8; x++) {
          expect(mask[(y * 8 + x) * 2]).toBe(x < 4 ? 1 : 0);
          // The current-frame channel compares against an identical depth and
          // must stay visible either way.
          expect(mask[(y * 8 + x) * 2 + 1]).toBe(1);
        }
      }
    });

    it("treats a depth gap below the Ksep separation as the same surface", async () => {
      const renderSize: [number, number] = [8, 4];
      // 6e-5 of device depth is about 2.4e-5 of view-space depth here, under the
      // ~4.0e-5 the Ksep threshold requires before two depths count as separate
      // surfaces, so this must read as visible rather than disoccluded.
      //
      // The value discriminates deliberately: the unreachable
      // `ComputeSampleDepthClip` variant in AMD's source, which the task brief
      // quoted, would require only ~1.3e-5 here and would call this a
      // disocclusion. If this ever starts failing, the formula was swapped.
      const mask = await runDisocclusionMask(renderSize, () => 0.5, () => 0.5 - 6e-5);

      expect(Array.from(mask).every((component) => component === 1)).toBe(true);
    });
  });

  describe("inpainting", () => {
    const RENDER_SIZE: [number, number] = [16, 16];
    const SURROUND: [number, number, number] = [0.5, 0.25, 0.75];
    const HOLE_MIN = 6;
    const HOLE_MAX = 10;

    const isHole = (x: number, y: number) =>
      x >= HOLE_MIN && x < HOLE_MAX && y >= HOLE_MIN && y < HOLE_MAX;

    // A frame of one flat colour with a black square punched out of it, and a
    // blend weight marking exactly that square. Anything the inpainting
    // produces has to be the surround colour, because it is the only colour any
    // covered pixel holds — which also makes a plain box filter visibly wrong.
    async function runInpainting() {
      const [width, height] = RENDER_SIZE;
      const pixels = width * height;

      const colorData = new Float32Array(pixels * 4);
      const weightData = new Float32Array(pixels);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = y * width + x;
          weightData[index] = isHole(x, y) ? 1 : 0;
          const rgb = isHole(x, y) ? [0, 0, 0] : SURROUND;
          colorData.set([...rgb, 1], index * 4);
        }
      }

      const preliminaryColor = harness.createStorageBuffer(colorData);
      const blendWeight = harness.createStorageBuffer(weightData);
      const pyramid = harness.createStorageBuffer(
        new Float32Array(pyramidTexelCount(RENDER_SIZE) * 4),
      );
      const interpolatedColor = harness.createStorageBuffer(new Float32Array(pixels * 4));
      // Any frame but the first, or the scene-cut fallback takes over.
      const state = harness.createStorageBuffer(new Uint32Array([5]));
      const currentColor = scalarTexture(harness, RENDER_SIZE, () => 0).createView();

      const pyramidPipeline = harness.createComputePipeline(inpaintingPyramidWgsl);
      for (let level = 0; level < INPAINTING_MIP_COUNT; level++) {
        const mipParams = harness.createUniformBuffer(
          frameInterpolationParams({ renderSize: RENDER_SIZE, inpaintingMipLevel: level }),
        );
        await harness.dispatch(
          pyramidPipeline,
          [{ buffer: mipParams }, { buffer: preliminaryColor }, { buffer: blendWeight },
            { buffer: pyramid }],
          [groupCount(width >> (level + 1)), groupCount(height >> (level + 1))],
        );
      }

      await harness.dispatch(
        harness.createComputePipeline(finalBlendWgsl),
        [
          { buffer: harness.createUniformBuffer(frameInterpolationParams({ renderSize: RENDER_SIZE })) },
          currentColor,
          { buffer: preliminaryColor },
          { buffer: blendWeight },
          { buffer: pyramid },
          { buffer: interpolatedColor },
          { buffer: state },
        ],
        [groupCount(width), groupCount(height)],
      );

      return {
        pyramid: new Float32Array(
          await harness.readBuffer(pyramid, pyramidTexelCount(RENDER_SIZE) * 16),
        ),
        result: new Float32Array(await harness.readBuffer(interpolatedColor, pixels * 16)),
      };
    }

    it("weights the pyramid reduction by coverage instead of averaging holes in", async () => {
      const { pyramid } = await runInpainting();

      // Mip 0 is 8x8. Its texels 3 and 4 in each axis sit wholly inside the
      // hole and must carry no coverage at all.
      for (const axisPos of [3, 4]) {
        expect(Array.from(pyramid.slice((axisPos * 8 + axisPos) * 4, (axisPos * 8 + axisPos) * 4 + 4)))
          .toEqual([0, 0, 0, 0]);
      }

      // Texel (2,3) straddles the hole edge: two covered taps, two not. A box
      // filter would halve the colour; the coverage weighting keeps it whole.
      const straddling = (3 * 8 + 2) * 4;
      expect(pyramid[straddling]).toBeCloseTo(SURROUND[0], 6);
      expect(pyramid[straddling + 1]).toBeCloseTo(SURROUND[1], 6);
      expect(pyramid[straddling + 3]).toBeGreaterThan(0);
    });

    it("fills a hole from the surrounding colour and leaves covered pixels alone", async () => {
      const { result } = await runInpainting();

      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          const base = (y * 16 + x) * 4;
          const actual = [result[base], result[base + 1], result[base + 2]];
          // Both cases land on the same colour: covered pixels because that is
          // what they already held, hole pixels because inpainting supplied it.
          for (let channel = 0; channel < 3; channel++) {
            expect(actual[channel]).toBeCloseTo(SURROUND[channel], 5);
          }
        }
      }
    });

    it("passes a pixel through untouched when its blend weight is zero", async () => {
      // Nothing distinguishes a zero-weight pixel from an inpainted one in the
      // fixture above, since both end up the surround colour. Give the pyramid
      // nothing to offer and the covered pixels must still survive.
      const [width, height] = RENDER_SIZE;
      const pixels = width * height;
      const colorData = new Float32Array(pixels * 4);
      for (let index = 0; index < pixels; index++) {
        colorData.set([index / pixels, 0.5, 0.25, 1], index * 4);
      }

      const preliminaryColor = harness.createStorageBuffer(colorData);
      const blendWeight = harness.createStorageBuffer(new Float32Array(pixels));
      const pyramid = harness.createStorageBuffer(
        new Float32Array(pyramidTexelCount(RENDER_SIZE) * 4),
      );
      const interpolatedColor = harness.createStorageBuffer(new Float32Array(pixels * 4));

      await harness.dispatch(
        harness.createComputePipeline(finalBlendWgsl),
        [
          { buffer: harness.createUniformBuffer(frameInterpolationParams({ renderSize: RENDER_SIZE })) },
          scalarTexture(harness, RENDER_SIZE, () => 0).createView(),
          { buffer: preliminaryColor },
          { buffer: blendWeight },
          { buffer: pyramid },
          { buffer: interpolatedColor },
          { buffer: harness.createStorageBuffer(new Uint32Array([5])) },
        ],
        [groupCount(width), groupCount(height)],
      );

      const result = new Float32Array(await harness.readBuffer(interpolatedColor, pixels * 16));
      for (let index = 0; index < pixels; index++) {
        expect(result[index * 4]).toBeCloseTo(index / pixels, 6);
      }
    });
  });

  describe("full pipeline", () => {
    const FLAT_DEPTH = 0.5;
    // No two columns alike, so a one-pixel misalignment shows up, and offset
    // far enough from zero that a shifted lookup stays a valid colour.
    const gradient = (x: number, y: number): [number, number, number] => [
      (x + 32) / 128,
      (y + 32) / 128,
      0.25,
    ];

    function colorAt(image: Float32Array, width: number, x: number, y: number): number[] {
      const base = (y * width + x) * 4;
      return [image[base], image[base + 1], image[base + 2]];
    }

    // The shader works in f32 and, at non-power-of-two sizes, through bilinear
    // weights that miss an exact 1.0 by an ulp, so exact equality is the wrong
    // bar even for a passthrough.
    function expectColorClose(actual: number[], expected: number[]): void {
      for (let channel = 0; channel < 3; channel++) {
        expect(actual[channel]).toBeCloseTo(expected[channel], 5);
      }
    }

    it("reproduces the source frame exactly when nothing moves", async () => {
      const renderSize: [number, number] = [16, 8];
      const pipeline = new Pipeline(harness, renderSize);
      const stationary = {
        currentColorAt: gradient,
        previousColorAt: gradient,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: (): [number, number] => [0, 0],
      };

      // The first frame of a sequence has no usable previous frame, so
      // orchestration must flag it as a reset.
      await pipeline.run({ ...stationary, reset: true });
      await pipeline.run(stationary);
      const result = await pipeline.interpolatedColor();

      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 16; x++) {
          expectColorClose(colorAt(result, 16, x, y), gradient(x, y));
        }
      }
    });

    it("lands a uniform translation halfway between the two frames", async () => {
      const renderSize: [number, number] = [32, 16];
      const shift = 2;
      // Content moved `shift` pixels right, so a current-frame pixel was `shift`
      // to its left in the previous frame.
      const previousColorAt = gradient;
      const currentColorAt = (x: number, y: number): [number, number, number] =>
        x >= shift ? gradient(x - shift, y) : [0, 0, 0];
      const frame = {
        currentColorAt,
        previousColorAt,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: (): [number, number] => [-shift, 0],
        opticalFlowPixelsAt: (): [number, number] => [-shift, 0],
      };

      const pipeline = new Pipeline(harness, renderSize);
      await pipeline.run({ ...frame, reset: true });
      await pipeline.run(frame);
      const result = await pipeline.interpolatedColor();

      // The interpolated frame sits half a step along, so it holds the source
      // pattern shifted by one pixel. Columns near the edges have one source
      // reprojecting off frame and are excluded.
      for (let y = 0; y < 16; y++) {
        for (let x = shift; x < 32 - shift; x++) {
          const expected = gradient(x - shift / 2, y);
          const actual = colorAt(result, 32, x, y);
          expect(actual[0]).toBeCloseTo(expected[0], 4);
          expect(actual[1]).toBeCloseTo(expected[1], 4);
        }
      }
    });

    it("handles an odd-dimensioned render target", async () => {
      const renderSize: [number, number] = [13, 7];
      const pipeline = new Pipeline(harness, renderSize);
      const stationary = {
        currentColorAt: gradient,
        previousColorAt: gradient,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: (): [number, number] => [0, 0],
      };

      await pipeline.run({ ...stationary, reset: true });
      await pipeline.run(stationary);
      const result = await pipeline.interpolatedColor();

      for (let y = 0; y < 7; y++) {
        for (let x = 0; x < 13; x++) {
          expectColorClose(colorAt(result, 13, x, y), gradient(x, y));
        }
      }
    });

    it("shows one real frame instead of interpolating across a scene cut", async () => {
      const renderSize: [number, number] = [16, 8];
      const pipeline = new Pipeline(harness, renderSize);
      const before = (): [number, number, number] => [0.9, 0.1, 0.1];
      const after = (): [number, number, number] => [0.1, 0.1, 0.9];

      await pipeline.run({
        currentColorAt: before,
        previousColorAt: before,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: () => [0, 0],
        reset: true,
      });
      await pipeline.run({
        currentColorAt: before,
        previousColorAt: before,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: () => [0, 0],
      });

      // Unrelated content on both sides — blending them would produce a colour
      // present in neither.
      await pipeline.run({
        currentColorAt: after,
        previousColorAt: before,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: () => [0, 0],
        sceneChanged: true,
      });
      const result = await pipeline.interpolatedColor();

      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 16; x++) {
          expectColorClose(colorAt(result, 16, x, y), [0.1, 0.1, 0.9]);
        }
      }
    });

    it("detects the strip a moving foreground uncovers", async () => {
      const renderSize: [number, number] = [32, 16];
      const [width, height] = renderSize;
      const shift = 8;
      const foregroundWidth = 8;
      const background: [number, number, number] = [0.8, 0.15, 0.15];
      const foreground: [number, number, number] = [0.15, 0.8, 0.15];

      // A foreground block at depth 0.2 slides right across a background at
      // depth 0.9, uncovering the columns it used to sit on.
      const blockAt = (origin: number) => (x: number): [number, number, number] =>
        x >= origin && x < origin + foregroundWidth ? foreground : background;
      const depthAt = (x: number) => (x >= shift && x < shift + foregroundWidth ? 0.2 : 0.9);
      const motionVectorPixelsAt = (x: number): [number, number] =>
        x >= shift && x < shift + foregroundWidth ? [-shift, 0] : [0, 0];

      const pipeline = new Pipeline(harness, renderSize);
      const frame = {
        currentColorAt: blockAt(shift),
        previousColorAt: blockAt(0),
        depthAt,
        motionVectorPixelsAt,
      };
      await pipeline.run({ ...frame, reset: true });
      await pipeline.run(frame);

      const mask = await pipeline.read("disocclusionMask", width * height * 2);
      const result = await pipeline.interpolatedColor();

      // The interpolated frame puts the block on columns 3..12 — half a step
      // along, widened by one on each side by the depth dilation. Columns 0..2
      // are background it has moved off, so the previous frame does not contain
      // them; columns 13..16 are background it has not reached yet, so the
      // current frame does not.
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < 3; x++) {
          expect(mask[(y * width + x) * 2]).toBe(0);
        }
        for (let x = 13; x < 17; x++) {
          expect(mask[(y * width + x) * 2 + 1]).toBe(0);
        }
        // Well clear of the block, both frames still see the background.
        for (let x = 20; x < width; x++) {
          expect(mask[(y * width + x) * 2]).toBe(1);
          expect(mask[(y * width + x) * 2 + 1]).toBe(1);
        }
      }

      // Whatever the fill, it has to stay inside the range the two source
      // frames actually contain — a NaN or a runaway divide would not.
      for (let i = 0; i < width * height; i++) {
        for (let channel = 0; channel < 3; channel++) {
          const value = result[i * 4 + channel];
          expect(Number.isFinite(value)).toBe(true);
          expect(value).toBeGreaterThanOrEqual(0.1);
          expect(value).toBeLessThanOrEqual(0.85);
        }
      }
    });

    it("stays stable across a run of consecutive frames", async () => {
      const renderSize: [number, number] = [32, 16];
      const pipeline = new Pipeline(harness, renderSize);
      const shift = 2;
      // Long enough to leave the window where the game vectors are trusted
      // unconditionally, so the last frames go through the similarity scoring
      // against the optical flow that the shorter fixtures never reach.
      const movingFrames = 11;

      for (let frameIndex = 0; frameIndex < movingFrames; frameIndex++) {
        const offset = frameIndex * shift;
        await pipeline.run({
          currentColorAt: (x, y) => gradient(x - offset, y),
          previousColorAt: (x, y) => gradient(x - offset + shift, y),
          depthAt: () => FLAT_DEPTH,
          motionVectorPixelsAt: () => [-shift, 0],
          opticalFlowPixelsAt: () => [-shift, 0],
          reset: frameIndex === 0,
        });
      }

      const frameCounter = new Uint32Array((await pipeline.read("state", 1)).buffer);
      expect(frameCounter[0]).toBe(movingFrames - 1);

      const movingResult = await pipeline.interpolatedColor();
      expect(Array.from(movingResult).every((value) => Number.isFinite(value))).toBe(true);

      // Eleven frames of accumulation must not have drifted the interior away
      // from the midpoint between the two source frames.
      const offset = (movingFrames - 1) * shift;
      for (let y = 0; y < 16; y++) {
        for (let x = shift; x < 32 - shift; x++) {
          const expected = gradient(x - offset + shift / 2, y);
          expect(movingResult[(y * 32 + x) * 4]).toBeCloseTo(expected[0], 4);
        }
      }

      // Now stop dead on the same content. Every frame so far carried identical
      // motion, so a stale field entry would have been indistinguishable from a
      // fresh one; against a stationary frame it is not. Without setup's clear
      // the previous frame's entry ties on priority and wins the half-float
      // tie-break — a negative coefficient has the high bit set — dragging the
      // gather a pixel sideways and breaking this passthrough.
      const settled = (x: number, y: number) => gradient(x - offset, y);
      await pipeline.run({
        currentColorAt: settled,
        previousColorAt: settled,
        depthAt: () => FLAT_DEPTH,
        motionVectorPixelsAt: () => [0, 0],
      });

      const stationaryResult = await pipeline.interpolatedColor();
      for (let y = 0; y < 16; y++) {
        for (let x = 0; x < 32; x++) {
          expectColorClose(colorAt(stationaryResult, 32, x, y), settled(x, y));
        }
      }
    });
  });
});
