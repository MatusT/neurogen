import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import { OpticalFlowStage } from "../src/orchestration/opticalFlow.js";
import {
  packFrameInterpolationParams,
  packOpticalFlowParams,
  TransferFunction,
} from "../src/orchestration/params.js";
import {
  flowLevelSize,
  lumaLevelSize,
  opticalFlowGridSize,
  pyramidMipOffset,
  pyramidMipSize,
  pyramidTexelCount,
  scdHistogramGroups,
  type Size,
} from "../src/orchestration/geometry.js";
import prepareLumaWgsl from "../src/wgsl/generated/opticalflow/prepare_luma.wgsl.js";
import luminancePyramidWgsl from "../src/wgsl/generated/opticalflow/luminance_pyramid.wgsl.js";
import scdHistogramWgsl from "../src/wgsl/generated/opticalflow/scd_histogram.wgsl.js";
import scdDivergenceWgsl from "../src/wgsl/generated/opticalflow/scd_divergence.wgsl.js";
import scdFinalizeWgsl from "../src/wgsl/generated/opticalflow/scd_finalize.wgsl.js";
import computeOpticalFlowWgsl from "../src/wgsl/generated/opticalflow/compute_optical_flow.wgsl.js";
import filterOpticalFlowWgsl from "../src/wgsl/generated/opticalflow/filter_optical_flow.wgsl.js";
import scaleOpticalFlowWgsl from "../src/wgsl/generated/opticalflow/scale_optical_flow.wgsl.js";

const GENERATED_PASSES: Record<string, string> = {
  prepareLuma: prepareLumaWgsl,
  luminancePyramid: luminancePyramidWgsl,
  scdHistogram: scdHistogramWgsl,
  scdDivergence: scdDivergenceWgsl,
  scdFinalize: scdFinalizeWgsl,
  computeOpticalFlow: computeOpticalFlowWgsl,
  filterOpticalFlow: filterOpticalFlowWgsl,
  scaleOpticalFlow: scaleOpticalFlowWgsl,
};

// scd_finalize.wgsl forces a scene change while frameIndex is inside its warmup
// window, and a scene change makes the search emit zeroes. Any assertion about
// real motion has to run past it.
const SCD_WARMUP_FRAMES = 5;

// A stable per-pixel value with no periodicity, so a block matches in exactly
// one place. Block matching on a smooth gradient would be ambiguous and on a
// periodic pattern would be wrong.
function noise(x: number, y: number): number {
  const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return h - Math.floor(h);
}

function noiseTexture(
  harness: GpuHarness,
  [width, height]: Size,
  shiftX: number,
): GPUTextureView {
  const texels = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = noise(x - shiftX, y);
      texels.set([value, value, value, 1], (y * width + x) * 4);
    }
  }

  return harness.createTexture(width, height, texels).createView();
}

// FrameGenerator does its own encoding rather than going through the harness's
// `dispatch()`, so nothing else would catch a mismatched binding — and a
// dispatch that never ran reads back as plausible zeroes.
async function checked<T>(harness: GpuHarness, body: () => T): Promise<T> {
  const { device } = harness;
  device.pushErrorScope("validation");
  try {
    return body();
  } finally {
    const error = await device.popErrorScope();
    if (error) {
      throw new Error(`WebGPU validation error: ${error.message}`);
    }
  }
}

async function submitPass(
  harness: GpuHarness,
  encode: (pass: GPUComputePassEncoder) => void,
): Promise<void> {
  await checked(harness, () => {
    const encoder = harness.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    encode(pass);
    pass.end();
    harness.device.queue.submit([encoder.finish()]);
  });
}

describe("geometry", () => {
  it("floor-halves luma levels and ceil-halves flow levels", () => {
    // The two conventions genuinely disagree in AMD's source and the WGSL keeps
    // them apart, so a buffer sized with the wrong one is short by a row.
    expect(lumaLevelSize([65, 33], 1)).toEqual([32, 16]);
    expect(flowLevelSize([65, 33], 0)).toEqual([9, 5]);
    expect(flowLevelSize([65, 33], 1)).toEqual([5, 3]);
  });

  it("clamps both level chains at one texel", () => {
    expect(lumaLevelSize([1, 1], 4)).toEqual([1, 1]);
    expect(pyramidMipSize([2, 2], 3)).toEqual([1, 1]);
  });

  it("lays the inpainting mips out coarsest-last in one buffer", () => {
    // Mip 0 is half resolution, not full: 32x16 + 16x8 + 8x4 + 4x2.
    expect(pyramidMipOffset([64, 32], 0)).toBe(0);
    expect(pyramidMipOffset([64, 32], 1)).toBe(512);
    expect(pyramidTexelCount([64, 32])).toBe(512 + 128 + 32 + 8);
  });

  it("covers every scene-change region column", () => {
    // scd_histogram strides four pixels per invocation across a region a third
    // of the frame wide, with 32 invocations per workgroup.
    expect(scdHistogramGroups([1920, 1080])).toEqual([5, 16, 9]);
    expect(scdHistogramGroups([64, 64])).toEqual([1, 16, 9]);
  });
});

describe("uniform packing", () => {
  it("puts renderSize first, where Task 5's kernel reads its own prefix", () => {
    const packed = packFrameInterpolationParams({
      renderSize: [1920, 1080],
      nearPlane: 0.1,
      farPlane: 100,
      verticalFovRadians: 1,
      transferFunction: TransferFunction.Pq,
      luminance: { min: 0.5, max: 1000 },
      inpaintingMipLevel: 3,
      reset: true,
    });
    const view = new DataView(packed.buffer);

    expect([view.getInt32(0, true), view.getInt32(4, true)]).toEqual([1920, 1080]);
    expect(view.getUint32(20, true)).toBe(TransferFunction.Pq);
    expect(view.getUint32(32, true)).toBe(3);
    expect(view.getUint32(36, true)).toBe(1);
    // The struct is 40 bytes; the binding is padded and must stay at least that.
    expect(packed.byteLength).toBeGreaterThanOrEqual(40);
  });

  it("packs the optical flow level and frame index where the passes read them", () => {
    const packed = packOpticalFlowParams({
      renderSize: [64, 32],
      pyramidLevel: 1,
      pyramidLevelCount: 2,
      frameIndex: 7,
      transferFunction: TransferFunction.LinearLdr,
      luminance: { min: 0, max: 1000 },
    });
    const view = new DataView(packed.buffer);

    expect(view.getUint32(8, true)).toBe(1);
    expect(view.getUint32(12, true)).toBe(2);
    expect(view.getUint32(16, true)).toBe(7);
  });
});

describe("bind group construction", () => {
  // Bind groups are built positionally — entry i binds to `@binding(i)` — which
  // is only equivalent to writing each layout out by hand while every pass
  // numbers its bindings from 0 with no gaps.
  it("finds every pass's bindings numbered contiguously from zero", () => {
    for (const [name, source] of Object.entries(GENERATED_PASSES)) {
      const declared = [...source.matchAll(/@group\(0\) @binding\((\d+)\)\s+var/g)]
        .map((match) => Number(match[1]))
        .sort((a, b) => a - b);

      expect([name, declared.length > 0]).toEqual([name, true]);
      expect([name, declared]).toEqual([name, declared.map((_, index) => index)]);
    }
  });
});

describe("orchestration", () => {
  let harness: GpuHarness;

  beforeAll(async () => {
    harness = await createGpuHarness();
  });

  describe("optical flow stage", () => {
    // A whole number of 8px blocks either way, and 64 is past the point where
    // the level-1 pyramid and the 9-region SCD grid both have room to work.
    const renderSize: Size = [64, 64];
    const SHIFT_PIXELS = 4;

    interface FlowResult {
      vectors: Int32Array;
      validity: Uint32Array;
      sceneChanged: number;
      grid: Size;
    }

    async function runFlow(frames: number): Promise<FlowResult> {
      const stage = await checked(
        harness,
        () =>
          new OpticalFlowStage(harness.device, {
            renderSize,
            transferFunction: TransferFunction.LinearLdr,
            luminance: { min: 0, max: 1000 },
          }),
      );
      const previousColor = noiseTexture(harness, renderSize, 0);
      const currentColor = noiseTexture(harness, renderSize, SHIFT_PIXELS);

      for (let frame = 0; frame < frames; frame++) {
        stage.setFrameIndex(frame);
        await submitPass(harness, (pass) => stage.encode(pass, currentColor, previousColor));
      }

      const grid = opticalFlowGridSize(renderSize);
      const cells = grid[0] * grid[1];
      const result: FlowResult = {
        vectors: new Int32Array(await harness.readBuffer(stage.vectorField, cells * 8)),
        validity: new Uint32Array(await harness.readBuffer(stage.validity, cells * 4)),
        sceneChanged: new Uint32Array(await harness.readBuffer(stage.sceneChange, 4))[0],
        grid,
      };
      stage.destroy();

      return result;
    }

    it("reports a scene change until the detector has warmed up", async () => {
      const warm = await runFlow(SCD_WARMUP_FRAMES);
      expect(warm.sceneChanged).toBe(1);

      const settled = await runFlow(SCD_WARMUP_FRAMES + 3);
      expect(settled.sceneChanged).toBe(0);
    });

    // A border cell's search window runs off the frame, where the packed-luma
    // loader replicates the edge sample rather than finding the real content,
    // so it cannot match and reports invalid. The median filter then emits the
    // winning tap's verdict, which spreads that one cell further in.
    const BORDER_CELLS = 2;

    it("recovers a uniform translation as a valid motion vector", async () => {
      const { vectors, validity, grid } = await runFlow(SCD_WARMUP_FRAMES + 3);
      const [gridWidth, gridHeight] = grid;

      let checked = 0;
      for (let y = BORDER_CELLS; y < gridHeight - BORDER_CELLS; y++) {
        for (let x = BORDER_CELLS; x < gridWidth - BORDER_CELLS; x++) {
          const cell = y * gridWidth + x;
          expect([cell, vectors[cell * 2], vectors[cell * 2 + 1]]).toEqual([cell, -SHIFT_PIXELS, 0]);
          expect([cell, validity[cell]]).toEqual([cell, 1]);
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(0);
    });
  });
});
