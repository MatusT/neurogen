import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import { FrameGenerator } from "../src/orchestration/FrameGenerator.js";
import { OpticalFlowStage } from "../src/orchestration/opticalFlow.js";
import blendWeightAsset from "../src/wgsl/neural/weights/blend_weight_mlp.json";
import type { NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";
import {
  History,
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
async function checked<T>(harness: GpuHarness, body: () => T): Promise<Awaited<T>> {
  const { device } = harness;
  device.pushErrorScope("validation");
  try {
    // Awaited inside the try, so an async body's work happens before the scope
    // is popped. Returning the promise unawaited would close the scope at the
    // return statement and capture nothing.
    return await body();
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
      reset: History.Discard,
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

describe("frame generator", () => {
  let harness: GpuHarness;

  beforeAll(async () => {
    harness = await createGpuHarness();
  });

  const renderSize: Size = [64, 64];
  const [WIDTH, HEIGHT] = renderSize;
  const PIXELS = WIDTH * HEIGHT;

  // Past both warmups: scene change detection forces a cut for six frames and
  // resets the frame counter each time, and preliminary_blend trusts the game
  // vectors unconditionally for ten frames after that. Only beyond both is the
  // whole graph — including the game-versus-optical-flow scoring — in play.
  const SETTLED_FRAMES = 20;

  const PROJECTION = { nearPlane: 0.1, farPlane: 100, verticalFovRadians: Math.PI / 3 };

  function texture(
    valueAt: (x: number, y: number) => readonly [number, number, number, number],
  ): GPUTextureView {
    const texels = new Float32Array(PIXELS * 4);
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        texels.set(valueAt(x, y), (y * WIDTH + x) * 4);
      }
    }

    return harness.createTexture(WIDTH, HEIGHT, texels).createView();
  }

  // A horizontal ramp in red and blue, so a pixel warped half a step really is
  // the midpoint of the two sources' values there and the assertion below can
  // be exact rather than "plausible". Green carries per-pixel noise, because a
  // pure ramp gives the optical flow's block matching nothing to lock onto.
  function translatedFrame(shift: number): GPUTextureView {
    return texture((x, y) => {
      const ramp = (x - shift) / WIDTH;
      return [ramp, noise(x - shift, y), ramp, 1];
    });
  }

  const SHIFT_PIXELS = 2;

  function translationInputs() {
    return {
      previousColor: translatedFrame(0),
      currentColor: translatedFrame(SHIFT_PIXELS),
      depth: texture(() => [0.5, 0, 0, 1]),
      // Whole pixels from a current-frame pixel to where it was previously.
      motionVectors: texture(() => [-SHIFT_PIXELS, 0, 0, 1]),
    };
  }

  // The handed-back texture is rgba16float, so a readback has to decode halves.
  // Only the exponent range the fixture uses is covered.
  function decodeHalf(bits: number): number {
    const sign = bits >>> 15 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) {
      return sign * mantissa * 2 ** -24;
    }

    return sign * (mantissa + 1024) * 2 ** (exponent - 25);
  }

  // copyTextureToBuffer needs a 256-byte row pitch, which is exactly the
  // constraint the blit pass exists to sidestep on the way in — so the readback
  // pads and unpads rather than assuming a convenient width.
  const COPY_ROW_ALIGNMENT = 256;

  async function readTexture(texture: GPUTexture): Promise<Float32Array> {
    const { width, height } = texture;
    const paddedRowBytes = Math.ceil((width * 8) / COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    const readback = harness.device.createBuffer({
      size: paddedRowBytes * height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = harness.device.createCommandEncoder();
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readback, bytesPerRow: paddedRowBytes },
      { width, height },
    );
    harness.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const padded = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();

    const texels = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      const row = new Uint16Array(padded, y * paddedRowBytes, width * 4);
      for (let i = 0; i < row.length; i++) {
        texels[y * width * 4 + i] = decodeHalf(row[i]);
      }
    }

    return texels;
  }

  // An asset may arrive from a fetch either side of configure(), so both orders
  // have to reach the same pass.
  enum InstallTime {
    BeforeConfigure,
    AfterConfigure,
  }

  async function generate(
    inputs: ReturnType<typeof translationInputs>,
    neuralWeights?: NeuralNetworkWeights,
    installTime: InstallTime = InstallTime.BeforeConfigure,
  ): Promise<{ color: Float32Array; texture: Float32Array; weight: Float32Array }> {
    const generator = new FrameGenerator({ device: harness.device });
    const install = async (at: InstallTime) => {
      if (neuralWeights && installTime === at) {
        await checked(harness, () => generator.installNeuralBlendWeight(neuralWeights));
      }
    };

    await install(InstallTime.BeforeConfigure);
    await checked(harness, () => generator.configure({ ...PROJECTION, renderWidth: WIDTH, renderHeight: HEIGHT }));
    await install(InstallTime.AfterConfigure);

    let texture!: GPUTexture;
    for (let frame = 0; frame < SETTLED_FRAMES; frame++) {
      await checked(harness, () => {
        generator.prepare(inputs);
        texture = generator.dispatch();
      });
    }

    const result = {
      color: new Float32Array(await harness.readBuffer(generator.interpolatedColor, PIXELS * 16)),
      texture: await readTexture(texture),
      weight: new Float32Array(await harness.readBuffer(generator.blendWeight, PIXELS * 4)),
    };
    generator.destroy();

    return result;
  }

  it("lands a uniform translation exactly halfway between the two frames", async () => {
    const { color, texture } = await generate(translationInputs());

    let worstRamp = 0;
    let worstTexel = 0;
    let worstNoise = 0;
    let crossfadeDistance = 0;
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        const index = (y * WIDTH + x) * 4;
        // Half of a two-pixel step: the midpoint of the two sources.
        const warpedRamp = (x - SHIFT_PIXELS / 2) / WIDTH;
        worstRamp = Math.max(worstRamp, Math.abs(color[index] - warpedRamp));
        worstTexel = Math.max(worstTexel, Math.abs(texture[index] - warpedRamp));

        // The ramp alone cannot tell a warp from a crossfade: averaging two
        // linear ramps offset by +a and -a gives the same value for every a,
        // so a pipeline that did no warping at all would satisfy the two
        // assertions above. The noise channel is what discriminates.
        const warpedNoise = noise(x - SHIFT_PIXELS / 2, y);
        const crossfade = (noise(x, y) + noise(x - SHIFT_PIXELS, y)) / 2;
        worstNoise = Math.max(worstNoise, Math.abs(color[index + 1] - warpedNoise));
        crossfadeDistance += Math.abs(warpedNoise - crossfade);

        expect([x, y, color[index + 3]]).toEqual([x, y, 1]);
      }
    }

    expect(worstRamp).toBeLessThan(1e-3);
    expect(worstNoise).toBeLessThan(1e-3);
    expect(color.every(Number.isFinite)).toBe(true);
    // The texture is the same frame at half the mantissa, so it only has to
    // agree to f16 precision over the 0..1 range this fixture spans.
    expect(worstTexel).toBeLessThan(1e-3);
    // ...and the crossfade the noise assertion rules out is not a near miss.
    expect(crossfadeDistance / PIXELS).toBeGreaterThan(0.1);
  });

  // The optical flow field only ever corroborates the game motion vectors in
  // the blend's scoring, so a scene where the game vectors are already correct
  // cannot tell a working handoff from a broken one. Zeroing them is what makes
  // the optical flow the only thing carrying the motion.
  it("carries the motion through the optical flow when the game vectors are blank", async () => {
    // A whole 8x8 block, so the flow grid gets a clean unambiguous vector.
    const FLOW_SHIFT = 8;
    const noiseFrame = (shift: number) =>
      texture((x, y) => {
        const value = noise(x - shift, y);
        return [value, value, value, 1];
      });

    const { color } = await generate({
      previousColor: noiseFrame(0),
      currentColor: noiseFrame(FLOW_SHIFT),
      depth: texture(() => [0.5, 0, 0, 1]),
      motionVectors: texture(() => [0, 0, 0, 1]),
    });

    // Measured against the frame a pipeline with no motion at all would emit.
    // Compared as a mean rather than a worst case: the optical flow moves the
    // blend off the crossfade over the frame, it does not reproduce the warp.
    let departure = 0;
    let samples = 0;
    for (let y = 8; y < HEIGHT - 8; y++) {
      for (let x = 16; x < WIDTH - 16; x++) {
        const crossfade = (noise(x, y) + noise(x - FLOW_SHIFT, y)) / 2;
        departure += Math.abs(color[(y * WIDTH + x) * 4] - crossfade);
        samples++;
      }
    }

    expect(color.every(Number.isFinite)).toBe(true);
    expect(departure / samples).toBeGreaterThan(0.08);
  });

  it("hands back a texture at a width no buffer copy could have filled", async () => {
    // 67 * 16 bytes per row is not a multiple of 256, so copyBufferToTexture
    // could not have written this and the blit pass is doing real work.
    const [width, height] = [67, 37];
    const generator = new FrameGenerator({ device: harness.device });
    await checked(harness, () =>
      generator.configure({ ...PROJECTION, renderWidth: width, renderHeight: height }),
    );

    const oddTexture = (valueAt: (x: number, y: number) => readonly [number, number, number, number]) => {
      const texels = new Float32Array(width * height * 4);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          texels.set(valueAt(x, y), (y * width + x) * 4);
        }
      }
      return harness.createTexture(width, height, texels).createView();
    };
    // Varying in both axes, and not symmetric between them, so a transposed or
    // otherwise permuted index in the blit cannot reproduce it. A flat colour
    // would pass any permutation.
    const scene = oddTexture((x, y) => [x / width, y / height, ((x * y) % 17) / 17, 1]);
    const inputs = {
      currentColor: scene,
      previousColor: scene,
      depth: oddTexture(() => [0.5, 0, 0, 1]),
      motionVectors: oddTexture(() => [0, 0, 0, 1]),
    };

    let texture!: GPUTexture;
    for (let frame = 0; frame < SETTLED_FRAMES; frame++) {
      await checked(harness, () => {
        generator.prepare(inputs);
        texture = generator.dispatch();
      });
    }
    const texels = await readTexture(texture);
    const buffer = new Float32Array(
      await harness.readBuffer(generator.interpolatedColor, width * height * 16),
    );
    generator.destroy();

    expect([texture.width, texture.height]).toEqual([width, height]);

    // The blit is the only thing between these two, so comparing them texel for
    // texel isolates it from the rest of the pipeline.
    let worst = 0;
    let spread = 0;
    for (let i = 0; i < width * height * 4; i++) {
      worst = Math.max(worst, Math.abs(texels[i] - buffer[i]));
      spread = Math.max(spread, Math.abs(buffer[i] - buffer[0]));
    }

    // f16 has ~3 decimal digits over this range.
    expect(worst).toBeLessThan(1e-3);
    // The comparison above only discriminates while the content varies.
    expect(spread).toBeGreaterThan(0.5);
  });

  it("dispatches the neural writer in place of the classical one", async () => {
    const inputs = translationInputs();
    const classical = await generate(inputs);
    const neural = await generate(inputs, blendWeightAsset);

    // The classical formula writes exactly 0 wherever both frames see the
    // surface. The network is a sigmoid rescaled onto the endpoints, so it can
    // only reach exactly 0 below its epsilon — a strictly positive weight where
    // the classical writer put a hard zero is the replacement having run.
    expect(classical.weight.every((weight) => weight === 0)).toBe(true);
    expect(Math.max(...neural.weight)).toBeGreaterThan(0);
    expect(neural.color.every(Number.isFinite)).toBe(true);

    const installedLate = await generate(inputs, blendWeightAsset, InstallTime.AfterConfigure);
    expect([...installedLate.weight]).toEqual([...neural.weight]);
  });

  it("rejects an asset the kernel's hard-coded layer offsets do not fit", () => {
    const generator = new FrameGenerator({ device: harness.device });
    const wrongShape = {
      name: "wrong",
      layers: [{ inputs: 3, outputs: 4, weights: new Array(12).fill(0), biases: new Array(4).fill(0) }],
    };

    expect(() => generator.installNeuralBlendWeight(wrongShape)).toThrow(/layer shape/);
  });

  describe("call order", () => {
    it("refuses to run before configure()", () => {
      const generator = new FrameGenerator({ device: harness.device });

      expect(() => generator.dispatch()).toThrow(/configure\(\)/);
    });

    it("refuses to dispatch a frame that was never prepared", async () => {
      const generator = new FrameGenerator({ device: harness.device });
      await checked(harness, () => generator.configure({ ...PROJECTION, renderWidth: WIDTH, renderHeight: HEIGHT }));

      expect(() => generator.dispatch()).toThrow(/before prepare/);
      generator.destroy();
    });

    it("refuses to prepare a second frame over one still in flight", async () => {
      const generator = new FrameGenerator({ device: harness.device });
      const inputs = translationInputs();
      await checked(harness, () => generator.configure({ ...PROJECTION, renderWidth: WIDTH, renderHeight: HEIGHT }));
      await checked(harness, () => generator.prepare(inputs));

      expect(() => generator.prepare(inputs)).toThrow(/without an intervening dispatch/);
      generator.destroy();
    });
  });

  describe("configure", () => {
    it("rejects a resolution whose colour buffer exceeds the device limit", () => {
      const generator = new FrameGenerator({ device: harness.device });
      const limit = harness.device.limits.maxStorageBufferBindingSize;
      // 16 bytes per texel, so this is one texel past what the limit allows.
      const height = Math.floor(limit / 16 / 4096) + 1;

      expect(() => generator.configure({ ...PROJECTION, renderWidth: 4096, renderHeight: height }))
        .toThrow(/maxStorageBufferBindingSize/);
    });

    it("rejects planes the view-space depth conversion cannot use", () => {
      const generator = new FrameGenerator({ device: harness.device });

      expect(() =>
        generator.configure({
          renderWidth: WIDTH,
          renderHeight: HEIGHT,
          nearPlane: 100,
          farPlane: 0.1,
          verticalFovRadians: 1,
        }),
      ).toThrow(/0 < near < far/);
    });
  });
});
