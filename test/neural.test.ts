import { beforeAll, describe, expect, it } from "vitest";
import {
  blendWeightFeatures,
  classicalBlendWeight,
  evaluateNetwork,
  mulberry32,
  sampleFeatures,
  snapEndpoints,
  targetBlendWeight,
  FEATURE_COUNT,
} from "./neural-reference.js";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import { packNeuralWeights, type NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";
import blendWeightMlp from "../src/wgsl/neural/weights/blend_weight_mlp.json";
import blendWeightMlpWgsl from "../src/wgsl/generated/neural/blend_weight_mlp.wgsl.js";

// Neither the training run's seed nor the seed its own held-out report uses, so
// these samples are unseen by both.
const EVALUATION_SEED = 20260910;
const EVALUATION_SAMPLES = 4096;

const trainedNetwork: NeuralNetworkWeights = blendWeightMlp;

// MLP_HIDDEN in src/wgsl/neural/blend_weight_mlp.wgsl.
const KERNEL_HIDDEN_CHANNELS = 8;

function meanSquaredError(predict: (features: number[]) => number): number {
  const random = mulberry32(EVALUATION_SEED);
  let total = 0;
  for (let i = 0; i < EVALUATION_SAMPLES; i++) {
    const features = sampleFeatures(random);
    total += (predict(features) - targetBlendWeight(features)) ** 2;
  }

  return total / EVALUATION_SAMPLES;
}

describe("neural reference", () => {
  // Hand-computed so the reference is pinned to an arithmetic result rather
  // than to itself: the second hidden unit is -1 before its relu, so clamping
  // it makes the output layer 2 rather than -1, and sigmoid(2) is the answer.
  it("applies relu to hidden layers and sigmoid to the output", () => {
    const network: NeuralNetworkWeights = {
      name: "hand",
      layers: [
        { inputs: 2, outputs: 2, weights: [1, 0, 0, 1], biases: [0, -2] },
        { inputs: 2, outputs: 1, weights: [2, 3], biases: [0] },
      ],
    };

    expect(evaluateNetwork(network, [1, 1])).toBeCloseTo(1 / (1 + Math.exp(-2)), 12);
  });

  it("packs each layer as a row-major matrix followed by its biases", () => {
    const network: NeuralNetworkWeights = {
      name: "hand",
      layers: [
        { inputs: 2, outputs: 2, weights: [1, 2, 3, 4], biases: [5, 6] },
        { inputs: 2, outputs: 1, weights: [7, 8], biases: [9] },
      ],
    };

    expect([...packNeuralWeights(network)]).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("rejects a layer whose weights do not match its declared shape", () => {
    const network: NeuralNetworkWeights = {
      name: "hand",
      layers: [{ inputs: 2, outputs: 2, weights: [1, 2, 3], biases: [4, 5] }],
    };

    expect(() => packNeuralWeights(network)).toThrow(/expected 4/);
  });
});

describe("trained blend weight network", () => {
  // The WGSL kernel derives its layer offsets from this shape, so a retrain
  // that changed it would silently read the wrong weights on the GPU. The
  // widths are pinned and not merely checked for self-consistency: the kernel's
  // MLP_HIDDEN is 8, and a chain that is consistent at some other width would
  // pass a consistency check while every offset after layer 0 was wrong.
  it("has the shape the kernel assumes", () => {
    const layers = trainedNetwork.layers;

    expect(layers.map((layer) => [layer.inputs, layer.outputs])).toEqual([
      [FEATURE_COUNT, KERNEL_HIDDEN_CHANNELS],
      [KERNEL_HIDDEN_CHANNELS, KERNEL_HIDDEN_CHANNELS],
      [KERNEL_HIDDEN_CHANNELS, 1],
    ]);
  });

  it("stays inside the blend weight contract's 0..1 range", () => {
    const random = mulberry32(EVALUATION_SEED);
    for (let i = 0; i < EVALUATION_SAMPLES; i++) {
      const weight = evaluateNetwork(trainedNetwork, sampleFeatures(random));
      expect(weight).toBeGreaterThanOrEqual(0);
      expect(weight).toBeLessThanOrEqual(1);
    }
  });

  // The point of the whole task: dispatching this network has to do something
  // the pass it replaces cannot. A bright pixel only the previous frame sees is
  // where the two disagree most.
  it("disagrees with the classical formula where the objective says it should", () => {
    const brightSingleSource = [1, 0, 0.9];
    const darkSingleSource = [0, 1, 0.05];

    expect(classicalBlendWeight(brightSingleSource)).toBe(0);
    expect(evaluateNetwork(trainedNetwork, brightSingleSource)).toBeGreaterThan(0.35);

    // And is not simply "always disagree": the same pixel when dark is one the
    // classical formula already handles, and the network leaves it alone.
    expect(evaluateNetwork(trainedNetwork, darkSingleSource)).toBeLessThan(0.05);
  });

  // The endpoint rescale only rescues an output already within its epsilon of
  // 1, so the margin the weights hold at the ends is a property, not a
  // coincidence: a retrain that left a hole at 0.998 would silently restore the
  // inpainting break. Exhaustive over the runtime input domain — the mask is
  // binarised, so a hole is exactly (0, 0) and only the luma varies.
  it("snaps every hole to exactly 1 across the whole luma range", () => {
    const lumaSteps = 1000;
    for (let step = 0; step <= lumaSteps; step++) {
      const weight = snapEndpoints(evaluateNetwork(trainedNetwork, [0, 0, step / lumaSteps]));
      expect(1 - weight).toBe(0);
    }
  });

  it("beats the classical formula and both constant baselines on unseen samples", () => {
    const trained = meanSquaredError((features) => evaluateNetwork(trainedNetwork, features));
    const classical = meanSquaredError(classicalBlendWeight);
    const zero = meanSquaredError(() => 0);

    const random = mulberry32(EVALUATION_SEED);
    const targets = Array.from({ length: EVALUATION_SAMPLES }, () => targetBlendWeight(sampleFeatures(random)));
    const targetMean = targets.reduce((sum, target) => sum + target, 0) / targets.length;
    const constantMean = meanSquaredError(() => targetMean);

    expect(trained).toBeLessThan(1e-3);
    expect(trained * 4).toBeLessThan(Math.min(classical, zero, constantMean));
  });
});

// Deliberately not a multiple of the 8x8 workgroup, so the last workgroup runs
// invocations past the edge of the image.
const RENDER_SIZE: [number, number] = [13, 5];
const PIXEL_COUNT = RENDER_SIZE[0] * RENDER_SIZE[1];
// Floats past the image, pre-filled with a value the sigmoid cannot produce.
const GUARD_FLOATS = 8;
const GUARD_VALUE = -1;
const AGREEMENT_TOLERANCE = 1e-3;
// FrameInterpolationParams padded, per test/frameinterpolation.test.ts.
const FRAME_INTERPOLATION_PARAMS_BYTES = 48;

// Cycles the four combinations of the binarised mask.
function fixtureMask(index: number): number[] {
  return [index % 2, Math.floor(index / 2) % 2];
}

// Luma spans 0..1.075, so pixel 0 is the darkest possible hole -- where the
// network is least certain -- and the brightest pixels drive the kernel's
// saturate. The channels are kept unequal so a wrong luma coefficient cannot
// cancel out.
function fixtureColor(index: number): number[] {
  const level = ((index * 7) % 17) / 12;
  return [level, level * 0.8, level * 0.3, 1];
}

describe("blend weight kernel on the GPU", () => {
  let harness: GpuHarness;
  let gpuWeights: Float32Array;

  beforeAll(async () => {
    harness = await createGpuHarness();

    const mask = new Float32Array(PIXEL_COUNT * 2);
    const color = new Float32Array(PIXEL_COUNT * 4);
    for (let i = 0; i < PIXEL_COUNT; i++) {
      mask.set(fixtureMask(i), i * 2);
      color.set(fixtureColor(i), i * 4);
    }

    const output = new Float32Array(PIXEL_COUNT + GUARD_FLOATS).fill(GUARD_VALUE);
    const outputBuffer = harness.createStorageBuffer(output);

    // Sized as the frame-interpolation params buffer rather than as this pass's
    // own 8-byte struct, because that is what orchestration will bind: the
    // kernel's uniform is a byte-compatible prefix of it, and this dispatch is
    // what makes that claim testable rather than asserted.
    const params = new Int32Array(FRAME_INTERPOLATION_PARAMS_BYTES / 4);
    params.set(RENDER_SIZE);

    await harness.dispatch(
      harness.createComputePipeline(blendWeightMlpWgsl),
      [
        { buffer: harness.createUniformBuffer(params) },
        { buffer: harness.createStorageBuffer(packNeuralWeights(trainedNetwork)) },
        { buffer: harness.createStorageBuffer(mask) },
        { buffer: harness.createStorageBuffer(color) },
        { buffer: outputBuffer },
      ],
      [Math.ceil(RENDER_SIZE[0] / 8), Math.ceil(RENDER_SIZE[1] / 8)],
    );

    gpuWeights = new Float32Array(await harness.readBuffer(outputBuffer, output.byteLength));
  });

  function referenceWeight(index: number): number {
    const features = blendWeightFeatures(fixtureMask(index), fixtureColor(index));
    return snapEndpoints(evaluateNetwork(trainedNetwork, features));
  }

  // The test that proves the port, rather than proving it compiles: the same
  // features through the CPU reference and through the kernel.
  it("agrees with the CPU reference", () => {
    let worstDifference = 0;
    for (let i = 0; i < PIXEL_COUNT; i++) {
      worstDifference = Math.max(worstDifference, Math.abs(gpuWeights[i] - referenceWeight(i)));
    }

    expect(worstDifference).toBeLessThan(AGREEMENT_TOLERANCE);
  });

  // The inpainting pyramid turns this weight into coverage, `1 - weight`, and
  // final_blend.wgsl gates that coverage with `f32(sample.w > 0.0)` — a binary
  // test, so a hole left at 0.9997 is as covered as a pixel that was never a
  // hole at all. Its own meaningless colour then survives every reduction and
  // wins at mip 0, and inpainting degenerates to identity. Nothing short of
  // exactly 1.0 makes the hole drop out.
  it("writes exactly 1.0 where neither frame sees the surface", () => {
    const holeWeights = [];
    for (let i = 0; i < PIXEL_COUNT; i++) {
      const [visiblePrevious, visibleCurrent] = fixtureMask(i);
      if (visiblePrevious === 0 && visibleCurrent === 0) {
        holeWeights.push(gpuWeights[i]);
      }
    }

    expect(holeWeights.length).toBeGreaterThan(0);
    expect(holeWeights.every((weight) => 1 - weight === 0)).toBe(true);
  });

  it("leaves the floats past the image untouched", () => {
    expect([...gpuWeights.slice(PIXEL_COUNT)]).toEqual(new Array(GUARD_FLOATS).fill(GUARD_VALUE));
  });

  // What the extension point is for: on the GPU, with the trained weights, the
  // buffer downstream reads is not the one the classical pass would have left.
  it("writes a weight the classical formula would not have", () => {
    let worstDifference = 0;
    for (let i = 0; i < PIXEL_COUNT; i++) {
      const classical = classicalBlendWeight(blendWeightFeatures(fixtureMask(i), fixtureColor(i)));
      worstDifference = Math.max(worstDifference, Math.abs(gpuWeights[i] - classical));
    }

    expect(worstDifference).toBeGreaterThan(0.3);
  });
});
