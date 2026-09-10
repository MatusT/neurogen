import { describe, expect, it } from "vitest";
import {
  classicalBlendWeight,
  evaluateNetwork,
  mulberry32,
  sampleFeatures,
  targetBlendWeight,
  FEATURE_COUNT,
} from "./neural-reference.js";
import { packNeuralWeights, type NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";
import blendWeightMlp from "../src/wgsl/neural/weights/blend_weight_mlp.json";

// Neither the training run's seed nor the seed its own held-out report uses, so
// these samples are unseen by both.
const EVALUATION_SEED = 20260910;
const EVALUATION_SAMPLES = 4096;

const trainedNetwork: NeuralNetworkWeights = blendWeightMlp;

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
  // than to itself: relu(-1) clamps to 0, which is what makes the second
  // layer's output 2 rather than 1, and sigmoid(2) is the final answer.
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
  // that changed it would silently read the wrong weights on the GPU.
  it("has the shape the kernel assumes", () => {
    const layers = trainedNetwork.layers;

    expect(layers[0].inputs).toBe(FEATURE_COUNT);
    expect(layers.at(-1)?.outputs).toBe(1);
    for (const [index, layer] of layers.slice(1).entries()) {
      expect(layer.inputs).toBe(layers[index].outputs);
    }
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
