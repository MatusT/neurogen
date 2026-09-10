// Trains the example blend-weight network and rewrites its weight asset.
//
//   npm run train:neural
//
// Offline: not part of `npm run build`, and the asset it writes is checked in.
// Run under vite-node (see package.json) so it can import the TypeScript
// reference in test/, which owns the objective being fitted here.
//
// Plain Adam over a hand-derived backward pass — 113 parameters do not justify
// an ML framework dependency. Seeded throughout, so re-running reproduces the
// checked-in asset byte for byte; that reproducibility is how "these weights
// came from an optimiser" can be verified rather than taken on trust.
import { writeFileSync } from "node:fs";
import {
  FEATURE_COUNT,
  applyLayer,
  classicalBlendWeight,
  evaluateNetwork,
  mulberry32,
  sampleFeatures,
  targetBlendWeight,
} from "../test/neural-reference.js";
import type { NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";

const ASSET_PATH = new URL("../src/wgsl/neural/weights/blend_weight_mlp.json", import.meta.url);

const NETWORK_NAME = "blend_weight_mlp";
const HIDDEN_CHANNELS = 8;
const LAYER_SHAPES = [
  [FEATURE_COUNT, HIDDEN_CHANNELS],
  [HIDDEN_CHANNELS, HIDDEN_CHANNELS],
  [HIDDEN_CHANNELS, 1],
] as const;

const TRAINING_SEED = 1;
const HELD_OUT_SEED = 7331;
const STEPS = 8000;
const BATCH_SIZE = 128;
const HELD_OUT_SAMPLES = 4096;
const LEARNING_RATE = 0.02;
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;

// The trained network has to beat every baseline by this factor to be worth
// checking in. A network that merely ties the classical formula would be an
// expensive way to change nothing.
const REQUIRED_IMPROVEMENT = 4;

interface Layer {
  inputs: number;
  outputs: number;
  weights: number[];
  biases: number[];
  weightMoment1: number[];
  weightMoment2: number[];
  biasMoment1: number[];
  biasMoment2: number[];
}

function zeros(length: number): number[] {
  return new Array<number>(length).fill(0);
}

// Box-Muller, so the initialisation is drawn from the same seeded stream as
// everything else rather than from an unseeded Math.random.
function gaussian(random: () => number): number {
  const u = Math.max(random(), Number.MIN_VALUE);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

// He initialisation: relu zeroes half its inputs, so the variance that survives
// a layer is halved and the weights are scaled by sqrt(2/fanIn) to compensate.
function initLayers(random: () => number): Layer[] {
  return LAYER_SHAPES.map(([inputs, outputs]) => {
    const scale = Math.sqrt(2 / inputs);
    const weights = zeros(inputs * outputs);
    for (let i = 0; i < weights.length; i++) {
      weights[i] = gaussian(random) * scale;
    }

    return {
      inputs,
      outputs,
      weights,
      biases: zeros(outputs),
      weightMoment1: zeros(weights.length),
      weightMoment2: zeros(weights.length),
      biasMoment1: zeros(outputs),
      biasMoment2: zeros(outputs),
    };
  });
}

function toAsset(layers: readonly Layer[]): NeuralNetworkWeights {
  return {
    name: NETWORK_NAME,
    // Rounded to f32 here, not at pack time: the asset then holds exactly the
    // values the GPU will use, so the CPU reference and the kernel differ only
    // in arithmetic precision and not in their inputs.
    layers: layers.map((layer) => ({
      inputs: layer.inputs,
      outputs: layer.outputs,
      weights: layer.weights.map(Math.fround),
      biases: layer.biases.map(Math.fround),
    })),
  };
}

// Every layer's post-activation output, with the input features as element 0 —
// the backward pass needs each layer's input, and this is it, shifted by one.
function forwardActivations(layers: readonly Layer[], features: readonly number[]): number[][] {
  const activations = [[...features]];
  const lastLayer = layers.length - 1;

  for (const [index, layer] of layers.entries()) {
    activations.push(applyLayer(layer, activations[index], index === lastLayer ? "sigmoid" : "relu"));
  }

  return activations;
}

// Accumulates dLoss/dWeight for one sample into `weightGradients`/`biasGradients`.
//
// Loss is mean squared error and the output is a sigmoid, so the gradient
// arriving at the output layer's pre-activation is
//
//   d/dz (y - t)^2 = 2 (y - t) * y (1 - y)      [sigmoid' = y (1 - y)]
//
// and relu's derivative below is the step function, read off the post-activation
// value since relu is zero exactly where its input was negative.
function accumulateGradients(
  layers: readonly Layer[],
  features: readonly number[],
  target: number,
  weightGradients: number[][],
  biasGradients: number[][],
): void {
  const activations = forwardActivations(layers, features);
  const prediction = activations[activations.length - 1][0];
  const error = prediction - target;

  let delta = [2 * error * prediction * (1 - prediction)];

  for (let index = layers.length - 1; index >= 0; index--) {
    const layer = layers[index];
    const input = activations[index];

    const inputDelta = new Array<number>(layer.inputs).fill(0);
    for (let o = 0; o < layer.outputs; o++) {
      biasGradients[index][o] += delta[o];
      for (let i = 0; i < layer.inputs; i++) {
        weightGradients[index][o * layer.inputs + i] += delta[o] * input[i];
        inputDelta[i] += delta[o] * layer.weights[o * layer.inputs + i];
      }
    }

    // The layer below is a relu (the input features have no gradient, and that
    // iteration exits the loop), so gate by whether it fired.
    for (let i = 0; i < layer.inputs; i++) {
      inputDelta[i] *= Number(input[i] > 0);
    }

    delta = inputDelta;
  }
}

function applyAdam(
  layers: readonly Layer[],
  weightGradients: number[][],
  biasGradients: number[][],
  step: number,
): void {
  const bias1 = 1 - Math.pow(ADAM_BETA1, step);
  const bias2 = 1 - Math.pow(ADAM_BETA2, step);

  for (const [index, layer] of layers.entries()) {
    adamStep(layer.weights, weightGradients[index], layer.weightMoment1, layer.weightMoment2, bias1, bias2);
    adamStep(layer.biases, biasGradients[index], layer.biasMoment1, layer.biasMoment2, bias1, bias2);
  }
}

function adamStep(
  values: number[],
  gradients: number[],
  moment1: number[],
  moment2: number[],
  bias1: number,
  bias2: number,
): void {
  for (let i = 0; i < values.length; i++) {
    const gradient = gradients[i] / BATCH_SIZE;
    moment1[i] = ADAM_BETA1 * moment1[i] + (1 - ADAM_BETA1) * gradient;
    moment2[i] = ADAM_BETA2 * moment2[i] + (1 - ADAM_BETA2) * gradient * gradient;

    const corrected1 = moment1[i] / bias1;
    const corrected2 = moment2[i] / bias2;
    values[i] -= (LEARNING_RATE * corrected1) / (Math.sqrt(corrected2) + ADAM_EPSILON);
  }
}

function train(): Layer[] {
  const random = mulberry32(TRAINING_SEED);
  const layers = initLayers(random);

  for (let step = 1; step <= STEPS; step++) {
    const weightGradients = layers.map((layer) => zeros(layer.weights.length));
    const biasGradients = layers.map((layer) => zeros(layer.biases.length));

    for (let sample = 0; sample < BATCH_SIZE; sample++) {
      const features = sampleFeatures(random);
      accumulateGradients(layers, features, targetBlendWeight(features), weightGradients, biasGradients);
    }

    applyAdam(layers, weightGradients, biasGradients, step);
  }

  return layers;
}

// Held-out: a stream the optimiser never drew from, so a network that memorised
// its batches has nothing to fall back on here.
function heldOutScores(network: NeuralNetworkWeights): Record<string, number> {
  const random = mulberry32(HELD_OUT_SEED);
  const samples = Array.from({ length: HELD_OUT_SAMPLES }, () => sampleFeatures(random));
  const targets = samples.map(targetBlendWeight);
  const targetMean = targets.reduce((sum, t) => sum + t, 0) / targets.length;

  const meanSquaredError = (predict: (features: number[]) => number) =>
    samples.reduce((sum, features, i) => sum + (predict(features) - targets[i]) ** 2, 0) / samples.length;

  return {
    trained: meanSquaredError((features) => evaluateNetwork(network, features)),
    classical: meanSquaredError(classicalBlendWeight),
    targetMean: meanSquaredError(() => targetMean),
    zero: meanSquaredError(() => 0),
  };
}

const network = toAsset(train());
const scores = heldOutScores(network);

console.log(`held-out mean squared error over ${HELD_OUT_SAMPLES} unseen samples:`);
for (const [name, score] of Object.entries(scores)) {
  console.log(`  ${name.padEnd(10)} ${score.toExponential(3)}`);
}

const baselines = Object.entries(scores).filter(([name]) => name !== "trained");
const unbeaten = baselines.filter(([, score]) => score < scores.trained * REQUIRED_IMPROVEMENT);
if (unbeaten.length > 0) {
  console.error(
    `refusing to write ${NETWORK_NAME}: did not beat ${unbeaten.map(([name]) => name).join(", ")} ` +
      `by ${REQUIRED_IMPROVEMENT}x`,
  );
  process.exit(1);
}

// A pixel one frame sees, bright: the classical formula writes 0 and the
// objective asks for roughly SINGLE_SOURCE_MAX_WEIGHT. Printed because "the
// weights are not a passthrough" is the claim this script exists to support.
const singleSourceBright = [1, 0, 0.9];
console.log(
  `\nnon-identity check at [mask 1,0 | luma 0.9]: ` +
    `network ${evaluateNetwork(network, singleSourceBright).toFixed(4)}, ` +
    `classical ${classicalBlendWeight(singleSourceBright).toFixed(4)}, ` +
    `objective ${targetBlendWeight(singleSourceBright).toFixed(4)}`,
);

const asset = {
  ...network,
  inputs: ["disocclusionMask.x", "disocclusionMask.y", "luma(preliminaryColor)"],
  hiddenActivation: "relu",
  outputActivation: "sigmoid",
  training: {
    script: "scripts/train-blend-mlp.ts",
    seed: TRAINING_SEED,
    steps: STEPS,
    batchSize: BATCH_SIZE,
    learningRate: LEARNING_RATE,
    heldOutSeed: HELD_OUT_SEED,
    heldOutMeanSquaredError: scores,
  },
};

writeFileSync(ASSET_PATH, `${JSON.stringify(asset, null, 2)}\n`);
console.log(`\nwrote ${ASSET_PATH.pathname}`);
