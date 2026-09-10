// CPU reference for the example blend-weight network, plus the synthetic
// objective it is trained against.
//
// Test-only scaffolding, deliberately outside src/ so it cannot reach the
// runtime bundle and is not exported from src/index.ts. Two things are checked
// against it: the trained weights (does the network actually fit the
// objective?) and the WGSL kernel (does the GPU port compute the same thing?).
//
// scripts/train-blend-mlp.ts imports the objective and the sampler from here —
// unusual for a script to reach into test/, but it means the function the
// weights are fitted to and the function the tests grade them against are one
// definition rather than two that can drift apart.
import type { NeuralLayerWeights, NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";

// [disocclusionMask.x, disocclusionMask.y, luma(preliminaryColor)] — see the
// feature contract in src/wgsl/neural/blend_weight_mlp.wgsl.
export const FEATURE_COUNT = 3;

type Activation = "relu" | "sigmoid";

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function activate(x: number, activation: Activation): number {
  if (activation === "relu") {
    return Math.max(x, 0);
  }

  return sigmoid(x);
}

export function applyLayer(
  layer: NeuralLayerWeights,
  input: readonly number[],
  activation: Activation,
): number[] {
  const output: number[] = [];
  for (let o = 0; o < layer.outputs; o++) {
    let sum = layer.biases[o];
    for (let i = 0; i < layer.inputs; i++) {
      sum += layer.weights[o * layer.inputs + i] * input[i];
    }

    output.push(activate(sum, activation));
  }

  return output;
}

// Hidden layers relu, output sigmoid — the arrangement the WGSL kernel mirrors.
// Relu because it is exact in both languages (`max(x, 0)` cannot disagree
// across implementations), leaving the output sigmoid as the only transcendental
// the GPU and this reference can round differently.
export function evaluateNetwork(network: NeuralNetworkWeights, features: readonly number[]): number {
  let activations = [...features];
  const lastLayer = network.layers.length - 1;

  for (const [index, layer] of network.layers.entries()) {
    activations = applyLayer(layer, activations, index === lastLayer ? "sigmoid" : "relu");
  }

  return activations[0];
}

// THE SYNTHETIC OBJECTIVE.
//
// The classical formula in preliminary_blend.wgsl hands a pixel to inpainting
// only when *neither* source frame sees it, and ignores the colour entirely.
// This objective keeps that verdict and adds one the classical formula cannot
// express: a pixel only one frame sees is warped from a single source, and when
// that pixel is bright the resulting smear reads as an artefact, where the same
// error in a dark region is invisible. So brightness partially disqualifies a
// single-source pixel too.
//
// Deliberately not the classical formula: a network that could be replaced by
// the pass it replaces would demonstrate nothing.
const LUMA_GATE_LOW = 0.2;
const LUMA_GATE_HIGH = 0.8;
const SINGLE_SOURCE_MAX_WEIGHT = 0.45;

function smoothstep(low: number, high: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - low) / (high - low)));
  return t * t * (3 - 2 * t);
}

export function targetBlendWeight(features: readonly number[]): number {
  const [visiblePrevious, visibleCurrent, luma] = features;

  const neitherSees = (1 - visiblePrevious) * (1 - visibleCurrent);
  const oneSees = visiblePrevious + visibleCurrent - 2 * visiblePrevious * visibleCurrent;
  const brightness = smoothstep(LUMA_GATE_LOW, LUMA_GATE_HIGH, luma);

  return Math.min(1, neitherSees + SINGLE_SOURCE_MAX_WEIGHT * oneSees * brightness);
}

// Mirrors `loadFeatures` in src/wgsl/neural/blend_weight_mlp.wgsl: the kernel
// assembles its own features on the GPU, so the agreement test needs this half
// of the port to compare against.
const REC709_LUMA = [0.2126, 0.7152, 0.0722];

export function blendWeightFeatures(mask: readonly number[], color: readonly number[]): number[] {
  const luma = REC709_LUMA.reduce((sum, coefficient, i) => sum + coefficient * color[i], 0);
  return [mask[0], mask[1], Math.min(1, Math.max(0, luma))];
}

// Mirrors `snapEndpoints` in src/wgsl/neural/blend_weight_mlp.wgsl, which pulls
// the sigmoid's unreachable ends onto exact 0 and 1 because the readers of
// `blendWeight` test the ends exactly.
//
// Applied around `evaluateNetwork` by the callers that compare against the
// kernel, never inside it: the training objective already targets exact 0 and 1
// at the endpoints, so folding this into the network's own forward pass would
// change what the optimiser fits and therefore the checked-in weights.
const ENDPOINT_EPSILON = 1e-3;

export function snapEndpoints(weight: number): number {
  return Math.min(1, Math.max(0, weight * (1 + 2 * ENDPOINT_EPSILON) - ENDPOINT_EPSILON));
}

// FI_EPSILON from src/wgsl/frameinterpolation/params.wgsl.
const FI_EPSILON = 1e-3;

// What preliminary_blend.wgsl writes for the same pixel: 1 where the mask says
// neither frame sees the surface, 0 everywhere else. The baseline that matters —
// it is what happens if the neural pass is not dispatched at all.
export function classicalBlendWeight(features: readonly number[]): number {
  const [visiblePrevious, visibleCurrent] = features;
  return Math.hypot(visiblePrevious, visibleCurrent) <= FI_EPSILON ? 1 : 0;
}

// The mask channels are binarised by disocclusion_mask.wgsl, so training sees
// them the way the shader will: one of the four combinations, uniformly. Luma
// is continuous, which is what makes the objective more than a 4-entry table.
export function sampleFeatures(random: () => number): number[] {
  return [Math.round(random()), Math.round(random()), random()];
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
