// Weight asset format and GPU weight-buffer packing for the neural passes in
// this directory. Original work, not a port of AMD code.
//
// ASSET FORMAT (src/wgsl/neural/weights/*.json). A JSON object with a `layers`
// array, each layer holding a row-major `weights` matrix and a `biases` vector.
// JSON rather than a binary blob because these networks are tiny — the whole
// example is ~3KB — and a readable asset can be diffed when it is retrained.
// The file also carries a `training` block recording how it was produced; no
// code reads it.
//
// WEIGHT-BUFFER CONVENTION. `packNeuralWeights` flattens the asset into the
// single `array<f32>` storage buffer that the WGSL primitives read. Layers are
// concatenated in evaluation order, and each layer occupies
//
//   [offset,                     offset + outputs * inputs)  row-major matrix,
//                                                            row = one output
//                                                            channel's weights
//   [offset + outputs * inputs,  offset + outputs * inputs + outputs)  biases
//
// so a layer's span is `outputs * (inputs + 1)` floats and its offset is the
// sum of the spans before it. That is the whole convention: a network is
// pluggable if its WGSL declares layer offsets computed this way from its own
// shape, and nothing else about the buffer is network-specific.
//
// Not exported from src/index.ts. Task 6 owns the public "install a custom
// neural op" surface and decides what of this to re-export.

export interface NeuralLayerWeights {
  inputs: number;
  outputs: number;
  // Row-major: `weights[o * inputs + i]` connects input i to output o.
  weights: number[];
  biases: number[];
}

export interface NeuralNetworkWeights {
  name: string;
  layers: NeuralLayerWeights[];
}

export function neuralWeightCount(network: NeuralNetworkWeights): number {
  return network.layers.reduce((total, layer) => total + layer.outputs * (layer.inputs + 1), 0);
}

// Throws on a malformed asset rather than packing a short buffer: the shader
// reads fixed offsets derived from the shape, so a layer of the wrong length
// silently shifts every later layer's weights instead of failing.
export function packNeuralWeights(network: NeuralNetworkWeights): Float32Array {
  const packed = new Float32Array(neuralWeightCount(network));

  let offset = 0;
  for (const layer of network.layers) {
    const expected = layer.outputs * layer.inputs;
    if (layer.weights.length !== expected || layer.biases.length !== layer.outputs) {
      throw new Error(
        `${network.name}: layer ${layer.inputs}x${layer.outputs} has ` +
          `${layer.weights.length} weights and ${layer.biases.length} biases, ` +
          `expected ${expected} and ${layer.outputs}`,
      );
    }

    packed.set(layer.weights, offset);
    packed.set(layer.biases, offset + expected);
    offset += expected + layer.outputs;
  }

  return packed;
}
