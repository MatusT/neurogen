export {
  FrameGenerator,
  type FrameGeneratorConfig,
  type FrameGeneratorInputs,
  type FrameGeneratorOptions,
} from "./orchestration/FrameGenerator.js";
export { TransferFunction, type LuminanceRange } from "./orchestration/params.js";
export { INTERPOLATED_TEXTURE_FORMAT } from "./orchestration/output.js";
// The asset `installNeuralBlendWeight` takes. Consumers bundler-import or fetch
// src/wgsl/neural/weights/blend_weight_mlp.json, or supply their own retrained
// network of the same shape.
export type { NeuralLayerWeights, NeuralNetworkWeights } from "./wgsl/neural/weights.js";
