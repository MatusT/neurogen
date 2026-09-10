// Task 5's blend-weight network as a drop-in replacement writer, dispatched
// between preliminary_blend and the inpainting pyramid. It takes the frame
// interpolation uniform at binding 0 unchanged: NeuralBlendParams is a
// byte-compatible prefix of FrameInterpolationParams, which is a documented
// convention rather than an import, so the neural directory stays independent
// of the frame interpolation module.

import blendWeightMlpWgsl from "../wgsl/generated/neural/blend_weight_mlp.wgsl.js";
import { packNeuralWeights, type NeuralNetworkWeights } from "../wgsl/neural/weights.js";
import { bindGroup, buf, type ComputeStep, pipeline } from "./gpu.js";

// blend_weight_mlp.wgsl derives its layer offsets into the flat weight buffer
// from these widths as compile-time constants. An asset of another shape packs
// to a different length and every layer after the first reads someone else's
// weights — silently, since the kernel has no way to notice. The asset is an
// external input, so this is checked at the boundary.
const EXPECTED_LAYER_SHAPE = [
  [3, 8],
  [8, 8],
  [8, 1],
];

// Validated where the asset enters the library rather than where it is packed:
// installing before configure() has no stage to build a pass on yet, and a
// malformed asset should be rejected at the call that supplied it.
export function assertBlendWeightShape(weights: NeuralNetworkWeights): void {
  const shape = weights.layers.map((layer) => [layer.inputs, layer.outputs]);
  if (JSON.stringify(shape) !== JSON.stringify(EXPECTED_LAYER_SHAPE)) {
    throw new Error(
      `neural blend weight asset has layer shape ${JSON.stringify(shape)}, ` +
        `expected ${JSON.stringify(EXPECTED_LAYER_SHAPE)} — ` +
        "blend_weight_mlp.wgsl hard-codes its layer offsets from those widths",
    );
  }
}

export interface NeuralBlendWeightInputs {
  params: GPUBuffer;
  disocclusionMask: GPUBuffer;
  preliminaryColor: GPUBuffer;
  blendWeight: GPUBuffer;
  groups: readonly [number, number, number];
}

export function neuralBlendWeightStep(
  device: GPUDevice,
  weights: NeuralNetworkWeights,
  inputs: NeuralBlendWeightInputs,
): { step: ComputeStep; weightBuffer: GPUBuffer } {
  const packed = packNeuralWeights(weights);
  const weightBuffer = device.createBuffer({
    label: "nn-weights",
    size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuffer, 0, packed);

  const target = pipeline(device, blendWeightMlpWgsl, "fi-neural-blend-weight");

  return {
    step: {
      pipeline: target,
      bindGroup: bindGroup(device, target, [
        buf(inputs.params),
        buf(weightBuffer),
        buf(inputs.disocclusionMask),
        buf(inputs.preliminaryColor),
        buf(inputs.blendWeight),
      ]),
      groups: inputs.groups,
    },
    weightBuffer,
  };
}
