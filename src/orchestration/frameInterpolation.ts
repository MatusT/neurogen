// Task 4's eight passes wired into one stage, split across the two phases the
// public API exposes. Consumes the optical flow module's frozen output and
// produces the interpolated frame.
//
// Pass order, from setup.wgsl's ordering note and preliminary_blend.wgsl's
// contract:
//
//   prepare:  setup -> reconstruct_and_dilate -> game_motion_vector_field
//             -> optical_flow_vector_field -> disocclusion_mask
//   dispatch: preliminary_blend -> [neural blend weight writer]
//             -> inpainting_pyramid mips 0..3 -> final_blend
//
// `setup` owns every per-frame reset in the module and must run after the
// optical flow module has published this frame's scene-change verdict, which it
// reads. Everything else this module produces is fully overwritten by its own
// pass, so nothing else is cleared.

import setupWgsl from "../wgsl/generated/frameinterpolation/setup.wgsl.js";
import reconstructAndDilateWgsl from "../wgsl/generated/frameinterpolation/reconstruct_and_dilate.wgsl.js";
import gameMotionVectorFieldWgsl from "../wgsl/generated/frameinterpolation/game_motion_vector_field.wgsl.js";
import opticalFlowVectorFieldWgsl from "../wgsl/generated/frameinterpolation/optical_flow_vector_field.wgsl.js";
import disocclusionMaskWgsl from "../wgsl/generated/frameinterpolation/disocclusion_mask.wgsl.js";
import preliminaryBlendWgsl from "../wgsl/generated/frameinterpolation/preliminary_blend.wgsl.js";
import inpaintingPyramidWgsl from "../wgsl/generated/frameinterpolation/inpainting_pyramid.wgsl.js";
import finalBlendWgsl from "../wgsl/generated/frameinterpolation/final_blend.wgsl.js";
import type { NeuralNetworkWeights } from "../wgsl/neural/weights.js";
import {
  groupCount,
  INPAINTING_MIP_COUNT,
  opticalFlowGridSize,
  pyramidMipSize,
  pyramidTexelCount,
  type Size,
} from "./geometry.js";
import {
  buf,
  type ComputeStep,
  encodeSteps,
  pipeline,
  step,
  storageBuffer,
  uniformBuffer,
} from "./gpu.js";
import { neuralBlendWeightStep } from "./neuralBlendWeight.js";
import {
  History,
  packFrameInterpolationParams,
  type LuminanceRange,
  type TransferFunction,
} from "./params.js";

const BYTES_PER_F32 = 4;
const BYTES_PER_U32 = 4;
const FRAME_WORKGROUP = 8;

export interface FrameInterpolationSettings {
  renderSize: Size;
  nearPlane: number;
  farPlane: number;
  verticalFovRadians: number;
  transferFunction: TransferFunction;
  luminance: LuminanceRange;
}

// All at render resolution, in the encodings this module's input contract
// names.
export interface FrameInputs {
  currentColor: GPUTextureView;
  previousColor: GPUTextureView;
  // .x is device depth, standard 0 = near / 1 = far, from a finite perspective
  // projection.
  depth: GPUTextureView;
  // .xy is the displacement in whole render-resolution pixels from a pixel in
  // the current frame to where it was in the previous frame.
  motionVectors: GPUTextureView;
}

export interface OpticalFlowInputs {
  vectorField: GPUBuffer;
  validity: GPUBuffer;
  sceneChange: GPUBuffer;
}

// The two passes that bind the caller's colour textures but belong to the
// dispatch phase. Handed back by `encodePrepare` so the phase split needs no
// mutable state inside this stage — the caller already holds the one that says
// whether a frame is in flight.
export interface FrameBlend {
  preliminaryBlend: ComputeStep;
  finalBlend: ComputeStep;
}

export class FrameInterpolationStage {
  // The module output: one vec4<f32> per render-resolution pixel, row-major.
  readonly interpolatedColor: GPUBuffer;
  // The module's extension point, whichever writer filled it.
  readonly blendWeight: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly settings: FrameInterpolationSettings;
  // One per inpainting mip. Only the pyramid reads the level, but a uniform
  // cannot be rewritten between dispatches in one submission, so every pass
  // other than the pyramid takes mip 0's copy.
  private readonly params: GPUBuffer[] = [];
  private readonly pipelines: Record<string, GPUComputePipeline>;
  private readonly buffers: Record<string, GPUBuffer>;
  private readonly opticalFlow: OpticalFlowInputs;
  // Built once: every step whose bindings are all buffers this stage owns.
  private readonly setupStep: ComputeStep;
  private readonly disocclusionStep: ComputeStep;
  private readonly pyramidSteps: readonly ComputeStep[];
  private neural: ComputeStep | null = null;
  // Tracked apart from `owned` so a second install can release the first
  // network's weights rather than holding them until destroy().
  private neuralWeights: GPUBuffer | null = null;
  private readonly owned: GPUBuffer[] = [];

  constructor(
    device: GPUDevice,
    settings: FrameInterpolationSettings,
    opticalFlow: OpticalFlowInputs,
  ) {
    this.device = device;
    this.settings = settings;
    this.opticalFlow = opticalFlow;

    const { renderSize } = settings;
    const [width, height] = renderSize;
    const pixels = width * height;
    const [gridWidth, gridHeight] = opticalFlowGridSize(renderSize);

    for (let mip = 0; mip < INPAINTING_MIP_COUNT; mip++) {
      this.params.push(
        this.own(uniformBuffer(device, this.packParams(mip, History.Keep), `fi-params-${mip}`)),
      );
    }

    this.buffers = {
      // Two interleaved u32 entries per pixel, x then y, so one atomic max
      // resolves each component's scatter.
      gameField: this.buffer(pixels * 2 * BYTES_PER_U32, "fi-game-field"),
      opticalFlowField: this.buffer(gridWidth * gridHeight * 2 * BYTES_PER_U32, "fi-of-field"),
      depthPrevious: this.buffer(pixels * BYTES_PER_U32, "fi-depth-previous"),
      depthInterpolated: this.buffer(pixels * BYTES_PER_U32, "fi-depth-interpolated"),
      // frameIndexSinceLastReset, advanced by setup and read by both blends.
      state: this.buffer(BYTES_PER_U32, "fi-state"),
      dilatedDepth: this.buffer(pixels * BYTES_PER_F32, "fi-dilated-depth"),
      dilatedMotionVectors: this.buffer(pixels * 2 * BYTES_PER_F32, "fi-dilated-mv"),
      disocclusionMask: this.buffer(pixels * 2 * BYTES_PER_F32, "fi-disocclusion-mask"),
      preliminaryColor: this.buffer(pixels * 4 * BYTES_PER_F32, "fi-preliminary-color"),
      blendWeight: this.buffer(pixels * BYTES_PER_F32, "fi-blend-weight"),
      inpaintingPyramid: this.buffer(
        pyramidTexelCount(renderSize) * 4 * BYTES_PER_F32,
        "fi-inpainting-pyramid",
      ),
      interpolatedColor: this.buffer(pixels * 4 * BYTES_PER_F32, "fi-interpolated-color"),
    };
    this.interpolatedColor = this.buffers.interpolatedColor;
    this.blendWeight = this.buffers.blendWeight;

    this.pipelines = {
      setup: pipeline(device, setupWgsl, "fi-setup"),
      reconstructAndDilate: pipeline(device, reconstructAndDilateWgsl, "fi-reconstruct-dilate"),
      gameMotionVectorField: pipeline(device, gameMotionVectorFieldWgsl, "fi-game-mv-field"),
      opticalFlowVectorField: pipeline(device, opticalFlowVectorFieldWgsl, "fi-of-mv-field"),
      disocclusionMask: pipeline(device, disocclusionMaskWgsl, "fi-disocclusion-mask"),
      preliminaryBlend: pipeline(device, preliminaryBlendWgsl, "fi-preliminary-blend"),
      inpaintingPyramid: pipeline(device, inpaintingPyramidWgsl, "fi-inpainting-pyramid"),
      finalBlend: pipeline(device, finalBlendWgsl, "fi-final-blend"),
    };

    const b = this.buffers;
    const params = buf(this.params[0]);
    const frameGroups = this.frameGroups();

    this.setupStep = this.buildStep(
      this.pipelines.setup,
      [
        params,
        buf(b.gameField),
        buf(b.opticalFlowField),
        buf(b.depthPrevious),
        buf(b.depthInterpolated),
        buf(this.opticalFlow.sceneChange),
        buf(b.state),
      ],
      frameGroups,
    );
    this.disocclusionStep = this.buildStep(
      this.pipelines.disocclusionMask,
      [
        params,
        buf(b.depthInterpolated),
        buf(b.depthPrevious),
        buf(b.dilatedDepth),
        buf(b.gameField),
        buf(b.disocclusionMask),
      ],
      frameGroups,
    );

    const pyramidSteps: ComputeStep[] = [];
    for (let mip = 0; mip < INPAINTING_MIP_COUNT; mip++) {
      const [mipWidth, mipHeight] = pyramidMipSize(renderSize, mip);
      pyramidSteps.push(
        this.buildStep(
          this.pipelines.inpaintingPyramid,
          [buf(this.params[mip]), buf(b.preliminaryColor), buf(b.blendWeight), buf(b.inpaintingPyramid)],
          [groupCount(mipWidth, FRAME_WORKGROUP), groupCount(mipHeight, FRAME_WORKGROUP), 1],
        ),
      );
    }
    this.pyramidSteps = pyramidSteps;
  }

  // Overwrites `blendWeight` with Task 5's network instead of the classical
  // occlusion formula. Takes the parsed asset rather than a path: `tsc` does not
  // copy the JSON into `dist/`, so the consumer bundler-imports or fetches it.
  installNeuralBlendWeight(weights: NeuralNetworkWeights): void {
    // Build the replacement before touching the old one — neuralBlendWeightStep
    // can still throw here (e.g. a weights array truncated relative to its own
    // declared shape), and destroying the working buffer first would leave the
    // retained dispatch step bound to freed GPU state with no way back.
    const { step: neural, weightBuffer } = neuralBlendWeightStep(this.device, weights, {
      params: this.params[0],
      disocclusionMask: this.buffers.disocclusionMask,
      preliminaryColor: this.buffers.preliminaryColor,
      blendWeight: this.buffers.blendWeight,
      groups: this.frameGroups(),
    });
    this.neuralWeights?.destroy();
    this.neuralWeights = weightBuffer;
    this.neural = neural;
  }

  // Discarding history is required on the very first frame: there is no
  // previous frame to interpolate from and no scene-change verdict will say so.
  setHistory(history: History): void {
    for (let mip = 0; mip < INPAINTING_MIP_COUNT; mip++) {
      this.device.queue.writeBuffer(this.params[mip], 0, this.packParams(mip, history));
    }
  }

  encodePrepare(pass: GPUComputePassEncoder, inputs: FrameInputs): FrameBlend {
    const b = this.buffers;
    const params = buf(this.params[0]);
    const groups = this.frameGroups();
    const [gridWidth, gridHeight] = opticalFlowGridSize(this.settings.renderSize);

    // Only the steps that bind one of the caller's four input textures are
    // rebuilt: the caller may hand over a different set each call, so a bind
    // group holding them cannot be cached. The rest were built at construction.
    const prepare: ComputeStep[] = [
      this.setupStep,
      this.buildStep(
        this.pipelines.reconstructAndDilate,
        [
          params,
          inputs.depth,
          inputs.motionVectors,
          buf(b.dilatedDepth),
          buf(b.dilatedMotionVectors),
          buf(b.depthPrevious),
        ],
        groups,
      ),
      this.buildStep(
        this.pipelines.gameMotionVectorField,
        [
          params,
          buf(b.dilatedDepth),
          buf(b.dilatedMotionVectors),
          inputs.currentColor,
          inputs.previousColor,
          buf(b.gameField),
          buf(b.depthInterpolated),
        ],
        groups,
      ),
      this.buildStep(
        this.pipelines.opticalFlowVectorField,
        [
          params,
          buf(this.opticalFlow.vectorField),
          buf(this.opticalFlow.validity),
          inputs.currentColor,
          inputs.previousColor,
          buf(b.opticalFlowField),
        ],
        [groupCount(gridWidth, FRAME_WORKGROUP), groupCount(gridHeight, FRAME_WORKGROUP), 1],
      ),
      this.disocclusionStep,
    ];

    const preliminaryBlend = this.buildStep(
      this.pipelines.preliminaryBlend,
      [
        params,
        inputs.currentColor,
        inputs.previousColor,
        buf(b.gameField),
        buf(b.opticalFlowField),
        buf(b.disocclusionMask),
        buf(b.preliminaryColor),
        buf(b.blendWeight),
        buf(b.state),
      ],
      groups,
    );
    const finalBlend = this.buildStep(
      this.pipelines.finalBlend,
      [
        params,
        inputs.currentColor,
        buf(b.preliminaryColor),
        buf(b.blendWeight),
        buf(b.inpaintingPyramid),
        buf(b.interpolatedColor),
        buf(b.state),
      ],
      groups,
    );

    encodeSteps(pass, prepare);

    return { preliminaryBlend, finalBlend };
  }

  encodeDispatch(pass: GPUComputePassEncoder, blend: FrameBlend): void {
    encodeSteps(pass, [blend.preliminaryBlend]);

    if (this.neural) {
      encodeSteps(pass, [this.neural]);
    }

    encodeSteps(pass, this.pyramidSteps);
    encodeSteps(pass, [blend.finalBlend]);
  }

  destroy(): void {
    for (const buffer of this.owned) {
      buffer.destroy();
    }
    this.owned.length = 0;
    this.neuralWeights?.destroy();
    this.neuralWeights = null;
  }

  private buildStep(
    target: GPUComputePipeline,
    resources: readonly (GPUBindingResource | undefined)[],
    groups: readonly [number, number, number],
  ): ComputeStep {
    return step(this.device, target, resources, groups);
  }

  private frameGroups(): readonly [number, number, number] {
    const [width, height] = this.settings.renderSize;

    return [groupCount(width, FRAME_WORKGROUP), groupCount(height, FRAME_WORKGROUP), 1];
  }

  private packParams(inpaintingMipLevel: number, reset: History): Uint8Array<ArrayBuffer> {
    return packFrameInterpolationParams({ ...this.settings, inpaintingMipLevel, reset });
  }

  private buffer(byteLength: number, label: string): GPUBuffer {
    return this.own(storageBuffer(this.device, byteLength, label));
  }

  private own(buffer: GPUBuffer): GPUBuffer {
    this.owned.push(buffer);

    return buffer;
  }
}
