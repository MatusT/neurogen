// The library's whole public surface: configure() once, then prepare() and
// dispatch() per generated frame.
//
//   configure(config)                 resolution, projection, colour encoding
//   prepare(inputs)                   optical flow, then the motion fields and
//                                     the disocclusion mask they feed
//   dispatch()                        blend, inpaint, and hand back the frame
//                                     midway between the two inputs
//
// The split is the SDK's: everything that only depends on this frame's inputs
// runs in prepare(), so a caller can overlap it with its own rendering, and
// dispatch() is the part that produces the image.

import {
  FrameInterpolationStage,
  type FrameBlend,
  type FrameInputs,
} from "./frameInterpolation.js";
import { assertBlendWeightShape } from "./neuralBlendWeight.js";
import { OpticalFlowStage } from "./opticalFlow.js";
import { InterpolatedOutput } from "./output.js";
import { History, TransferFunction, type LuminanceRange } from "./params.js";
import type { NeuralNetworkWeights } from "../wgsl/neural/weights.js";

// One vec4<f32> per pixel. The limit this is checked against defaults to
// 128MiB, which covers 3840x2160 but not 4096x2160.
const BYTES_PER_OUTPUT_PIXEL = 16;

// Below this the level-1 packed-luma loader's four-wide sample window no longer
// fits inside the level, and its clamp range inverts.
const MIN_RENDER_EXTENT = 8;

const DEFAULT_LUMINANCE: LuminanceRange = { min: 0, max: 1000 };

export interface FrameGeneratorOptions {
  device: GPUDevice;
}

export interface FrameGeneratorConfig {
  renderWidth: number;
  renderHeight: number;
  // Finite, non-inverted perspective planes. The disocclusion mask compares
  // depths in view space, which device depth alone cannot be converted back to.
  nearPlane: number;
  farPlane: number;
  verticalFovRadians: number;
  // How the colour inputs are encoded. Defaults to linear LDR.
  transferFunction?: TransferFunction;
  // The display's luminance range in nits, read only by the HDR transfer
  // functions.
  luminance?: LuminanceRange;
}

export type FrameGeneratorInputs = FrameInputs;

export class FrameGenerator {
  readonly device: GPUDevice;

  private opticalFlow: OpticalFlowStage | null = null;
  private frameInterpolation: FrameInterpolationStage | null = null;
  private output: InterpolatedOutput | null = null;
  private neuralWeights: NeuralNetworkWeights | null = null;
  private pending: FrameBlend | null = null;
  private frameIndex = 0;

  constructor(options: FrameGeneratorOptions) {
    this.device = options.device;
  }

  // Allocates every resource the graph needs, sized from the render resolution.
  // Calling it again reallocates: any previously handed-out buffer is destroyed
  // and the next prepare() discards history.
  configure(config: FrameGeneratorConfig): void {
    validate(this.device, config);
    this.destroy();

    const renderSize = [config.renderWidth, config.renderHeight] as const;
    const settings = {
      renderSize,
      transferFunction: config.transferFunction ?? TransferFunction.LinearLdr,
      luminance: config.luminance ?? DEFAULT_LUMINANCE,
    };

    this.opticalFlow = new OpticalFlowStage(this.device, settings);
    this.frameInterpolation = new FrameInterpolationStage(
      this.device,
      {
        ...settings,
        nearPlane: config.nearPlane,
        farPlane: config.farPlane,
        verticalFovRadians: config.verticalFovRadians,
      },
      this.opticalFlow,
    );

    this.output = new InterpolatedOutput(
      this.device,
      renderSize,
      this.frameInterpolation.interpolatedColor,
    );

    if (this.neuralWeights) {
      this.frameInterpolation.installNeuralBlendWeight(this.neuralWeights);
    }
  }

  // Replaces the classical occlusion formula that fills `blendWeight` with a
  // trained network. Takes the parsed asset — `tsc` does not copy the JSON into
  // `dist/`, so the consumer bundler-imports or fetches it. Survives a
  // reconfigure; may be called before or after configure().
  installNeuralBlendWeight(weights: NeuralNetworkWeights): void {
    assertBlendWeightShape(weights);
    this.neuralWeights = weights;
    this.frameInterpolation?.installNeuralBlendWeight(weights);
  }

  prepare(inputs: FrameGeneratorInputs): void {
    const { opticalFlow, frameInterpolation } = this.stages();
    if (this.pending) {
      throw new Error("prepare() called twice without an intervening dispatch()");
    }

    // The very first frame has no previous frame to interpolate from, and no
    // scene-change verdict will say so — the host has to.
    frameInterpolation.setHistory(this.frameIndex === 0 ? History.Discard : History.Keep);
    opticalFlow.setFrameIndex(this.frameIndex);

    const encoder = this.device.createCommandEncoder({ label: "neurogen-prepare" });
    const pass = encoder.beginComputePass();
    // Frame interpolation's setup pass reads this frame's scene-change verdict,
    // so it has to follow the whole optical flow chain, not precede it.
    opticalFlow.encode(pass, inputs.currentColor, inputs.previousColor);
    this.pending = frameInterpolation.encodePrepare(pass, inputs);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // Runs the blend and inpainting and returns the interpolated frame: the frame
  // midway between the two handed to prepare(). Owned by this generator, in
  // INTERPOLATED_TEXTURE_FORMAT, and overwritten by the next dispatch().
  dispatch(): GPUTexture {
    const { frameInterpolation, output } = this.stages();
    if (!this.pending) {
      throw new Error("dispatch() called before prepare()");
    }

    const encoder = this.device.createCommandEncoder({ label: "neurogen-dispatch" });
    const pass = encoder.beginComputePass();
    frameInterpolation.encodeDispatch(pass, this.pending);
    output.encode(pass);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.pending = null;
    this.frameIndex++;

    return output.texture;
  }

  // The same frame as a storage buffer, one vec4<f32> per pixel, row-major,
  // alpha 1 — what the module actually writes, before the copy into the
  // texture. For a caller that would rather bind the buffer than sample.
  get interpolatedColor(): GPUBuffer {
    return this.stages().frameInterpolation.interpolatedColor;
  }

  // The blend weight the installed writer produced for the last dispatch(): one
  // f32 per pixel, row-major, 0 where the warped colour stands and 1 where
  // neither source frame contains the content and it had to be inpainted.
  // Exposed because it is the module's extension point — visualising it is how
  // a caller sees what a replacement writer actually decided.
  get blendWeight(): GPUBuffer {
    return this.stages().frameInterpolation.blendWeight;
  }

  destroy(): void {
    this.opticalFlow?.destroy();
    this.frameInterpolation?.destroy();
    this.output?.destroy();
    this.opticalFlow = null;
    this.frameInterpolation = null;
    this.output = null;
    this.pending = null;
    this.frameIndex = 0;
  }

  private stages(): {
    opticalFlow: OpticalFlowStage;
    frameInterpolation: FrameInterpolationStage;
    output: InterpolatedOutput;
  } {
    if (!this.opticalFlow || !this.frameInterpolation || !this.output) {
      throw new Error("configure() must be called before prepare() or dispatch()");
    }

    return {
      opticalFlow: this.opticalFlow,
      frameInterpolation: this.frameInterpolation,
      output: this.output,
    };
  }
}

function validate(device: GPUDevice, config: FrameGeneratorConfig): void {
  const { renderWidth, renderHeight } = config;
  for (const [name, extent] of [["renderWidth", renderWidth], ["renderHeight", renderHeight]] as const) {
    if (!Number.isInteger(extent) || extent < MIN_RENDER_EXTENT) {
      throw new Error(`${name} must be an integer of at least ${MIN_RENDER_EXTENT}, got ${extent}`);
    }
  }

  // Every colour resource in the graph is an array<vec4<f32>> storage buffer,
  // because WebGPU has no atomics on storage textures and the inpainting pyramid
  // has to be both read and written. Checked here rather than left to fail at
  // allocation, where the message says nothing about the resolution.
  const required = BYTES_PER_OUTPUT_PIXEL * renderWidth * renderHeight;
  const limit = device.limits.maxStorageBufferBindingSize;
  if (required > limit) {
    throw new Error(
      `${renderWidth}x${renderHeight} needs a ${required}-byte colour buffer but this ` +
        `device's maxStorageBufferBindingSize is ${limit}. Request a higher limit when ` +
        "creating the device, or configure a lower resolution.",
    );
  }

  // fiViewSpaceDepth divides by (deviceDepth - far/(far - near)); swapped or
  // zero planes make that a division by zero or a negative view depth, and the
  // disocclusion mask reads the result as metres.
  if (!(config.nearPlane > 0 && config.farPlane > config.nearPlane)) {
    throw new Error(
      `nearPlane and farPlane must satisfy 0 < near < far, got ${config.nearPlane} and ${config.farPlane}`,
    );
  }

  if (!(config.verticalFovRadians > 0 && config.verticalFovRadians < Math.PI)) {
    throw new Error(`verticalFovRadians must be in (0, pi), got ${config.verticalFovRadians}`);
  }
}
