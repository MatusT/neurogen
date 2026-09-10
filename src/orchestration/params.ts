// Host-side packing of the two uniform structs the WGSL passes read. TypeScript
// has no compiler check against a WGSL struct, so the byte offsets below are
// part of the frozen contract: OpticalFlowParams in opticalflow/params.wgsl and
// FrameInterpolationParams in frameinterpolation/params.wgsl.

import type { Size } from "./geometry.js";

// OpticalFlowParams.backbufferTransferFunction, and the identically-encoded
// FrameInterpolationParams field.
export enum TransferFunction {
  LinearLdr = 0,
  Pq = 1,
  ScRgb = 2,
}

// FrameInterpolationParams.reset — a host-forced history discard, folded into
// the same counter the GPU-side scene-change verdict drives.
export enum History {
  Keep = 0,
  Discard = 1,
}

export interface LuminanceRange {
  min: number;
  max: number;
}

// vec2<i32> then four scalars then vec2<f32> at its 8-byte alignment: 32 bytes.
const OPTICAL_FLOW_PARAMS_BYTES = 32;

export interface OpticalFlowParams {
  renderSize: Size;
  pyramidLevel: number;
  pyramidLevelCount: number;
  frameIndex: number;
  transferFunction: TransferFunction;
  luminance: LuminanceRange;
}

export function packOpticalFlowParams(params: OpticalFlowParams): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(OPTICAL_FLOW_PARAMS_BYTES);
  const view = new DataView(bytes.buffer);

  view.setInt32(0, params.renderSize[0], true);
  view.setInt32(4, params.renderSize[1], true);
  view.setUint32(8, params.pyramidLevel, true);
  view.setUint32(12, params.pyramidLevelCount, true);
  view.setUint32(16, params.frameIndex, true);
  view.setUint32(20, params.transferFunction, true);
  view.setFloat32(24, params.luminance.min, true);
  view.setFloat32(28, params.luminance.max, true);

  return bytes;
}

// The struct itself is 40 bytes; padded to 48, which is what Task 4's fixtures
// bind and what Task 5's kernel reads its byte-compatible prefix from.
const FRAME_INTERPOLATION_PARAMS_BYTES = 48;

export interface FrameInterpolationParams {
  renderSize: Size;
  nearPlane: number;
  farPlane: number;
  verticalFovRadians: number;
  transferFunction: TransferFunction;
  luminance: LuminanceRange;
  inpaintingMipLevel: number;
  reset: History;
}

export function packFrameInterpolationParams(
  params: FrameInterpolationParams,
): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(FRAME_INTERPOLATION_PARAMS_BYTES);
  const view = new DataView(bytes.buffer);

  view.setInt32(0, params.renderSize[0], true);
  view.setInt32(4, params.renderSize[1], true);
  view.setFloat32(8, params.nearPlane, true);
  view.setFloat32(12, params.farPlane, true);
  view.setFloat32(16, params.verticalFovRadians, true);
  view.setUint32(20, params.transferFunction, true);
  view.setFloat32(24, params.luminance.min, true);
  view.setFloat32(28, params.luminance.max, true);
  view.setUint32(32, params.inpaintingMipLevel, true);
  view.setUint32(36, params.reset, true);

  return bytes;
}
