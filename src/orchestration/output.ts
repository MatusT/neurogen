// The texture the library hands back, and the pass that fills it from the
// frame interpolation module's output buffer.

import blitWgsl from "../wgsl/generated/orchestration/blit_interpolated.wgsl.js";
import { groupCount, type Size } from "./geometry.js";
import { bindGroup, buf, type ComputeStep, encodeSteps, pipeline, uniformBuffer } from "./gpu.js";

// Fixed rather than configurable: WGSL bakes a storage texture's format into
// its type, so supporting another would mean another entry point.
export const INTERPOLATED_TEXTURE_FORMAT: GPUTextureFormat = "rgba16float";

const BLIT_WORKGROUP = 8;

// BlitParams: a single vec2<i32>.
const BLIT_PARAMS_BYTES = 8;

export class InterpolatedOutput {
  readonly texture: GPUTexture;

  private readonly params: GPUBuffer;
  private readonly step: ComputeStep;

  constructor(device: GPUDevice, renderSize: Size, source: GPUBuffer) {
    const [width, height] = renderSize;

    this.texture = device.createTexture({
      label: "interpolated-color",
      size: [width, height],
      format: INTERPOLATED_TEXTURE_FORMAT,
      // TEXTURE_BINDING and COPY_SRC are for the caller: sampling it into a
      // canvas, or copying it out to inspect.
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC,
    });

    const bytes = new Uint8Array(BLIT_PARAMS_BYTES);
    const view = new DataView(bytes.buffer);
    view.setInt32(0, width, true);
    view.setInt32(4, height, true);
    this.params = uniformBuffer(device, bytes, "blit-params");

    const target = pipeline(device, blitWgsl, "blit-interpolated");
    this.step = {
      pipeline: target,
      bindGroup: bindGroup(device, target, [
        buf(this.params),
        buf(source),
        this.texture.createView(),
      ]),
      groups: [groupCount(width, BLIT_WORKGROUP), groupCount(height, BLIT_WORKGROUP), 1],
    };
  }

  encode(pass: GPUComputePassEncoder): void {
    encodeSteps(pass, [this.step]);
  }

  destroy(): void {
    this.texture.destroy();
    this.params.destroy();
  }
}
