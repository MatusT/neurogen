// Task 2's eight passes wired into one stage. Produces the module's frozen
// output — a motion vector field and its per-cell validity on a 1/8-resolution
// grid, plus a global scene-change flag — which the frame interpolation stage
// then consumes.
//
// Pass order, from opticalflow/params.wgsl and the passes' own headers:
//
//   prepare_luma (current) -> prepare_luma (previous)
//   -> luminance_pyramid (current, level 1) -> luminance_pyramid (previous)
//   -> scd_histogram -> scd_divergence -> scd_finalize
//   -> for level 1 down to 0:
//        compute_optical_flow -> filter_optical_flow -> scale (levels above 0)
//
// The luma of *both* input frames is recomputed every frame rather than the
// previous frame's being carried over. `prepare()` is handed both colour
// textures, so recomputing removes the need to track which texture was last
// frame's. Cost is two extra dispatches out of the fourteen here.
//
// The stage is still not stateless across frames. `scdPreviousHistogram` is
// this frame's histogram waiting to be compared against the next, which is the
// whole basis of scene-change detection; and at render widths that are not a
// multiple of 16 the level-1 search's dispatch falls a cell short of the
// ceil-halved flow grid, so those cells keep the previous frame's vector. Task
// 2 documented the second as benign — it feeds the median filter as one more
// tap, and level >= 1 validity is never read downstream.

import prepareLumaWgsl from "../wgsl/generated/opticalflow/prepare_luma.wgsl.js";
import luminancePyramidWgsl from "../wgsl/generated/opticalflow/luminance_pyramid.wgsl.js";
import scdHistogramWgsl from "../wgsl/generated/opticalflow/scd_histogram.wgsl.js";
import scdDivergenceWgsl from "../wgsl/generated/opticalflow/scd_divergence.wgsl.js";
import scdFinalizeWgsl from "../wgsl/generated/opticalflow/scd_finalize.wgsl.js";
import computeOpticalFlowWgsl from "../wgsl/generated/opticalflow/compute_optical_flow.wgsl.js";
import filterOpticalFlowWgsl from "../wgsl/generated/opticalflow/filter_optical_flow.wgsl.js";
import scaleOpticalFlowWgsl from "../wgsl/generated/opticalflow/scale_optical_flow.wgsl.js";
import {
  flowLevelSize,
  groupCount,
  lumaLevelSize,
  OF_PYRAMID_LEVEL_COUNT,
  SCD_BIN_TOTAL,
  SCD_HISTOGRAM_COUNT,
  SCD_SHIFT_COUNT,
  scdHistogramGroups,
  type Size,
} from "./geometry.js";
import {
  bindGroup,
  buf,
  type ComputeStep,
  encodeSteps,
  pipeline,
  step,
  storageBuffer,
  UNREFERENCED,
  uniformBuffer,
} from "./gpu.js";
import { packOpticalFlowParams, type LuminanceRange, type TransferFunction } from "./params.js";

const BYTES_PER_U32 = 4;
const BYTES_PER_FLOW_CELL = 8;

// prepare_luma covers a 2x2 quad per invocation.
const LUMA_PIXELS_PER_THREAD = 2;
const LUMA_WORKGROUP = 16;
const PYRAMID_WORKGROUP = 8;
const SEARCH_PIXELS_PER_GROUP = 16;
const FILTER_WORKGROUP: Size = [16, 4];
const SCALE_WORKGROUP = 4;

const COARSEST_LEVEL = OF_PYRAMID_LEVEL_COUNT - 1;
const FINEST_LEVEL = 0;

export interface OpticalFlowSettings {
  renderSize: Size;
  transferFunction: TransferFunction;
  luminance: LuminanceRange;
}

export class OpticalFlowStage {
  // Task 2's frozen module output, at ceil(renderSize / 8).
  readonly vectorField: GPUBuffer;
  readonly validity: GPUBuffer;
  readonly sceneChange: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly settings: OpticalFlowSettings;
  // One per pyramid level; every pass reads its level from the uniform, and a
  // uniform cannot be rewritten between dispatches inside one submission.
  private readonly params: GPUBuffer[] = [];
  private readonly lumaCurrent: GPUBuffer[] = [];
  private readonly lumaPrevious: GPUBuffer[] = [];
  private readonly prepareLuma: GPUComputePipeline;
  // Everything after the two prepare_luma dispatches. Those two are the only
  // steps that bind a caller-owned texture, so they are the only ones rebuilt
  // per frame.
  private readonly steps: readonly ComputeStep[];
  private readonly owned: GPUBuffer[] = [];

  constructor(device: GPUDevice, settings: OpticalFlowSettings) {
    this.device = device;
    this.settings = settings;
    const { renderSize } = settings;

    for (let level = 0; level < OF_PYRAMID_LEVEL_COUNT; level++) {
      this.params.push(this.own(uniformBuffer(device, this.packParams(level, 0), `of-params-${level}`)));
      const [width, height] = lumaLevelSize(renderSize, level);
      this.lumaCurrent.push(this.buffer(width * height * BYTES_PER_U32, `of-luma-current-${level}`));
      this.lumaPrevious.push(this.buffer(width * height * BYTES_PER_U32, `of-luma-previous-${level}`));
    }

    // Vectors and validity are ping-ponged per level: the median filter reads a
    // 3x3 neighbourhood, so it cannot run in place. At level 0 the filter's
    // destination *is* the module output, mirroring AMD's rebinding.
    const flowSearch: GPUBuffer[] = [];
    const flowFiltered: GPUBuffer[] = [];
    const validitySearch: GPUBuffer[] = [];
    const validityFiltered: GPUBuffer[] = [];
    for (let level = 0; level < OF_PYRAMID_LEVEL_COUNT; level++) {
      const [width, height] = flowLevelSize(renderSize, level);
      const cells = width * height;
      flowSearch.push(this.buffer(cells * BYTES_PER_FLOW_CELL, `of-flow-search-${level}`));
      flowFiltered.push(this.buffer(cells * BYTES_PER_FLOW_CELL, `of-flow-filtered-${level}`));
      validitySearch.push(this.buffer(cells * BYTES_PER_U32, `of-validity-search-${level}`));
      // Above level 0 this one is a dead sink: only the scale pass consumes a
      // coarse level, and it takes vectors without validity. It still has to
      // exist for the filter's binding.
      validityFiltered.push(this.buffer(cells * BYTES_PER_U32, `of-validity-filtered-${level}`));
    }
    this.vectorField = flowFiltered[FINEST_LEVEL];
    this.validity = validityFiltered[FINEST_LEVEL];

    const scdHistogram = this.buffer(SCD_BIN_TOTAL * BYTES_PER_U32, "of-scd-histogram");
    const scdPreviousHistogram = this.buffer(SCD_BIN_TOTAL * BYTES_PER_U32, "of-scd-previous");
    // Task 2 added this over AMD's resource list to break a read/write race
    // between the divergence pass's three shifts.
    const scdFiltered = this.buffer(SCD_BIN_TOTAL * BYTES_PER_U32, "of-scd-filtered");
    const scdTemp = this.buffer(SCD_SHIFT_COUNT * BYTES_PER_U32, "of-scd-temp");
    this.sceneChange = this.buffer(BYTES_PER_U32, "of-scene-change");

    this.prepareLuma = pipeline(device, prepareLumaWgsl, "of-prepare-luma");
    const pyramid = pipeline(device, luminancePyramidWgsl, "of-luminance-pyramid");
    const histogram = pipeline(device, scdHistogramWgsl, "of-scd-histogram");
    const divergence = pipeline(device, scdDivergenceWgsl, "of-scd-divergence");
    const finalize = pipeline(device, scdFinalizeWgsl, "of-scd-finalize");
    const search = pipeline(device, computeOpticalFlowWgsl, "of-compute");
    const filter = pipeline(device, filterOpticalFlowWgsl, "of-filter");
    const scale = pipeline(device, scaleOpticalFlowWgsl, "of-scale");

    const encode = (
      target: GPUComputePipeline,
      resources: readonly (GPUBindingResource | undefined)[],
      groups: readonly [number, number, number],
    ): ComputeStep => step(device, target, resources, groups);

    const steps: ComputeStep[] = [];

    for (let level = 1; level < OF_PYRAMID_LEVEL_COUNT; level++) {
      const [width, height] = lumaLevelSize(renderSize, level);
      const groups = [
        groupCount(width, PYRAMID_WORKGROUP),
        groupCount(height, PYRAMID_WORKGROUP),
        1,
      ] as const;
      for (const luma of [this.lumaCurrent, this.lumaPrevious]) {
        steps.push(
          encode(pyramid, [buf(this.params[level]), buf(luma[level - 1]), buf(luma[level])], groups),
        );
      }
    }

    steps.push(
      encode(
        histogram,
        [buf(this.params[FINEST_LEVEL]), buf(this.lumaCurrent[FINEST_LEVEL]), buf(scdHistogram)],
        scdHistogramGroups(renderSize),
      ),
      encode(
        divergence,
        [
          UNREFERENCED,
          buf(scdHistogram),
          buf(scdPreviousHistogram),
          buf(scdFiltered),
          buf(scdTemp),
        ],
        [SCD_HISTOGRAM_COUNT, SCD_SHIFT_COUNT, 1],
      ),
      encode(
        finalize,
        [
          buf(this.params[FINEST_LEVEL]),
          buf(scdHistogram),
          buf(scdPreviousHistogram),
          buf(scdFiltered),
          buf(scdTemp),
          buf(this.sceneChange),
        ],
        [1, 1, 1],
      ),
    );

    for (let level = COARSEST_LEVEL; level >= FINEST_LEVEL; level--) {
      const params = buf(this.params[level]);
      const [lumaWidth, lumaHeight] = lumaLevelSize(renderSize, level);
      const [flowWidth, flowHeight] = flowLevelSize(renderSize, level);

      steps.push(
        encode(
          search,
          [
            params,
            buf(this.lumaCurrent[level]),
            buf(this.lumaPrevious[level]),
            // Read-write: at every level but the coarsest, the scale pass has
            // already written this frame's prediction here.
            buf(flowSearch[level]),
            buf(this.sceneChange),
            buf(validitySearch[level]),
          ],
          [
            groupCount(lumaWidth, SEARCH_PIXELS_PER_GROUP),
            groupCount(lumaHeight, SEARCH_PIXELS_PER_GROUP),
            1,
          ],
        ),
        encode(
          filter,
          [
            params,
            buf(flowSearch[level]),
            buf(flowFiltered[level]),
            buf(validitySearch[level]),
            buf(validityFiltered[level]),
          ],
          [
            groupCount(flowWidth, FILTER_WORKGROUP[0]),
            groupCount(flowHeight, FILTER_WORKGROUP[1]),
            1,
          ],
        ),
      );

      if (level === FINEST_LEVEL) { continue; }

      // Dispatched over the *destination* grid with the uniform still naming
      // the source level, which is what the pass's header specifies.
      const [destWidth, destHeight] = flowLevelSize(renderSize, level - 1);
      steps.push(
        encode(
          scale,
          [
            params,
            buf(this.lumaCurrent[level]),
            buf(this.lumaPrevious[level]),
            buf(flowFiltered[level]),
            buf(flowSearch[level - 1]),
            buf(this.sceneChange),
          ],
          [groupCount(destWidth, SCALE_WORKGROUP), groupCount(destHeight, SCALE_WORKGROUP), 1],
        ),
      );
    }

    this.steps = steps;
  }

  // scd_finalize forces a scene change while `frameIndex` is inside its warmup
  // window, so this has to advance for the flag to ever go quiet.
  setFrameIndex(frameIndex: number): void {
    for (let level = 0; level < OF_PYRAMID_LEVEL_COUNT; level++) {
      this.device.queue.writeBuffer(this.params[level], 0, this.packParams(level, frameIndex));
    }
  }

  encode(pass: GPUComputePassEncoder, currentColor: GPUTextureView, previousColor: GPUTextureView): void {
    const [width, height] = this.settings.renderSize;
    const groups = [
      groupCount(Math.ceil(width / LUMA_PIXELS_PER_THREAD), LUMA_WORKGROUP),
      groupCount(Math.ceil(height / LUMA_PIXELS_PER_THREAD), LUMA_WORKGROUP),
      1,
    ] as const;

    // Rebuilt every frame: the caller owns these textures and may hand over a
    // different pair each call, so a bind group holding them cannot be cached.
    const lumaSteps: ComputeStep[] = [
      [currentColor, this.lumaCurrent[FINEST_LEVEL]] as const,
      [previousColor, this.lumaPrevious[FINEST_LEVEL]] as const,
    ].map(([color, target]) => ({
      pipeline: this.prepareLuma,
      bindGroup: bindGroup(this.device, this.prepareLuma, [
        buf(this.params[FINEST_LEVEL]),
        color,
        buf(target),
      ]),
      groups,
    }));

    encodeSteps(pass, lumaSteps);
    encodeSteps(pass, this.steps);
  }

  destroy(): void {
    for (const buffer of this.owned) {
      buffer.destroy();
    }
    this.owned.length = 0;
  }

  private packParams(pyramidLevel: number, frameIndex: number): Uint8Array<ArrayBuffer> {
    return packOpticalFlowParams({
      renderSize: this.settings.renderSize,
      pyramidLevel,
      pyramidLevelCount: OF_PYRAMID_LEVEL_COUNT,
      frameIndex,
      transferFunction: this.settings.transferFunction,
      luminance: this.settings.luminance,
    });
  }

  private buffer(byteLength: number, label: string): GPUBuffer {
    return this.own(storageBuffer(this.device, byteLength, label));
  }

  private own(buffer: GPUBuffer): GPUBuffer {
    this.owned.push(buffer);

    return buffer;
  }
}
