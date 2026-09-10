// Size math mirroring the WGSL helpers that derive every buffer extent and
// dispatch grid from the render resolution. Duplicated in TypeScript rather
// than read back from a shader because host-side allocation has to happen
// before anything is dispatched; each function names the WGSL helper it must
// agree with, and `test/orchestration.test.ts` pins the pairs that are easy to
// get subtly wrong (floor- versus ceil-halving).

export type Size = readonly [number, number];

// FFX_OPTICALFLOW_BLOCK_SIZE — one motion vector per 8x8 block of render pixels.
export const OF_BLOCK_SIZE = 8;

// OpticalFlowParams.pyramidLevelCount. Reduced v1 scope: AMD runs 7.
export const OF_PYRAMID_LEVEL_COUNT = 2;

// FI_INPAINTING_MIP_COUNT in frameinterpolation/common.wgsl.
export const INPAINTING_MIP_COUNT = 4;

export function groupCount(extent: number, workgroupSize: number): number {
  return Math.ceil(extent / workgroupSize);
}

// ofLumaLevelSize. Floor-halving, matching AMD's level textures.
export function lumaLevelSize([width, height]: Size, level: number): Size {
  return [Math.max(width >> level, 1), Math.max(height >> level, 1)];
}

// ofFlowLevelSize. Ceil-halving from a ceil-divided level-0 grid — a
// deliberately different convention from the luma above, matching AMD.
export function flowLevelSize([width, height]: Size, level: number): Size {
  let size: Size = [Math.ceil(width / OF_BLOCK_SIZE), Math.ceil(height / OF_BLOCK_SIZE)];

  for (let i = 0; i < level; i++) {
    size = [Math.ceil(size[0] / 2), Math.ceil(size[1] / 2)];
  }

  return size;
}

// fiOpticalFlowGridSize — the frame interpolation module keeps Task 2's grid.
export function opticalFlowGridSize(renderSize: Size): Size {
  return flowLevelSize(renderSize, 0);
}

// fiPyramidMipSize. Mip 0 is half the render resolution, not full.
export function pyramidMipSize([width, height]: Size, level: number): Size {
  return [Math.max(width >> (level + 1), 1), Math.max(height >> (level + 1), 1)];
}

// fiPyramidMipOffset — all four mips share one buffer, coarsest last.
export function pyramidMipOffset(renderSize: Size, level: number): number {
  let offset = 0;

  for (let i = 0; i < level; i++) {
    const [width, height] = pyramidMipSize(renderSize, i);
    offset += width * height;
  }

  return offset;
}

export function pyramidTexelCount(renderSize: Size): number {
  return pyramidMipOffset(renderSize, INPAINTING_MIP_COUNT);
}

//
// SCENE CHANGE DETECTION — geometry from opticalflow/scd.wgsl and the loop
// bounds of the three passes that consume it.
//

const SCD_HISTOGRAM_BINS = 256;
const SCD_HISTOGRAMS_PER_DIM = 3;
export const SCD_HISTOGRAM_COUNT = SCD_HISTOGRAMS_PER_DIM * SCD_HISTOGRAMS_PER_DIM;
export const SCD_SHIFT_COUNT = 3;
export const SCD_BIN_TOTAL = SCD_HISTOGRAM_COUNT * SCD_HISTOGRAM_BINS;

// scd_histogram.wgsl: PIXELS_PER_STEP, and ROW_STRIDE divided by the 8-row
// workgroup — the pass strides rows by the full dispatch height, so the y
// extent is fixed rather than derived from the resolution.
const SCD_PIXELS_PER_STEP = 4;
const SCD_ROW_GROUPS = 16;
const SCD_HISTOGRAM_WORKGROUP_WIDTH = 32;

export function scdHistogramGroups([width]: Size): [number, number, number] {
  const regionWidth = Math.floor(width / SCD_HISTOGRAMS_PER_DIM);
  const steps = groupCount(regionWidth, SCD_PIXELS_PER_STEP);

  return [groupCount(steps, SCD_HISTOGRAM_WORKGROUP_WIDTH), SCD_ROW_GROUPS, SCD_HISTOGRAM_COUNT];
}
