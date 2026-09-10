// The quantitative check the demo exists to make: a bar crossing the frame at
// constant velocity, and the interpolated frame's copy of it measured against
// the midpoint of the two real frames it was generated from.
//
// Two things have to hold, and only together do they mean anything:
//
//   position  the bar's centroid sits halfway between the two real frames'
//   shape     it is still one bar of the original width
//
// The bar is narrower than its per-frame travel, so a pipeline that merely
// crossfaded the two real frames would leave two bars with a gap between them.
// That has the same centroid as the correct answer — the position test alone
// cannot tell the two apart, which is why the shape test is not decoration.

import { FrameGenerator } from "../src/orchestration/FrameGenerator.js";
import type { NeuralNetworkWeights } from "../src/wgsl/neural/weights.js";
import { GBuffer } from "./gbuffer.js";
import { markerAt, markerGeometry, markerPixelX, PROJECTION, projectionFor } from "./scene.js";

// Past both of the module's warmups: the scene-change detector forces a cut for
// its first six frames, and the preliminary blend trusts the game motion
// vectors unconditionally for ten frames after that. Only beyond both is the
// whole graph in play.
export const SETTLED_FRAMES = 24;

// The centroid is a subpixel measure over hundreds of rows, so a pixel is a
// loose bound on a correct warp and far inside the 16px error a whole-frame
// mistiming would produce at this velocity.
export const MIDPOINT_TOLERANCE_PIXELS = 1;

// copyTextureToBuffer's row pitch.
const COPY_ROW_ALIGNMENT = 256;
const BYTES_PER_HALF4 = 8;
const CHANNELS = 4;

// Only the exponent range a frame of linear LDR colour spans.
function decodeHalf(bits: number): number {
  const sign = bits >>> 15 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    return sign * mantissa * 2 ** -24;
  }

  return sign * (mantissa + 1024) * 2 ** (exponent - 25);
}

async function readColor(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const { width, height } = texture;
  const paddedRowBytes =
    Math.ceil((width * BYTES_PER_HALF4) / COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
  const readback = device.createBuffer({
    size: paddedRowBytes * height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const encoder = device.createCommandEncoder({ label: "demo-readback" });
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: readback, bytesPerRow: paddedRowBytes },
    { width, height },
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const padded = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();

  const texels = new Float32Array(width * height * CHANNELS);
  for (let y = 0; y < height; y++) {
    const row = new Uint16Array(padded, y * paddedRowBytes, width * CHANNELS);
    for (let i = 0; i < row.length; i++) {
      texels[y * width * CHANNELS + i] = decodeHalf(row[i]);
    }
  }

  return texels;
}

export interface MarkerProfile {
  // Weighted mean column of the bar itself, in pixels.
  centroid: number;
  // Columns carrying more than half the peak column's weight: the bar's width.
  span: number;
  // How many separate bars those columns form. Two means a crossfade.
  bars: number;
  // Fraction of the frame's red lying outside the bar. The blend leaves a
  // partial trail behind a fast edge, and averaging it into the centroid would
  // report that ghost as a timing error, so it is measured rather than mixed in.
  ghost: number;
}

// Runs of columns carrying more than half the peak column's weight. The widest
// is the bar; a second one is the signature of a crossfade.
function barRuns(columns: Float32Array, peak: number): { from: number; to: number }[] {
  const half = peak / 2;
  const runs: { from: number; to: number }[] = [];
  let start = -1;

  for (let x = 0; x <= columns.length; x++) {
    const above = x < columns.length && columns[x] > half;
    if (above) {
      start = start < 0 ? x : start;
      continue;
    }

    if (start < 0) {
      continue;
    }

    runs.push({ from: start, to: x - 1 });
    start = -1;
  }

  return runs;
}

// The marker is the only red thing in the scene and the backdrop is blue-grey,
// so red beyond both other channels isolates it exactly rather than by a
// threshold, and survives the blend attenuating it.
export function measureMarker(
  texels: Float32Array,
  size: readonly [number, number],
): MarkerProfile {
  const [width, height] = size;
  const columns = new Float32Array(width);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * CHANNELS;
      columns[x] += Math.max(0, texels[at] - Math.max(texels[at + 1], texels[at + 2]));
    }
  }

  let total = 0;
  let peak = 0;
  for (let x = 0; x < width; x++) {
    total += columns[x];
    peak = Math.max(peak, columns[x]);
  }

  const runs = barRuns(columns, peak);
  const bar = runs.reduce((widest, run) => (run.to - run.from > widest.to - widest.from ? run : widest), {
    from: 0,
    to: -1,
  });
  // One column either side: an edge the bar covers only partly falls below the
  // half-peak threshold but is still the bar, and dropping it would bias the
  // centroid towards whichever end happened to land on a pixel boundary.
  const from = Math.max(0, bar.from - 1);
  const to = Math.min(width - 1, bar.to + 1);

  let weight = 0;
  let moment = 0;
  for (let x = from; x <= to; x++) {
    weight += columns[x];
    // Pixel centres, so the centroid is in the same continuous coordinates
    // projectToPixel reports rather than half a pixel to the left of them.
    moment += columns[x] * (x + 0.5);
  }

  return {
    centroid: weight > 0 ? moment / weight : Number.NaN,
    span: bar.to - bar.from + 1,
    bars: runs.length,
    ghost: total > 0 ? (total - weight) / total : 0,
  };
}

export interface MidpointMeasurement {
  size: readonly [number, number];
  frames: number;
  neural: boolean;
  // Constant by construction: the marker translates at a fixed step in view
  // space at a fixed depth.
  pixelsPerRealFrame: number;
  // Where the scene's own transforms put the marker in the two real frames,
  // independent of anything that was rendered.
  expected: { previous: number; current: number };
  previous: MarkerProfile;
  current: MarkerProfile;
  interpolated: MarkerProfile;
  midpoint: number;
  error: number;
  tolerance: number;
}

export interface MidpointOptions {
  size: readonly [number, number];
  frames?: number;
  neuralWeights?: NeuralNetworkWeights;
}

// Runs the marker scene through FrameGenerator for long enough to settle, then
// reads back the last two real frames and the interpolated frame between them.
export async function measureMidpointTiming(
  device: GPUDevice,
  options: MidpointOptions,
): Promise<MidpointMeasurement> {
  const { size, neuralWeights } = options;
  const frames = options.frames ?? SETTLED_FRAMES;
  const projection = projectionFor(size);
  const gbuffer = new GBuffer(device, size, projection);
  const generator = new FrameGenerator({ device });

  if (neuralWeights) {
    generator.installNeuralBlendWeight(neuralWeights);
  }
  generator.configure({ ...PROJECTION, renderWidth: size[0], renderHeight: size[1] });

  let interpolated: GPUTexture | null = null;
  for (let frame = 0; frame < frames; frame++) {
    gbuffer.renderFrame(markerAt(projection, size, frame));
    // Frame 0 has no predecessor to interpolate from.
    if (frame === 0) {
      continue;
    }

    generator.prepare(gbuffer.frameInputs());
    interpolated = generator.dispatch();
  }

  if (!interpolated) {
    throw new Error(`midpoint timing needs at least two frames, got ${frames}`);
  }

  const previous = measureMarker(await readColor(device, gbuffer.previousColor), size);
  const current = measureMarker(await readColor(device, gbuffer.currentColor), size);
  const measured = measureMarker(await readColor(device, interpolated), size);

  generator.destroy();
  gbuffer.destroy();

  const midpoint = (previous.centroid + current.centroid) / 2;

  return {
    size,
    frames,
    neural: Boolean(neuralWeights),
    pixelsPerRealFrame: markerGeometry(projection, size).pixelsPerFrame,
    expected: {
      previous: markerPixelX(projection, size, frames - 2),
      current: markerPixelX(projection, size, frames - 1),
    },
    previous,
    current,
    interpolated: measured,
    midpoint,
    error: Math.abs(measured.centroid - midpoint),
    tolerance: MIDPOINT_TOLERANCE_PIXELS,
  };
}

export function formatMeasurement(m: MidpointMeasurement): string {
  const px = (value: number) => value.toFixed(2).padStart(8);

  return [
    `${m.size[0]}x${m.size[1]}, ${m.frames} real frames, neural blend weight ${m.neural ? "on" : "off"}`,
    `marker velocity          ${px(m.pixelsPerRealFrame)} px / real frame`,
    `real frame n-1 centroid  ${px(m.previous.centroid)} px   (scene says ${m.expected.previous.toFixed(2)})`,
    `real frame n   centroid  ${px(m.current.centroid)} px   (scene says ${m.expected.current.toFixed(2)})`,
    `temporal midpoint        ${px(m.midpoint)} px`,
    `interpolated centroid    ${px(m.interpolated.centroid)} px`,
    `midpoint error           ${px(m.error)} px   (tolerance ${m.tolerance})`,
    `bar width above half     ${m.previous.span} / ${m.interpolated.span} / ${m.current.span} px  (real / interpolated / real)`,
    `separate bars            ${m.previous.bars} / ${m.interpolated.bars} / ${m.current.bars}      (2 in the middle would be a crossfade)`,
    `red outside the bar      ${(m.interpolated.ghost * 100).toFixed(1)}%      (the blend's trail behind a fast edge)`,
  ].join("\n");
}
