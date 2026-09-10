// The quantitative check the demo exists to make: a bar crossing the frame at
// constant velocity, and the interpolated frame's copy of it measured against
// the midpoint of the two real frames it was generated from.
//
// Seven things have to hold, and only together do they mean anything:
//
//   across    the bar's centroid sits halfway between the two real frames'
//   shape     it is one bar, not two, and the same width
//   down      it sits on the same row, at the same height
//   mass      it carries the same red
//   interior  its pixels match the real frames', box-aligned
//
// Each covers a failure the others are blind to. The bar is narrower than its
// per-frame travel, so a crossfade of the two real frames leaves two bars with
// a gap — and the same centroid as the right answer, which is why counting the
// bars is not decoration. A column profile says nothing at all about the other
// axis, so without the row figures a bar slid a hundred pixels down, or reduced
// to a single row, measures identically to a correct one. `ghost` is a ratio,
// so a uniformly dimmer frame leaves it unchanged; `mass` is what sees that.
//
// And every one of those is a sum along an axis, which cannot see structure that
// cancels: a checkerboard of plus and minus a quarter over the bar leaves every
// row total and every column total bit-identical. The last check is the only one
// that looks inside the bar, and being box-aligned it is deliberately blind to
// translation — so it adds to the centroids rather than replacing them.

import {
  FrameGenerator,
  INTERPOLATED_TEXTURE_FORMAT,
  type NeuralNetworkWeights,
} from "../src/index.js";
import { GBuffer } from "./gbuffer.js";
import { markerAt, markerGeometry, markerPixelX, PROJECTION, projectionFor } from "./scene.js";

// Past both of the module's warmups: the scene-change detector forces a cut for
// its first six frames, and the preliminary blend trusts the game motion
// vectors unconditionally for ten frames after that. Only beyond both is the
// whole graph in play.
const SETTLED_FRAMES = 24;

// The centroid is a subpixel measure over hundreds of rows, so a pixel is a
// loose bound on a correct warp and far inside the 16px error a whole-frame
// mistiming would produce at this velocity.
const MIDPOINT_TOLERANCE_PIXELS = 1;

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
  // decodeHalf and BYTES_PER_HALF4 only describe this one format. The demo's
  // own colour targets are in it deliberately, so one decoder serves both them
  // and the frame the library hands back — which means a change to the
  // library's output format has to fail here rather than quietly decode the
  // bytes as something they are not.
  if (texture.format !== INTERPOLATED_TEXTURE_FORMAT) {
    throw new Error(
      `readColor decodes ${INTERPOLATED_TEXTURE_FORMAT}, got a ${texture.format} texture`,
    );
  }

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
  // Weighted mean column of the bar, in pixels.
  centroid: number;
  // Columns carrying more than half the peak column's weight: the bar's width.
  span: number;
  // How many separate bars those columns form. Two means a crossfade.
  bars: number;
  // The same two figures down the other axis. A column profile alone says
  // nothing about where the bar sits vertically or how much of its height
  // survived, so a frame with the bar slid down or reduced to one row measures
  // identically to a correct one.
  rowCentroid: number;
  rowSpan: number;
  // Total red in the bar, unnormalised. `ghost` is a ratio and cannot see a
  // frame that is uniformly dimmer; this can.
  mass: number;
  // The bar's own pixels, cropped to its bounding box. Everything above is a
  // sum along one axis or the other, and a sum cannot see structure that
  // cancels: a checkerboard of plus and minus a quarter leaves every row total
  // and every column total exactly as they were. Keeping the pixels is the only
  // thing that catches it.
  bar: MarkerBox;
  // Fraction of the frame's red lying outside the bar. The blend leaves a
  // partial trail behind a fast edge, and averaging it into the centroid would
  // report that ghost as a timing error, so it is measured rather than mixed in.
  ghost: number;
}

interface AxisProfile {
  centroid: number;
  span: number;
  bars: number;
  weight: number;
  // The measured window, one cell wider than the bar at each end.
  from: number;
  to: number;
}

// Where the marker sits along one axis, from that axis's summed profile.
//
// The bar is the widest run above half the peak. Strictly above: a warp landing
// exactly on a half pixel would split its two end columns to half weight each
// and measure one column narrow, which the current whole-pixel-per-frame
// velocities never produce but a fractional one could.
function measureAxis(values: Float32Array): AxisProfile {
  let total = 0;
  let peak = 0;
  for (const value of values) {
    total += value;
    peak = Math.max(peak, value);
  }

  const runs: { from: number; to: number }[] = [];
  let start = -1;
  for (let at = 0; at <= values.length; at++) {
    const above = at < values.length && values[at] > peak / 2;
    if (above) {
      start = start < 0 ? at : start;
      continue;
    }

    if (start < 0) {
      continue;
    }

    runs.push({ from: start, to: at - 1 });
    start = -1;
  }

  const bar = runs.reduce((widest, run) => (run.to - run.from > widest.to - widest.from ? run : widest), {
    from: 0,
    to: -1,
  });
  // One cell either side. The marker is a thin box seen from off-centre, so the
  // camera catches a dim sliver of its side face in the column just beyond the
  // bar: below the half-peak threshold, but still the marker. Including it is
  // what makes the measured position agree with the scene's own transforms
  // rather than sit a fifth of a pixel out, and the same rule runs over all
  // three frames.
  const from = Math.max(0, bar.from - 1);
  const to = Math.min(values.length - 1, bar.to + 1);

  let weight = 0;
  let moment = 0;
  for (let at = from; at <= to; at++) {
    weight += values[at];
    // Pixel centres, so the centroid is in the same continuous coordinates
    // projectToPixel reports rather than half a pixel to the left of them.
    moment += values[at] * (at + 0.5);
  }

  return {
    centroid: weight > 0 ? moment / weight : Number.NaN,
    span: bar.to - bar.from + 1,
    bars: runs.length,
    weight: total > 0 ? weight : 0,
    from,
    to,
  };
}

export interface MarkerBox {
  width: number;
  height: number;
  pixels: Float32Array;
}

// Cropped to the bar's own bounding box rather than to a fixed region, so the
// three frames' boxes line up with each other however far the bar has moved.
// That makes this comparison blind to translation by construction, which is why
// the centroids above are still what places the bar.
function crop(
  red: Float32Array,
  width: number,
  across: AxisProfile,
  down: AxisProfile,
): MarkerBox {
  const box = { width: across.to - across.from + 1, height: down.to - down.from + 1 };
  const pixels = new Float32Array(box.width * box.height);
  for (let y = 0; y < box.height; y++) {
    for (let x = 0; x < box.width; x++) {
      pixels[y * box.width + x] = red[(down.from + y) * width + across.from + x];
    }
  }

  return { ...box, pixels };
}

// The marker is the only red thing in the scene and the backdrop is blue-grey,
// so red beyond both other channels isolates it exactly rather than by a
// threshold, and survives the blend attenuating it.
export function measureMarker(
  texels: Float32Array,
  size: readonly [number, number],
): MarkerProfile {
  const [width, height] = size;
  const red = new Float32Array(width * height);
  const columns = new Float32Array(width);
  const rows = new Float32Array(height);
  let total = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * CHANNELS;
      const value = Math.max(0, texels[at] - Math.max(texels[at + 1], texels[at + 2]));
      red[y * width + x] = value;
      columns[x] += value;
      rows[y] += value;
      total += value;
    }
  }

  const across = measureAxis(columns);
  const down = measureAxis(rows);

  return {
    centroid: across.centroid,
    span: across.span,
    bars: across.bars,
    rowCentroid: down.centroid,
    rowSpan: down.span,
    mass: across.weight,
    ghost: total > 0 ? (total - across.weight) / total : 0,
    bar: crop(red, width, across, down),
  };
}

// A width or height is allowed to differ by one cell: the half-peak threshold
// lands on a pixel boundary, and the warp does not have to land on the same one
// the real frames did.
const SPAN_TOLERANCE_PIXELS = 1;
// The warped bar carries the same red as the real ones — measured, 0.04% less.
// Five percent leaves room for a driver rasterising an edge differently while
// staying nowhere near what losing rows or dimming the frame would cost.
const MASS_TOLERANCE_FRACTION = 0.05;
// The real pipeline measures 0.04% here, and the smallest corruption that gets
// past every other check — a patch cut out of the bar small enough to stay
// inside the mass tolerance — measures 4.5%. Two percent sits fifty times above
// the one and twice below the other.
const BAR_TOLERANCE_FRACTION = 0.02;

// How far the interpolated bar's pixels sit from the real frames', box-aligned,
// as a fraction of the red they contain.
//
// The marker moves a whole number of pixels per real frame, so the frame between
// two of them is a whole-pixel shift and the boxes line up exactly. A fractional
// velocity would resample the bar and this would have to compare against a
// resampled reference instead.
function barMismatch(previous: MarkerBox, current: MarkerBox, interpolated: MarkerBox): number {
  const sameSize = (box: MarkerBox) =>
    box.width === interpolated.width && box.height === interpolated.height;
  if (!sameSize(previous) || !sameSize(current)) {
    // The span checks already say so, and there is nothing to line up against.
    return Number.POSITIVE_INFINITY;
  }

  let difference = 0;
  let mass = 0;
  for (let at = 0; at < interpolated.pixels.length; at++) {
    const expected = (previous.pixels[at] + current.pixels[at]) / 2;
    difference += Math.abs(interpolated.pixels[at] - expected);
    mass += expected;
  }

  return mass > 0 ? difference / mass : Number.POSITIVE_INFINITY;
}

// Everything about the interpolated frame this measurement can see going wrong,
// as sentences rather than a boolean — a caller that only knows "it failed"
// cannot tell a mistimed frame from one missing half its rows.
export function midpointFailures(
  previous: MarkerProfile,
  current: MarkerProfile,
  interpolated: MarkerProfile,
): string[] {
  const failures: string[] = [];
  const mean = (from: number, to: number) => (from + to) / 2;
  const report = (wrong: boolean, message: string) => {
    if (wrong) {
      failures.push(message);
    }
  };

  const midpoint = mean(previous.centroid, current.centroid);
  report(
    Math.abs(interpolated.centroid - midpoint) > MIDPOINT_TOLERANCE_PIXELS,
    `bar at ${interpolated.centroid.toFixed(2)}px, not the ${midpoint.toFixed(2)}px midpoint of ` +
      `${previous.centroid.toFixed(2)} and ${current.centroid.toFixed(2)}`,
  );
  report(interpolated.bars !== 1, `${interpolated.bars} bars, not one — a crossfade, not a warp`);

  const width = mean(previous.span, current.span);
  report(
    Math.abs(interpolated.span - width) > SPAN_TOLERANCE_PIXELS,
    `bar ${interpolated.span}px wide, against ${width}px in the real frames`,
  );

  // The marker only ever moves horizontally, so both real frames agree here and
  // so must the frame between them.
  const row = mean(previous.rowCentroid, current.rowCentroid);
  report(
    Math.abs(interpolated.rowCentroid - row) > MIDPOINT_TOLERANCE_PIXELS,
    `bar centred on row ${interpolated.rowCentroid.toFixed(2)}, against ${row.toFixed(2)} in the real frames`,
  );

  const height = mean(previous.rowSpan, current.rowSpan);
  report(
    Math.abs(interpolated.rowSpan - height) > SPAN_TOLERANCE_PIXELS,
    `bar ${interpolated.rowSpan}px tall, against ${height}px in the real frames`,
  );

  const mass = mean(previous.mass, current.mass);
  report(
    Math.abs(interpolated.mass - mass) > MASS_TOLERANCE_FRACTION * mass,
    `bar carries ${(interpolated.mass / mass).toFixed(3)} of the real frames' red`,
  );

  // Last, because it is the only check that looks at the bar's interior rather
  // than at a total: a hole punched inside it, or a pattern that sums to
  // nothing along both axes, changes no figure above.
  const mismatch = barMismatch(previous.bar, current.bar, interpolated.bar);
  report(
    mismatch > BAR_TOLERANCE_FRACTION,
    `bar's pixels differ from the real frames' by ${(mismatch * 100).toFixed(1)}% of their red`,
  );

  return failures;
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
  // The bar's interior against the real frames', box-aligned.
  mismatch: number;
  // Empty when the interpolated frame passed every check.
  failures: string[];
}

export interface MidpointOptions {
  size: readonly [number, number];
  neuralWeights?: NeuralNetworkWeights;
}

// Runs the marker scene through FrameGenerator for long enough to settle, then
// reads back the last two real frames and the interpolated frame between them.
export async function measureMidpointTiming(
  device: GPUDevice,
  options: MidpointOptions,
): Promise<MidpointMeasurement> {
  const { size, neuralWeights } = options;
  const projection = projectionFor(size);
  const gbuffer = new GBuffer(device, size, projection);
  const generator = new FrameGenerator({ device });

  if (neuralWeights) {
    generator.installNeuralBlendWeight(neuralWeights);
  }
  generator.configure({ ...PROJECTION, renderWidth: size[0], renderHeight: size[1] });

  // Written by every frame but the first, and there are SETTLED_FRAMES of them.
  let interpolated!: GPUTexture;
  for (let frame = 0; frame < SETTLED_FRAMES; frame++) {
    gbuffer.renderFrame(markerAt(projection, size, frame));
    // Frame 0 has no predecessor to interpolate from.
    if (frame === 0) {
      continue;
    }

    generator.prepare(gbuffer.frameInputs());
    interpolated = generator.dispatch();
  }

  const previous = measureMarker(await readColor(device, gbuffer.previousColor), size);
  const current = measureMarker(await readColor(device, gbuffer.currentColor), size);
  const measured = measureMarker(await readColor(device, interpolated), size);

  generator.destroy();
  gbuffer.destroy();

  const midpoint = (previous.centroid + current.centroid) / 2;

  return {
    size,
    frames: SETTLED_FRAMES,
    neural: Boolean(neuralWeights),
    pixelsPerRealFrame: markerGeometry(projection, size).pixelsPerFrame,
    expected: {
      previous: markerPixelX(projection, size, SETTLED_FRAMES - 2),
      current: markerPixelX(projection, size, SETTLED_FRAMES - 1),
    },
    previous,
    current,
    interpolated: measured,
    midpoint,
    error: Math.abs(measured.centroid - midpoint),
    tolerance: MIDPOINT_TOLERANCE_PIXELS,
    mismatch: barMismatch(previous.bar, current.bar, measured.bar),
    failures: midpointFailures(previous, current, measured),
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
    `bar height above half    ${m.previous.rowSpan} / ${m.interpolated.rowSpan} / ${m.current.rowSpan} px`,
    `bar row centre           ${px(m.previous.rowCentroid)} / ${m.interpolated.rowCentroid.toFixed(2)} / ${m.current.rowCentroid.toFixed(2)}   (the marker never moves vertically)`,
    `red in the bar           ${px(m.previous.mass)} / ${m.interpolated.mass.toFixed(2)} / ${m.current.mass.toFixed(2)}`,
    `separate bars            ${m.previous.bars} / ${m.interpolated.bars} / ${m.current.bars}      (2 in the middle would be a crossfade)`,
    `red outside the bar      ${(m.interpolated.ghost * 100).toFixed(1)}%      (the blend's trail behind a fast edge)`,
    `bar pixels vs real       ${(m.mismatch * 100).toFixed(2)}%      (structure no row or column sum can see)`,
    m.failures.length === 0
      ? "all checks passed"
      : `FAILED: ${m.failures.join("; ")}`,
  ].join("\n");
}
