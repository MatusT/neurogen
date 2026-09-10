// The demo is an application, not a library module, so what is covered here is
// only what a browser cannot check for you: the projection and motion-vector
// geometry it feeds the library, and the midpoint-timing measurement that is
// the demo's actual claim about frame generation.
//
// Imported before the demo modules: the harness puts WebGPU's runtime globals
// on globalThis, and the demo is written against the browser ambients.
import { beforeAll, describe, expect, it } from "vitest";
import { createGpuHarness, type GpuHarness } from "./gpu-harness.js";
import { perspective, projectToPixel, viewUnitsPerPixel, type Projection } from "../demo/math.js";
import {
  CUBE_FLOATS_PER_VERTEX,
  CUBE_VERTEX_COUNT,
  cubeVertices,
  markerGeometry,
  markerPixelX,
  projectionFor,
} from "../demo/scene.js";
import {
  formatMeasurement,
  measureMarker,
  measureMidpointTiming,
  midpointFailures,
  type MidpointMeasurement,
} from "../demo/verify.js";
import { enabledFrom, FrameGen, shown, Shown } from "../demo/schedule.js";

const VIEWPORT: readonly [number, number] = [1280, 720];

function projection(): Projection {
  return projectionFor(VIEWPORT);
}

describe("demo geometry", () => {
  it("projects the near and far planes onto the ends of WebGPU's depth range", () => {
    // The library's disocclusion mask converts device depth back to metres with
    // the planes handed to configure(), so the demo's projection has to be the
    // standard non-inverted one those planes describe.
    const p = projection();
    const m = perspective(p);
    const deviceDepth = (viewZ: number) => (m[10] * viewZ + m[14]) / -viewZ;

    expect(deviceDepth(-p.nearPlane)).toBeCloseTo(0, 5);
    expect(deviceDepth(-p.farPlane)).toBeCloseTo(1, 5);
  });

  it("converts pixels to view-space units the projection agrees with", () => {
    // The marker's velocity is stated in pixels and applied in view space; this
    // is the conversion that has to hold for the midpoint tolerance to mean
    // pixels at all.
    const p = projection();
    const depth = 7;
    const offset = 40;
    const unit = viewUnitsPerPixel(p, depth, VIEWPORT[0]);

    const centre = projectToPixel(p, [0, 0, -depth], VIEWPORT);
    const moved = projectToPixel(p, [offset * unit, 0, -depth], VIEWPORT);

    expect(moved[0] - centre[0]).toBeCloseTo(offset, 3);
    expect(moved[1]).toBeCloseTo(centre[1], 6);
  });

  it("moves the marker at a constant pixel velocity", () => {
    const p = projection();
    const { pixelsPerFrame } = markerGeometry(p, VIEWPORT);

    for (let frame = 1; frame < 30; frame++) {
      const step = markerPixelX(p, VIEWPORT, frame) - markerPixelX(p, VIEWPORT, frame - 1);
      expect([frame, step]).toEqual([frame, expect.closeTo(pixelsPerFrame, 3)]);
    }
  });

  it("builds a unit cube of six axis-aligned faces", () => {
    const vertices = cubeVertices();
    expect(vertices.length).toBe(CUBE_VERTEX_COUNT * CUBE_FLOATS_PER_VERTEX);

    const normals = new Set<string>();
    for (let vertex = 0; vertex < CUBE_VERTEX_COUNT; vertex++) {
      const at = vertex * CUBE_FLOATS_PER_VERTEX;
      const position = [...vertices.slice(at, at + 3)];
      const normal = [...vertices.slice(at + 3, at + 6)];

      expect([vertex, position.every((c) => Math.abs(c) === 0.5)]).toEqual([vertex, true]);
      expect([vertex, normal.reduce((sum, c) => sum + Math.abs(c), 0)]).toEqual([vertex, 1]);
      // The face's own axis is the one the corner offsets never move.
      const axis = normal.findIndex((c) => c !== 0);
      expect([vertex, position[axis]]).toEqual([vertex, normal[axis] / 2]);
      normals.add(normal.join(","));
    }

    expect(normals.size).toBe(6);
  });
});

describe("marker measurement", () => {
  const SIZE: readonly [number, number] = [128, 96];
  const BAR = { width: 8, top: 20, height: 50 };

  // A red bar on a grey field: the arrangement measureMarker is built to read,
  // with the answer known by construction rather than by rendering.
  function frame(bar: { left: number; top?: number; height?: number; red?: number }): Float32Array {
    const [width, height] = SIZE;
    const texels = new Float32Array(width * height * 4);
    const top = bar.top ?? BAR.top;
    const tall = bar.height ?? BAR.height;
    const red = bar.red ?? 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const inside = x >= bar.left && x < bar.left + BAR.width && y >= top && y < top + tall;
        texels.set(inside ? [red, 0, 0, 1] : [0.3, 0.34, 0.4, 1], (y * width + x) * 4);
      }
    }

    return texels;
  }

  const profile = (bar: Parameters<typeof frame>[0]) => measureMarker(frame(bar), SIZE);
  // Two real frames sixteen pixels apart, so the midpoint bar sits at 48.
  const previous = () => profile({ left: 40 });
  const current = () => profile({ left: 56 });

  it("finds the centroid and extent of one bar", () => {
    const bar = profile({ left: 48 });

    expect([bar.centroid, bar.span, bar.bars, bar.ghost]).toEqual([52, 8, 1, 0]);
    expect([bar.rowCentroid, bar.rowSpan]).toEqual([45, 50]);
  });

  it("reports the two bars a crossfade would leave", () => {
    // Averaging the whole frame's red would put this at 52 — the same answer as
    // the single bar above, and the reason the midpoint position on its own
    // cannot tell a warp from a crossfade.
    const split = new Float32Array(frame({ left: 40 }));
    const both = frame({ left: 56 });
    for (let i = 0; i < split.length; i += 4) {
      split[i] = Math.max(split[i], both[i]);
    }
    const bar = measureMarker(split, SIZE);

    expect(bar.bars).toBe(2);
    expect(midpointFailures(previous(), current(), bar).join(" ")).toMatch(/2 bars/);
  });

  // A column profile alone is blind to all three of these: each one measures
  // identically to a correct frame across the horizontal axis.
  it("rejects an interpolated frame the horizontal profile cannot see is wrong", () => {
    expect(midpointFailures(previous(), current(), profile({ left: 48 }))).toEqual([]);

    const shifted = midpointFailures(previous(), current(), profile({ left: 48, top: 50 }));
    expect(shifted.join(" ")).toMatch(/centred on row/);

    const oneRow = midpointFailures(previous(), current(), profile({ left: 48, height: 1 }));
    expect(oneRow.join(" ")).toMatch(/1px tall/);
    expect(oneRow.join(" ")).toMatch(/of the real frames' red/);

    const dimmed = midpointFailures(previous(), current(), profile({ left: 48, red: 0.001 }));
    expect(dimmed.join(" ")).toMatch(/of the real frames' red/);

    // Position, width and bar count all still pass on every one of them.
    for (const wrong of [shifted, oneRow, dimmed]) {
      expect(wrong.join(" ")).not.toMatch(/midpoint of|bars|wide/);
    }
  });

  it("rejects a mistimed bar", () => {
    // The failure the whole fixture exists to catch: the current frame handed
    // back unchanged.
    expect(midpointFailures(previous(), current(), current()).join(" ")).toMatch(/not the 52.00px midpoint/);
  });
});

describe("frame generation timing", () => {
  let harness: GpuHarness;
  let measurement: MidpointMeasurement;

  beforeAll(async () => {
    harness = await createGpuHarness();
    measurement = await measureMidpointTiming(harness.device, { size: VIEWPORT });
  }, 60_000);

  it("renders the marker where the scene's transforms put it", () => {
    // Ties the measurement to the scene rather than to itself: if the renderer
    // and the projection helpers disagreed, every figure below would be
    // internally consistent and wrong.
    expect(measurement.previous.centroid).toBeCloseTo(measurement.expected.previous, 0);
    expect(measurement.current.centroid).toBeCloseTo(measurement.expected.current, 0);
    expect(measurement.current.centroid - measurement.previous.centroid).toBeCloseTo(
      measurement.pixelsPerRealFrame,
      0,
    );
  });

  it("puts the interpolated marker at the temporal midpoint", () => {
    // Printed because the figures are the point: a pass says the error is under
    // a pixel, and the reviewer wants to know it is 0.00 and not 0.99.
    console.log("\n" + formatMeasurement(measurement));
    // Position across, one bar of the same width, the same rows, the same red —
    // the fixtures above are what establish that each of those can fail alone.
    expect(measurement.failures).toEqual([]);
    expect(measurement.error).toBeLessThan(measurement.tolerance);
  });

  it("leaves only a partial trail behind the moving bar", () => {
    // The blend's ghost is a quality figure rather than a correctness one, so
    // it is bounded rather than required to be zero — but it is zero in the
    // real frames, which is what says the bar window captures the whole marker.
    expect(measurement.previous.ghost).toBe(0);
    expect(measurement.interpolated.ghost).toBeLessThan(0.25);
  });

  it("does not simply hand back one of the two real frames", () => {
    const half = measurement.pixelsPerRealFrame / 2;
    expect(Math.abs(measurement.interpolated.centroid - measurement.current.centroid)).toBeGreaterThan(
      half / 2,
    );
    expect(Math.abs(measurement.interpolated.centroid - measurement.previous.centroid)).toBeGreaterThan(
      half / 2,
    );
  });
});

describe("presentation schedule", () => {
  const REFRESHES = 16;
  const NEVER = -1;

  // Displayed time in half real frames: real frame n is 2n, and the frame
  // interpolated between n-1 and n is 2n-1.
  function time(what: Shown, realFrame: number): number {
    if (what === Shown.RealCurrent) {
      return 2 * realFrame;
    }

    return what === Shown.RealPrevious ? 2 * (realFrame - 1) : 2 * realFrame - 1;
  }

  // The render loop's schedule with the GPU taken out, flipping the toggle
  // between refresh `flipAt - 1` and `flipAt` the way a click between two
  // refreshes does. `gate` is the rule under test.
  function run(from: FrameGen, flipAt: number, gate = enabledFrom): number[] {
    let mode = from;
    let realFrame = -1;
    let enabled = gate(0);
    const times: number[] = [];

    for (let refresh = 0; refresh < REFRESHES; refresh++) {
      if (refresh === flipAt) {
        mode = mode === FrameGen.On ? FrameGen.Off : FrameGen.On;
        enabled = gate(realFrame);
      }

      const rendered = refresh % 2 === 0;
      realFrame += rendered ? 1 : 0;
      times.push(time(shown({ mode, rendered, realFrame, enabledFrom: enabled }), realFrame));
    }

    return times;
  }

  const nonDecreasing = (times: number[]) => times.every((at, i) => i === 0 || at >= times[i - 1]);

  it("never steps displayed time backwards, whenever the toggle is flipped", () => {
    for (let flipAt = 0; flipAt < REFRESHES; flipAt++) {
      expect([flipAt, "off->on", nonDecreasing(run(FrameGen.Off, flipAt))]).toEqual([flipAt, "off->on", true]);
      expect([flipAt, "on->off", nonDecreasing(run(FrameGen.On, flipAt))]).toEqual([flipAt, "on->off", true]);
    }
  });

  it("would step backwards if frame generation engaged on the current real frame", () => {
    // Engaging it on the refresh right after real frame n was presented: the
    // frame between n-1 and n is older than what is already on screen. The gate
    // is what stops that, and this is the test above being able to see it.
    const engagedImmediately = () => 1;
    const times = run(FrameGen.Off, 3, engagedImmediately);

    expect(times.slice(2, 4)).toEqual([2, 1]);
    expect(nonDecreasing(times)).toBe(false);
  });

  it("advances half a real frame per refresh once engaged", () => {
    const settled = run(FrameGen.On, NEVER).slice(4);

    expect(settled.every((at, i) => i === 0 || at === settled[i - 1] + 1)).toBe(true);
  });

  it("holds each real frame for two refreshes when off", () => {
    expect(run(FrameGen.Off, NEVER).slice(0, 6)).toEqual([0, 0, 2, 2, 4, 4]);
  });
});
