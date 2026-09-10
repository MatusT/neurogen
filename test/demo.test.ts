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
  type MidpointMeasurement,
} from "../demo/verify.js";

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
  // Three columns of pure red on a grey field, so the answer is arithmetic
  // rather than a rendering.
  function field(columns: readonly number[]): Float32Array {
    const [width, height] = [16, 4];
    const texels = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const red = columns.includes(x) ? 1 : 0.3;
        texels.set([red, 0.3, 0.3, 1], (y * width + x) * 4);
      }
    }

    return texels;
  }

  it("finds the centroid and width of one bar", () => {
    const profile = measureMarker(field([6, 7, 8]), [16, 4]);

    expect(profile.centroid).toBeCloseTo(7.5, 6);
    expect([profile.span, profile.bars, profile.ghost]).toEqual([3, 1, 0]);
  });

  it("reports the two bars a crossfade would leave", () => {
    // Averaging the whole frame's red would put this at 7.5 — the same answer
    // as the single bar above, and the reason the midpoint position on its own
    // cannot tell a warp from a crossfade.
    const profile = measureMarker(field([4, 10]), [16, 4]);

    expect(profile.bars).toBe(2);
    expect(profile.centroid).toBeCloseTo(4.5, 6);
    expect(profile.ghost).toBeCloseTo(0.5, 6);
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
    expect(formatMeasurement(measurement)).toBeTypeOf("string");
    expect(measurement.error).toBeLessThan(measurement.tolerance);
  });

  it("emits one warped bar rather than a crossfade of two", () => {
    // The bar is narrower than its per-frame travel, so a crossfade separates
    // into two bars whose midpoint is the same — which is why the position
    // check above needs this one beside it.
    expect(measurement.interpolated.bars).toBe(1);
    expect(measurement.interpolated.span).toBe(measurement.current.span);
    // The blend leaves a partial trail behind the bar, but the bar is still
    // where nearly all of the marker is.
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
