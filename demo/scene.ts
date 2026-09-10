// The demo's geometry and its two scenes: the one you look at, and the
// deterministic one the midpoint-timing check measures. Everything is stated
// per real frame — a frame index, not a wall-clock time — so the same scene
// replays identically under vitest and in the browser.

import {
  compose,
  depthOf,
  projectToPixel,
  viewUnitsPerPixel,
  type Mat4,
  type Projection,
} from "./math.js";

export interface Instance {
  model: Mat4;
  // The same body one real frame earlier. Differencing the two projections is
  // where the motion vectors come from: the scene's own transforms, not an
  // estimate recovered from the rendered image.
  previousModel: Mat4;
  color: readonly [number, number, number];
  // Cells of procedural detail across one unit of object space. Optical flow
  // block matching needs texture to lock onto, and the background needs far
  // more of it per unit than a one-unit cube does.
  patternScale: number;
}

export const PROJECTION = {
  verticalFovRadians: Math.PI / 3,
  nearPlane: 0.5,
  farPlane: 60,
} as const;

export function projectionFor(viewport: readonly [number, number]): Projection {
  return { ...PROJECTION, aspect: viewport[0] / viewport[1] };
}

const BACKGROUND_Z = -18;
const BACKGROUND_COLOR = [0.30, 0.34, 0.40] as const;
// ~4px cells at 1280 wide, small enough that an 8x8 optical flow block sees
// several of them and matches in exactly one place.
const BACKGROUND_PATTERN_SCALE = 300;
// The frame's own extent at the background's depth, with margin so no clear
// colour shows through at the edges. A pixel with no geometry would read back
// as the near plane and confuse the disocclusion mask.
const BACKGROUND_OVERSCAN = 1.1;

// A static backdrop, sized to cover the frame. Motion vectors are zero here, so
// it is also the region that proves the pipeline is not simply crossfading.
function background(projection: Projection): Instance {
  const halfHeight = Math.tan(projection.verticalFovRadians / 2) * depthOf(BACKGROUND_Z);
  const scale = [
    2 * halfHeight * projection.aspect * BACKGROUND_OVERSCAN,
    2 * halfHeight * BACKGROUND_OVERSCAN,
    0.2,
  ] as const;
  const model = compose([0, 0, BACKGROUND_Z], 0, 0, scale);

  return {
    model,
    previousModel: model,
    color: BACKGROUND_COLOR,
    patternScale: BACKGROUND_PATTERN_SCALE,
  };
}

interface Body {
  color: readonly [number, number, number];
  size: number;
  depth: number;
  orbitRadius: number;
  orbitSpeed: number;
  phase: number;
  spinX: number;
  spinY: number;
}

// Deliberately modest in screen coverage: a large fast-moving bright object
// shifts the luminance histogram enough for the optical flow's scene-change
// detector to call a cut, and a cut makes the interpolated frame a copy of the
// current one — a toggle that looks like a no-op.
const BODIES: readonly Body[] = [
  { color: [0.90, 0.35, 0.20], size: 1.4, depth: 7.0, orbitRadius: 3.4, orbitSpeed: 0.055, phase: 0.0, spinX: 0.021, spinY: 0.034 },
  { color: [0.25, 0.70, 0.90], size: 1.1, depth: 9.5, orbitRadius: 4.6, orbitSpeed: -0.038, phase: 2.1, spinX: -0.030, spinY: 0.019 },
  { color: [0.95, 0.80, 0.25], size: 0.9, depth: 5.5, orbitRadius: 2.2, orbitSpeed: 0.081, phase: 4.0, spinX: 0.017, spinY: -0.045 },
  { color: [0.55, 0.90, 0.45], size: 1.2, depth: 12.0, orbitRadius: 6.0, orbitSpeed: 0.029, phase: 1.2, spinX: 0.026, spinY: 0.022 },
];

const BODY_PATTERN_SCALE = 7;
// Vertical travel is flatter than horizontal, so the bodies sweep rather than
// circle and spend more of the loop crossing the frame.
const ORBIT_FLATTEN = 0.42;

function bodyModel(body: Body, frame: number): Mat4 {
  const angle = body.phase + body.orbitSpeed * frame;
  const position = [
    Math.cos(angle) * body.orbitRadius,
    Math.sin(angle) * body.orbitRadius * ORBIT_FLATTEN,
    -body.depth,
  ] as const;

  return compose(position, body.spinX * frame, body.spinY * frame, [body.size, body.size, body.size]);
}

// What the page renders: a handful of bodies orbiting and spinning in front of
// a static backdrop.
export function demoScene(projection: Projection, frame: number): Instance[] {
  const instances = [background(projection)];
  for (const body of BODIES) {
    instances.push({
      model: bodyModel(body, frame),
      previousModel: bodyModel(body, frame - 1),
      color: body.color,
      patternScale: BODY_PATTERN_SCALE,
    });
  }

  return instances;
}

// Marker constants are fractions of the frame, so the pixel figures the
// midpoint check reports mean the same thing at any resolution.
//
// The width is deliberately smaller than the per-frame travel: a crossfade of
// the two real frames would then show two bars with a gap between them, where a
// correctly warped frame shows one bar at the midpoint. That gap is what makes
// the check discriminating rather than a shape a blur would also satisfy.
const MARKER_SPEED_FRACTION = 0.025;
const MARKER_WIDTH_FRACTION = 0.0125;
const MARKER_START_FRACTION = 0.12;
const MARKER_HEIGHT_FRACTION = 0.55;
const MARKER_Z = -7;
// Pure red against a blue-grey backdrop, so `red - max(green, blue)` isolates
// the marker from the background exactly rather than by a threshold.
const MARKER_COLOR = [1, 0, 0] as const;
// Deliberately flat. Procedural detail across the bar would weight its centroid
// towards the brighter side, and a measurement that has to be corrected for a
// bias of its own is worth less than one that does not. The backdrop is what
// gives the optical flow something to lock onto.
const MARKER_PATTERN_SCALE = 0;
// Thin, and translated so the face the camera sees sits exactly at MARKER_Z.
// The camera sees the front face, not the box's centre, and off to the side of
// the frame those two project a visible distance apart.
const MARKER_THICKNESS = 0.02;

export interface MarkerGeometry {
  startX: number;
  // View-space x added per real frame. Constant, and the depth is constant, so
  // the projected pixel velocity is constant too — the premise the midpoint
  // check rests on.
  stepX: number;
  scale: readonly [number, number, number];
  pixelsPerFrame: number;
}

export function markerGeometry(
  projection: Projection,
  viewport: readonly [number, number],
): MarkerGeometry {
  const unitsPerPixel = viewUnitsPerPixel(projection, depthOf(MARKER_Z), viewport[0]);
  const pixelsPerFrame = MARKER_SPEED_FRACTION * viewport[0];
  const widthPixels = MARKER_WIDTH_FRACTION * viewport[0];

  return {
    startX: (MARKER_START_FRACTION - 0.5) * viewport[0] * unitsPerPixel,
    stepX: pixelsPerFrame * unitsPerPixel,
    scale: [
      widthPixels * unitsPerPixel,
      MARKER_HEIGHT_FRACTION * viewport[1] * unitsPerPixel,
      MARKER_THICKNESS,
    ],
    pixelsPerFrame,
  };
}

// The centre of the visible face, which is what the midpoint check measures.
export function markerPosition(
  geometry: MarkerGeometry,
  frame: number,
): readonly [number, number, number] {
  return [geometry.startX + geometry.stepX * frame, 0, MARKER_Z];
}

function markerModel(geometry: MarkerGeometry, frame: number): Mat4 {
  const [x, y, z] = markerPosition(geometry, frame);

  return compose([x, y, z - MARKER_THICKNESS / 2], 0, 0, geometry.scale);
}

// Where the marker's centre lands in the rendered frame, straight from the
// scene's transforms. The midpoint check compares its measured centroid
// against this.
export function markerPixelX(
  projection: Projection,
  viewport: readonly [number, number],
  frame: number,
): number {
  const geometry = markerGeometry(projection, viewport);

  return projectToPixel(projection, markerPosition(geometry, frame), viewport)[0];
}

// One bar translating at constant velocity across the static backdrop. Nothing
// else moves, so any displacement in the interpolated frame is the marker's.
export function markerAt(
  projection: Projection,
  viewport: readonly [number, number],
  frame: number,
): Instance[] {
  const geometry = markerGeometry(projection, viewport);

  return [
    background(projection),
    {
      model: markerModel(geometry, frame),
      previousModel: markerModel(geometry, frame - 1),
      color: MARKER_COLOR,
      patternScale: MARKER_PATTERN_SCALE,
    },
  ];
}

// Unit cube, 36 vertices of position and normal, no index buffer. Small enough
// that the indirection would cost more to read than it saves.
const FACES: readonly { normal: readonly [number, number, number]; u: readonly [number, number, number]; v: readonly [number, number, number] }[] = [
  { normal: [1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
  { normal: [-1, 0, 0], u: [0, 1, 0], v: [0, 0, 1] },
  { normal: [0, 1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { normal: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { normal: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
  { normal: [0, 0, -1], u: [1, 0, 0], v: [0, 1, 0] },
];

export const CUBE_FLOATS_PER_VERTEX = 6;
export const CUBE_VERTEX_COUNT = FACES.length * 6;

export function cubeVertices(): Float32Array<ArrayBuffer> {
  const data = new Float32Array(CUBE_VERTEX_COUNT * CUBE_FLOATS_PER_VERTEX);
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, -1],
    [1, 1],
    [-1, 1],
  ] as const;

  let at = 0;
  for (const face of FACES) {
    for (const [du, dv] of corners) {
      for (let axis = 0; axis < 3; axis++) {
        data[at++] = (face.normal[axis] + du * face.u[axis] + dv * face.v[axis]) / 2;
      }
      data.set(face.normal, at);
      at += 3;
    }
  }

  return data;
}

export const FLOATS_PER_INSTANCE = 36;

export function packInstances(instances: readonly Instance[]): Float32Array<ArrayBuffer> {
  const data = new Float32Array(instances.length * FLOATS_PER_INSTANCE);
  instances.forEach((instance, index) => {
    const at = index * FLOATS_PER_INSTANCE;
    data.set(instance.model, at);
    data.set(instance.previousModel, at + 16);
    data.set(instance.color, at + 32);
    data[at + 35] = instance.patternScale;
  });

  return data;
}
