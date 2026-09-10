// Column-major 4x4 matrices, matching WGSL's mat4x4<f32> memory layout, plus
// the projection helpers the marker's screen-space velocity is derived from.
// Nothing here touches the GPU or the DOM, so the demo's geometry claims are
// checkable in a plain unit test.

export type Mat4 = Float32Array<ArrayBuffer>;

export interface Projection {
  verticalFovRadians: number;
  aspect: number;
  nearPlane: number;
  farPlane: number;
}

const MAT4_ELEMENTS = 16;

// The camera sits at the view-space origin looking down -z, so a point's
// distance in front of it is -z. Kept as a named conversion because the sign
// trips up every reader once.
export function depthOf(viewSpaceZ: number): number {
  return -viewSpaceZ;
}

// WebGPU clip space: y up, z in [0, 1] with 0 at the near plane. That is the
// standard non-inverted convention FrameGenerator's disocclusion mask assumes,
// and it must match the nearPlane/farPlane handed to configure().
export function perspective(projection: Projection): Mat4 {
  const { verticalFovRadians, aspect, nearPlane, farPlane } = projection;
  const focal = 1 / Math.tan(verticalFovRadians / 2);
  const range = nearPlane - farPlane;

  const m = new Float32Array(MAT4_ELEMENTS);
  m[0] = focal / aspect;
  m[5] = focal;
  m[10] = farPlane / range;
  m[11] = -1;
  m[14] = (nearPlane * farPlane) / range;

  return m;
}

// Translation * rotateX * rotateY * scale, which is every transform the scene
// needs. A general matrix stack would be more code than the demo uses.
export function compose(
  translation: readonly [number, number, number],
  rotationX: number,
  rotationY: number,
  scale: readonly [number, number, number],
): Mat4 {
  const [sx, cx] = [Math.sin(rotationX), Math.cos(rotationX)];
  const [sy, cy] = [Math.sin(rotationY), Math.cos(rotationY)];

  const m = new Float32Array(MAT4_ELEMENTS);
  // Columns of rotateX * rotateY, each scaled by that axis's factor.
  m[0] = cy * scale[0];
  m[1] = sx * sy * scale[0];
  m[2] = -cx * sy * scale[0];

  m[4] = 0;
  m[5] = cx * scale[1];
  m[6] = sx * scale[1];

  m[8] = sy * scale[2];
  m[9] = -sx * cy * scale[2];
  m[10] = cx * cy * scale[2];

  m[12] = translation[0];
  m[13] = translation[1];
  m[14] = translation[2];
  m[15] = 1;

  return m;
}

// Where a view-space point lands in the render target, in pixels with y down —
// the convention the motion vectors and the centroid measurement both use.
// Goes through perspective() rather than repeating its algebra, so a bug in the
// matrix the GPU renders with shows up here too.
export function projectToPixel(
  projection: Projection,
  point: readonly [number, number, number],
  viewport: readonly [number, number],
): [number, number] {
  const m = perspective(projection);
  const [x, y, z] = point;
  const clipX = m[0] * x;
  const clipY = m[5] * y;
  const clipW = m[11] * z;

  const ndcX = clipX / clipW;
  const ndcY = clipY / clipW;

  return [(ndcX * 0.5 + 0.5) * viewport[0], (0.5 - ndcY * 0.5) * viewport[1]];
}

// One pixel of screen motion expressed in view-space units, for a point at
// `depth` in front of the camera. Lets the marker's velocity and width be
// stated as fractions of the frame and stay the same fraction at any
// resolution, which is what makes the midpoint tolerance a pixel count.
export function viewUnitsPerPixel(
  projection: Projection,
  depth: number,
  viewportWidth: number,
): number {
  const focal = 1 / Math.tan(projection.verticalFovRadians / 2);

  return (2 * projection.aspect * depth) / (viewportWidth * focal);
}
