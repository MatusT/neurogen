// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_reconstruct_dilated_velocity_and_previous_depth.h:
// dilate depth and motion vectors to the nearest surface in a 3x3 neighbourhood,
// then scatter the dilated depth into the previous frame to reconstruct what the
// previous frame's depth buffer held.
//
// In AMD's pipeline the upscaler produces the dilated buffers and this module
// only consumes them. There is no upscaler here, so this pass owns them. Every
// later frame-interpolation pass reads dilated depth and dilated motion vectors,
// never the raw inputs — dilating first is what stops a thin foreground edge
// from being reprojected with the background's motion.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution.

// @import ./common.wgsl

@group(0) @binding(1) var inputDepth: texture_2d<f32>;
// .xy in whole render-resolution pixels, current frame -> previous frame.
@group(0) @binding(2) var inputMotionVectors: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> dilatedDepth: array<f32>;
// UV-space, i.e. the input vectors divided by the render size. Converted once
// here so no later pass has to know the input's units.
@group(0) @binding(4) var<storage, read_write> dilatedMotionVectors: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> reconstructedDepthPrevious: array<atomic<u32>>;

// Below this the reprojection lands back on the source pixel anyway, and
// scattering it only smears the depth over that pixel's neighbours.
const MIN_REPROJECTION_PIXELS: f32 = 0.1;
// A tap this weak contributes nothing but would still claim the cell for its
// depth, since the scatter is a max and carries no weight.
const RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD: f32 = FI_EPSILON;

struct NearestDepth {
    depth: f32,
    coord: vec2<i32>,
}

// Standard depth, so nearest is smallest. AMD unrolls a fixed 9-entry offset
// table; the 3x3 loop is the same set.
fn findNearestDepth(pos: vec2<i32>) -> NearestDepth {
    var nearest: NearestDepth;
    nearest.coord = pos;
    nearest.depth = textureLoad(inputDepth, pos, 0).x;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let samplePos = pos + vec2<i32>(x, y);
            // An out-of-range textureLoad reads zero, which is the near plane
            // and would win every comparison — check before, not after.
            if (!fiInBounds(samplePos, params.renderSize)) { continue; }

            let depth = textureLoad(inputDepth, samplePos, 0).x;
            if (depth >= nearest.depth) { continue; }

            nearest.depth = depth;
            nearest.coord = samplePos;
        }
    }

    return nearest;
}

// Push this pixel's depth to every previous-frame pixel its reprojection has
// bilinear weight in, keeping the nearest depth at each destination.
fn reconstructPreviousDepth(pos: vec2<i32>, depth: f32, motionVector: vec2<f32>) {
    if (length(motionVector * vec2<f32>(params.renderSize)) <= MIN_REPROJECTION_PIXELS) { return; }

    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(params.renderSize);
    let bilinear = fiBilinear(uv + motionVector, params.renderSize);

    for (var i = 0; i < 4; i++) {
        if (bilinear.weights[i] <= RECONSTRUCTED_DEPTH_WEIGHT_THRESHOLD) { continue; }

        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, params.renderSize)) { continue; }

        // Non-negative floats compare in the same order as their bit patterns,
        // so an integer atomic min is a float min here.
        atomicMin(&reconstructedDepthPrevious[fiFlatIndex(samplePos, params.renderSize)], bitcast<u32>(depth));
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (!fiInBounds(pos, params.renderSize)) { return; }

    let nearest = findNearestDepth(pos);
    let motionVector = textureLoad(inputMotionVectors, nearest.coord, 0).xy / vec2<f32>(params.renderSize);

    let index = fiFlatIndex(pos, params.renderSize);
    dilatedDepth[index] = nearest.depth;
    dilatedMotionVectors[index] = motionVector;

    reconstructPreviousDepth(pos, nearest.depth, motionVector);
}
