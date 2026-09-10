// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation.h: warp the previous and current frames to the
// midpoint along both motion fields and blend them, producing the interpolated
// colour plus the weight that says how much of it is guesswork.
//
// AMD packs the weight into the colour's alpha channel. It is a separate buffer
// here because it is this module's extension point — see the `blendWeight`
// binding below.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution. Must run after
// disocclusion_mask.wgsl, and after both vector field passes.

// @import ./common.wgsl

@group(0) @binding(1) var currentColor: texture_2d<f32>;
@group(0) @binding(2) var previousColor: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> gameMotionVectorField: array<u32>;
@group(0) @binding(4) var<storage, read> opticalFlowMotionVectorField: array<u32>;
@group(0) @binding(5) var<storage, read> disocclusionMask: array<vec2<f32>>;
// Alpha is always 1; the weight that AMD keeps there lives in `blendWeight`.
@group(0) @binding(6) var<storage, read_write> preliminaryColor: array<vec4<f32>>;

// FROZEN CONTRACT — the module's neural extension point.
//
//   blendWeight : array<f32>, one scalar per render-resolution pixel, row-major,
//                 index `y * renderSize.x + x`. Range 0..1.
//
// Meaning: the fraction of the final colour that must come from inpainting
// rather than from either warped source. 0 means both source frames agree and
// the warped colour stands; 1 means neither frame contains this content at all
// and the colour beside it is meaningless.
//
// Written by: this pass, from the classical occlusion formula below.
// Read by:    inpainting_pyramid.wgsl (as coverage, `1 - blendWeight`) and
//             final_blend.wgsl (as the blend factor against the inpainted
//             colour). Neither reads the weight from anywhere else.
//
// Dispatch-order requirement: preliminary_blend -> [optional replacement pass]
// -> inpainting_pyramid mips 0..3 -> final_blend. A replacement runs as its own
// dispatch between the first two and overwrites this buffer wholesale; because
// the pyramid reads it too, a replacement changes both which pixels count as
// holes and what gets used to fill them, which is intended. Nothing downstream
// needs to change to accept one, and this file does not need to change to
// provide one.
@group(0) @binding(7) var<storage, read_write> blendWeight: array<f32>;
@group(0) @binding(8) var<storage, read> state: FrameInterpolationState;

// Below this many frames since a reset the motion fields have not settled, so
// the game vectors are trusted unconditionally rather than scored against the
// optical flow.
const MOTION_FIELD_WARMUP_FRAMES: u32 = 10u;

fn loadGameFieldMv(uv: vec2<f32>) -> FiVectorFieldEntry {
    let pos = vec2<i32>(uv * vec2<f32>(params.renderSize));
    if (!fiInBounds(pos, params.renderSize)) { return FiVectorFieldEntry(); }

    let index = fiFlatIndex(pos, params.renderSize) * 2u;
    var entry = fiUnpackVectorField(vec2<u32>(gameMotionVectorField[index], gameMotionVectorField[index + 1u]));
    fiAnnotateReprojection(&entry, uv);
    return entry;
}

// The optical flow field is an eighth of the render resolution, so unlike the
// game field it is interpolated rather than point sampled. Unpacking each tap
// before averaging, instead of averaging the packed words, is what keeps a
// high-priority neighbour from dragging the vector with its priority bits.
fn sampleOpticalFlowFieldMv(uv: vec2<f32>) -> FiVectorFieldEntry {
    let gridSize = fiOpticalFlowGridSize();
    let bilinear = fiBilinear(uv, gridSize);

    var entry: FiVectorFieldEntry;
    var weightSum = 0.0;
    for (var i = 0; i < 4; i++) {
        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, gridSize)) { continue; }

        let index = fiFlatIndex(samplePos, gridSize) * 2u;
        let sample = fiUnpackVectorField(vec2<u32>(
            opticalFlowMotionVectorField[index],
            opticalFlowMotionVectorField[index + 1u],
        ));

        let weight = bilinear.weights[i];
        entry.motionVector += sample.motionVector * weight;
        entry.highPriorityFactor += sample.highPriorityFactor * weight;
        entry.lowPriorityFactor += sample.lowPriorityFactor * weight;
        weightSum += weight;
    }

    if (weightSum > 0.0) {
        entry.motionVector /= weightSum;
        entry.highPriorityFactor /= weightSum;
        entry.lowPriorityFactor /= weightSum;
    }

    fiAnnotateReprojection(&entry, uv);
    return entry;
}

fn raiseInpaintingWeight(weight: ptr<function, f32>, factor: f32) {
    *weight = saturate(max(*weight, factor));
}

fn computeInterpolatedColor(pos: vec2<i32>, weight: ptr<function, f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(params.renderSize);

    let gameMv = loadGameFieldMv(uv);
    let opticalFlowMv = sampleOpticalFlowFieldMv(uv);

    // The mask is already binarised and sits at this exact resolution, so
    // AMD's filtered fetch of it degenerates to reading this pixel's texel.
    var disocclusionFactor = disocclusionMask[fiFlatIndex(pos, params.renderSize)];

    // A field entry is the half-frame vector, so the previous frame is reached
    // by adding it and the current frame by subtracting it.
    let previousGame = fiGatherColor(previousColor, uv + gameMv.motionVector, params.renderSize);
    let currentGame = fiGatherColor(currentColor, uv - gameMv.motionVector, params.renderSize);
    let previousOpticalFlow = fiGatherColor(previousColor, uv + opticalFlowMv.motionVector, params.renderSize);
    let currentOpticalFlow = fiGatherColor(currentColor, uv - opticalFlowMv.motionVector, params.renderSize);

    disocclusionFactor.x *= f32(!gameMv.posOutside);
    disocclusionFactor.y *= f32(!gameMv.negOutside);

    // Neither direction usable: nothing to warp from, so this is a hole.
    raiseInpaintingWeight(weight, f32(length(disocclusionFactor) <= FI_EPSILON));

    // Slide the blend entirely onto whichever frame still sees this surface.
    // With both visible this stays at the midpoint.
    let t = 0.5 + 0.5 * (1.0 - disocclusionFactor.x) - 0.5 * (1.0 - disocclusionFactor.y);
    var color = mix(previousGame.rgb, currentGame.rgb, saturate(t));
    let disoccludedFactor = saturate(1.0 - min(disocclusionFactor.x, disocclusionFactor.y));

    // A zero gather weight means the reprojection left the frame entirely,
    // which the disocclusion factor does not catch on its own.
    if (previousGame.w == 0.0) { color = currentGame.rgb; }
    else if (currentGame.w == 0.0) { color = previousGame.rgb; }

    // Subsumed by the disocclusion test above as long as the interpolation rect
    // is the whole frame: a gather only empties when every tap left the frame,
    // which means the reprojected UV left it too, which already zeroed both
    // disocclusion channels. Kept because AMD's version is not redundant — its
    // gather tests the letterboxed interpolation rect while the UV test covers
    // the full frame — and because a motion vector from a source other than
    // this module's scatter can break the implication.
    if (previousGame.w == 0.0 && currentGame.w == 0.0) { *weight = 1.0; }

    var opticalFlowT = 0.5;
    if (previousOpticalFlow.w == 0.0) { opticalFlowT = 1.0; }
    else if (currentOpticalFlow.w == 0.0) { opticalFlowT = 0.0; }

    let opticalFlowColor = mix(previousOpticalFlow.rgb, currentOpticalFlow.rgb, opticalFlowT);

    // Score the two motion fields against each other: whichever warp brings the
    // two frames into closer agreement is the one that tracked the content.
    let opticalFlowSimilarity = fiNormalizedDot3(previousOpticalFlow.rgb, currentOpticalFlow.rgb);
    var gameSimilarity = fiNormalizedDot3(previousGame.rgb, currentGame.rgb);
    // Disagreement inside a disocclusion is expected, not a sign the game
    // vectors are wrong, so forgive it in proportion.
    gameSimilarity = mix(max(FI_EPSILON, gameSimilarity), 1.0, saturate(disoccludedFactor));

    var gameMvBias = saturate(gameSimilarity / max(FI_EPSILON, opticalFlowSimilarity));
    gameMvBias = mix(gameMvBias, 1.0, f32(state.frameIndexSinceLastReset < MOTION_FIELD_WARMUP_FRAMES));

    return mix(opticalFlowColor, color, saturate(gameMvBias));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (!fiInBounds(pos, params.renderSize)) { return; }

    var weight = 0.0;
    var color: vec3<f32>;

    // Scene-cut fallback. On the first frame of a sequence the previous frame
    // belongs to different content and every temporal resource was just
    // cleared, so interpolating would blend in reconstructed garbage. Show one
    // real frame instead — the standard remedy, and the only one that cannot
    // introduce an artefact.
    if (state.frameIndexSinceLastReset == 0u) {
        color = textureLoad(currentColor, pos, 0).rgb;
    } else {
        color = computeInterpolatedColor(pos, &weight);
    }

    let index = fiFlatIndex(pos, params.renderSize);
    preliminaryColor[index] = vec4<f32>(color, 1.0);
    blendWeight[index] = weight;
}
