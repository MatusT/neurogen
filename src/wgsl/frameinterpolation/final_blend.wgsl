// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_inpainting.h: fill the holes the preliminary blend
// left, by pulling colour from every pyramid level at once. Fine levels supply
// a nearby colour where one survived; coarse levels supply a plausible average
// where nothing nearby did. Pixels the preliminary blend was confident about
// pass straight through.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution. Must run last, after
// every inpainting_pyramid mip.

// @import ./common.wgsl

@group(0) @binding(1) var currentColor: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> preliminaryColor: array<vec4<f32>>;
// See the contract at this buffer's binding in preliminary_blend.wgsl. Read
// here as the fraction of the final colour to take from inpainting; this pass
// never computes a weight of its own, so a replacement writer fully controls it.
@group(0) @binding(3) var<storage, read> blendWeight: array<f32>;
@group(0) @binding(4) var<storage, read> inpaintingPyramid: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> interpolatedColor: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> state: FrameInterpolationState;

// Coarse levels are plausible but vague, so their say falls off steeply.
const MIP_WEIGHT_FALLOFF_EXPONENT: f32 = 3.0;

// Colour and total weight gathered from one pyramid level. Uncovered taps are
// dropped rather than averaged in as black.
fn computeInpaintingLevel(uv: vec2<f32>, level: i32) -> vec4<f32> {
    let mipSize = fiPyramidMipSize(level);
    let bilinear = fiBilinear(uv, mipSize);
    let mipOffset = fiPyramidMipOffset(level);

    var color = vec4<f32>(0.0);
    for (var i = 0; i < 4; i++) {
        let samplePos = bilinear.basePos + fiBilinearOffset(i);
        if (!fiInBounds(samplePos, mipSize)) { continue; }

        let sample = inpaintingPyramid[mipOffset + fiFlatIndex(samplePos, mipSize)];
        let weight = bilinear.weights[i] * f32(sample.w > 0.0);
        color += vec4<f32>(sample.rgb * weight, weight);
    }

    return color;
}

// Returns the accumulated colour in .rgb and the total weight in .w. A weight
// of zero means no level had any covered pixel within reach — impossible with
// AMD's 13 mips, where the coarsest is a single texel, but reachable with the
// four this port builds. The caller has to check it rather than divide.
fn computeInpainting(pos: vec2<i32>) -> vec4<f32> {
    let uv = (vec2<f32>(pos) + 0.5) / vec2<f32>(params.renderSize);

    var color = vec4<f32>(0.0);
    for (var level = 0; level < FI_INPAINTING_MIP_COUNT; level++) {
        let mipColor = computeInpaintingLevel(uv, level);
        if (mipColor.w <= 0.0) { continue; }

        let falloff = 1.0 - f32(level) / f32(FI_INPAINTING_MIP_COUNT);
        let mipWeight = pow(falloff, MIP_WEIGHT_FALLOFF_EXPONENT) * mipColor.w;
        color += vec4<f32>(mipColor.rgb / mipColor.w, 1.0) * mipWeight;
    }

    return color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (!fiInBounds(pos, params.renderSize)) { return; }

    let index = fiFlatIndex(pos, params.renderSize);

    // Scene-cut fallback, repeated here rather than left to the preliminary
    // blend's copy of the current frame: the weight this pass reads may have
    // been rewritten by a replacement pass in between, and inpainting over a
    // real frame would undo a fallback that had already done its job.
    if (state.frameIndexSinceLastReset == 0u) {
        interpolatedColor[index] = vec4<f32>(textureLoad(currentColor, pos, 0).rgb, 1.0);
        return;
    }

    var color = preliminaryColor[index].rgb;
    // Clamped to the range the contract declares rather than trusted: a
    // replacement writer producing 1.0001 would extrapolate past the inpainted
    // colour here instead of blending towards it.
    let weight = saturate(blendWeight[index]);

    if (weight > FI_EPSILON) {
        let inpainted = computeInpainting(pos);
        if (inpainted.w > 0.0) {
            color = mix(color, inpainted.rgb / inpainted.w, weight);
        }
    }

    interpolatedColor[index] = vec4<f32>(color, 1.0);
}
