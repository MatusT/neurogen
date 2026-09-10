// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_compute_inpainting_pyramid.h: a mip chain over the
// preliminary blend's output, where each level averages only the pixels that
// carried real content. Coarser levels therefore reach further for a colour,
// which is what lets the final blend fill a hole from progressively wider
// context.
//
// AMD builds this with SPD, a single dispatch that downsamples every mip at
// once by passing tiles through wave intrinsics and a cross-workgroup atomic
// counter. WebGPU has neither guaranteed subgroups nor a cross-workgroup
// barrier, so the chain is built one mip per dispatch instead — slower, and
// exactly equivalent, as long as the reduction operator is preserved. It is not
// a box filter: taps are weighted by their own coverage, so a pixel that was
// itself a hole contributes nothing rather than averaging a hole's colour in.
//
// Dispatch: once per mip level 0..3, coarsest last, each with
// `params.inpaintingMipLevel` set and (ceil(mipW/8), ceil(mipH/8)) workgroups.
// Must run after preliminary_blend.wgsl, and after any pass that replaces
// `blendWeight`.

// @import ./common.wgsl

@group(0) @binding(1) var<storage, read> preliminaryColor: array<vec4<f32>>;
// See the contract at this buffer's binding in preliminary_blend.wgsl. Read
// here as coverage — how much of the pixel is real — hence the inversion.
@group(0) @binding(2) var<storage, read> blendWeight: array<f32>;
// All four mips share one buffer, laid out coarsest-last; see fiPyramidMipOffset.
@group(0) @binding(3) var<storage, read_write> inpaintingPyramid: array<vec4<f32>>;

fn loadSource(pos: vec2<i32>, level: i32) -> vec4<f32> {
    if (level == 0) {
        if (!fiInBounds(pos, params.renderSize)) { return vec4<f32>(0.0); }

        let index = fiFlatIndex(pos, params.renderSize);
        return vec4<f32>(preliminaryColor[index].rgb, saturate(1.0 - blendWeight[index]));
    }

    let sourceSize = fiPyramidMipSize(level - 1);
    if (!fiInBounds(pos, sourceSize)) { return vec4<f32>(0.0); }

    return inpaintingPyramid[fiPyramidMipOffset(level - 1) + fiFlatIndex(pos, sourceSize)];
}

// AMD's SpdReduce4. Note the result's own coverage is the coverage-weighted
// mean of the four coverages, not their average: a quad with one fully covered
// tap stays strongly covered, while four barely-covered taps fade out.
fn reduceQuad(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    let coverageSum = v0.w + v1.w + v2.w + v3.w;
    if (coverageSum == 0.0) { return vec4<f32>(0.0); }

    return (v0 * v0.w + v1 * v1.w + v2 * v2.w + v3 * v3.w) / coverageSum;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    let level = i32(params.inpaintingMipLevel);
    let mipSize = fiPyramidMipSize(level);
    if (!fiInBounds(pos, mipSize)) { return; }

    // An odd source dimension leaves the last column or row of quads with taps
    // past the edge; those read back as zero coverage and drop out.
    let sourcePos = pos * 2;
    let reduced = reduceQuad(
        loadSource(sourcePos, level),
        loadSource(sourcePos + vec2<i32>(1, 0), level),
        loadSource(sourcePos + vec2<i32>(0, 1), level),
        loadSource(sourcePos + vec2<i32>(1, 1), level),
    );

    inpaintingPyramid[fiPyramidMipOffset(level) + fiFlatIndex(pos, mipSize)] = reduced;
}
