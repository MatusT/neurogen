// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// LoadFirstImagePackedLuma / LoadSecondImagePackedLuma from
// ffx_opticalflow_callbacks_hlsl.h. "First" is the current frame's luma,
// "second" the previous frame's, both at params.pyramidLevel. Bindings 1 and 2
// are reserved for this pair in every pass that matches blocks between them.

// @import ./common.wgsl
// @import ./params.wgsl

@group(0) @binding(1) var<storage, read> lumaFirst: array<u32>;
@group(0) @binding(2) var<storage, read> lumaSecond: array<u32>;

// Clamping the 4-wide sample window into the level keeps every index inside
// the buffer; ofPackLuma then replicates the edge sample over whatever fell
// off screen, so no separate bounds check is needed.
fn ofClampLumaWindow(pos: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(clamp(pos.x, 0, size.x - 4), clamp(pos.y, 0, size.y - 1));
}

fn ofLoadFirstPackedLuma(pos: vec2<i32>) -> u32 {
    let size = ofLumaLevelSize(params.pyramidLevel);
    let base = ofFlatIndex(ofClampLumaWindow(pos, size), size);

    return ofPackLuma(size.x, pos.x, lumaFirst[base], lumaFirst[base + 1u], lumaFirst[base + 2u], lumaFirst[base + 3u]);
}

fn ofLoadSecondPackedLuma(pos: vec2<i32>) -> u32 {
    let size = ofLumaLevelSize(params.pyramidLevel);
    let base = ofFlatIndex(ofClampLumaWindow(pos, size), size);

    return ofPackLuma(size.x, pos.x, lumaSecond[base], lumaSecond[base + 1u], lumaSecond[base + 2u], lumaSecond[base + 3u]);
}
