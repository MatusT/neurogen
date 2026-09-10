// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// cbOF (ffx_opticalflow_callbacks_hlsl.h) plus the pyramid geometry every
// optical-flow pass derives from it, and the module's frozen output types.
// `@group(0) @binding(0)` is reserved for this uniform in every pass.
//
// Module output contract (consumed by frame interpolation):
//   opticalFlowVectorField : array<vec2<i32>>, row-major, one motion vector per
//                            OF_BLOCK_SIZE x OF_BLOCK_SIZE block of render-
//                            resolution pixels — so its grid is
//                            ceil(renderSize / 8), a 1/8 working resolution.
//                            Units are whole render-resolution pixels; the
//                            vector points from the block in the current frame
//                            to its match in the previous frame, the same
//                            direction as the game motion vectors.
//   opticalFlowValidity    : array<u32>, same grid and indexing as the vector
//                            field. 1 where the cell's winning block match was
//                            close enough to trust, 0 otherwise. Without it a
//                            zero vector is ambiguous three ways — scene-change
//                            reset, a block confidently judged stationary, and
//                            a cell the search never reached — and only the
//                            middle one is usable motion. Caveat: it scores the
//                            match residual, so a featureless cell reads valid
//                            even though its vector is arbitrary.
//   sceneChangeDetected    : OpticalFlowSceneChange, 0 or 1 for this frame.
//
// The search covers a whole tile per workgroup, so at some resolutions its
// dispatch stops one cell short of the flow grid. The orchestration must clear
// the validity buffer the search writes into before each of its dispatches,
// which is what makes "never reached" read as 0 rather than as last frame's
// verdict.
//
// Reduced v1 scope: pyramidLevelCount is 2, where AMD runs 7 levels.

// FFX_OPTICALFLOW_BLOCK_SIZE.
const OF_BLOCK_SIZE: i32 = 8;

// Frozen v1 pyramid depth. Only used to bound the level loop below.
const OF_MAX_PYRAMID_LEVELS: u32 = 2u;

struct OpticalFlowParams {
    // Level-0 luma dimensions, i.e. the render resolution.
    lumaSize: vec2<i32>,
    // Pyramid level this dispatch operates on; 0 is finest.
    pyramidLevel: u32,
    pyramidLevelCount: u32,
    frameIndex: u32,
    // 0 = linear LDR, 1 = PQ, 2 = scRGB.
    backbufferTransferFunction: u32,
    minMaxLuminance: vec2<f32>,
}

// AMD keeps a bitfield of the last four frames' verdicts; v1 only needs to
// know whether the scene changed this frame.
struct OpticalFlowSceneChange {
    detected: u32,
}

@group(0) @binding(0) var<uniform> params: OpticalFlowParams;

// Passes that write the next-finer level derive it as `pyramidLevel - 1`,
// which wraps to 0xffffffff if one is ever dispatched at level 0. Clamping
// turns that mistake into a wrong size rather than an indeterminate shift or a
// four-billion iteration loop.
fn ofClampLevel(level: u32) -> u32 {
    return min(level, OF_MAX_PYRAMID_LEVELS);
}

// The luma pyramid floor-halves per level, matching AMD's level textures and
// the clamp its packed-luma loads assume.
fn ofLumaLevelSize(level: u32) -> vec2<i32> {
    return max(params.lumaSize >> vec2<u32>(ofClampLevel(level)), vec2<i32>(1));
}

// The motion vector grid ceil-halves instead — the two conventions genuinely
// disagree in AMD's source, so they are kept apart here.
fn ofFlowLevelSize(level: u32) -> vec2<i32> {
    var size = (params.lumaSize + vec2<i32>(OF_BLOCK_SIZE - 1)) / vec2<i32>(OF_BLOCK_SIZE);

    for (var i = 0u; i < ofClampLevel(level); i++) {
        size = (size + vec2<i32>(1)) / vec2<i32>(2);
    }

    return size;
}
