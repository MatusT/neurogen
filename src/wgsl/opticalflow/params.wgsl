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
//   sceneChangeDetected    : OpticalFlowSceneChange, 0 or 1 for this frame.
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

// The luma pyramid floor-halves per level, matching AMD's level textures and
// the clamp its packed-luma loads assume.
fn ofLumaLevelSize(level: u32) -> vec2<i32> {
    return max(params.lumaSize >> vec2<u32>(level), vec2<i32>(1));
}

// The motion vector grid ceil-halves instead — the two conventions genuinely
// disagree in AMD's source, so they are kept apart here.
fn ofFlowLevelSize(level: u32) -> vec2<i32> {
    var size = (params.lumaSize + vec2<i32>(OF_BLOCK_SIZE - 1)) / vec2<i32>(OF_BLOCK_SIZE);

    // Callers reach the next-finer level as `pyramidLevel - 1`, which wraps to
    // 0xffffffff if a caller ever dispatches them at level 0. Bounding the
    // count turns that mistake into a wrong size rather than a GPU hang.
    for (var i = 0u; i < min(level, OF_MAX_PYRAMID_LEVELS); i++) {
        size = (size + vec2<i32>(1)) / vec2<i32>(2);
    }

    return size;
}
