// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// cbFrameInterpolation (ffx_frameinterpolation_callbacks_hlsl.h) reduced to the
// fields v1 actually reads, plus this module's frozen output contract.
// `@group(0) @binding(0)` is reserved for this uniform in every pass.
//
// Module input contract:
//   currentColor, previousColor : texture_2d<f32>, render resolution, raw
//                                 backbuffer values in `backbufferTransferFunction`'s
//                                 encoding.
//   inputDepth                  : texture_2d<f32>, .x is device depth, standard
//                                 (non-inverted) 0 = near / 1 = far, from a
//                                 finite perspective projection.
//   inputMotionVectors          : texture_2d<f32>, .xy is the displacement in
//                                 whole render-resolution pixels from a pixel in
//                                 the current frame to where it was in the
//                                 previous frame.
//   opticalFlowVectorField,
//   opticalFlowValidity         : Task 2's frozen output — see
//                                 ../opticalflow/params.wgsl.
//
// Module output contract:
//   interpolatedColor : array<vec4<f32>>, row-major at render resolution, the
//                       frame midway between previousColor and currentColor.
//                       Alpha is always 1; orchestration copies this buffer into
//                       whatever texture it hands back.
//   blendWeight       : array<f32> — the neural extension point. Declared at its
//                       binding in preliminary_blend.wgsl (writer) and
//                       final_blend.wgsl (reader); read those for the contract.
//
// Reduced v1 scope vs AMD's pass graph:
// - Render resolution == display resolution. No upscaler, so no jitter, no
//   low-res/high-res split, no `GetMaxRenderSize` rescale.
// - No interpolation rect / letterbox: the interpolation rect is the whole
//   frame, so `IsInRect` collapses into the ordinary bounds check and every
//   `fUvLetterBoxScale` factor is 1.
// - No distortion field, no debug views, no HUD-less present compositing.
// - Inpainting pyramid is 4 mips, where AMD builds 13.
// - AMD's separate 11-level *motion vector* inpainting pyramid is not built at
//   all. An unwritten game-motion-vector-field cell reads back invalid and
//   raises the colour inpainting weight instead — those cells are the
//   disoccluded regions the colour pyramid already covers.

// Beyond this the disocclusion mask's depth comparisons are noise.
const FI_EPSILON: f32 = 1e-03;

struct FrameInterpolationParams {
    renderSize: vec2<i32>,
    // Finite, non-inverted perspective projection planes. Required — the
    // disocclusion mask compares depths in view space (metres), and device
    // depth alone cannot be converted back without them.
    nearPlane: f32,
    farPlane: f32,
    verticalFovRadians: f32,
    // 0 = linear LDR, 1 = PQ, 2 = scRGB. Same encoding as OpticalFlowParams.
    backbufferTransferFunction: u32,
    minMaxLuminance: vec2<f32>,
    // Which inpainting mip the current pyramid dispatch writes; ignored by
    // every other pass.
    inpaintingMipLevel: u32,
    // Host-forced reset: resolution change, history discarded, and required on
    // the very first `prepare()` — there is no previous frame to interpolate
    // from and no scene-change flag will say so. Folded into the same counter
    // the scene-change flag drives; see FrameInterpolationState.
    reset: u32,
}

@group(0) @binding(0) var<uniform> params: FrameInterpolationParams;

// Task 2's frozen scene-change output, redeclared rather than imported:
// ../opticalflow/params.wgsl also declares `@group(0) @binding(0)`, so
// importing it would collide with this module's uniform.
struct OpticalFlowSceneChange {
    detected: u32,
}

// Frames since the last reset or scene cut, owned by setup.wgsl.
//
// AMD keeps this in a counter resource for the same reason it is needed here:
// the scene-change verdict is produced on the GPU (Task 2's scene-change
// buffer), so the host cannot know synchronously whether this frame starts a
// new sequence. Zero means "no usable previous frame" and is the scene-cut
// fallback condition both blends test.
struct FrameInterpolationState {
    frameIndexSinceLastReset: u32,
}

// FFX_OPTICALFLOW_BLOCK_SIZE — Task 2's vector field is one cell per 8x8 block
// of render-resolution pixels, and this module's optical-flow vector field
// keeps that same grid.
const OF_BLOCK_SIZE: i32 = 8;

fn fiOpticalFlowGridSize() -> vec2<i32> {
    return (params.renderSize + vec2<i32>(OF_BLOCK_SIZE - 1)) / vec2<i32>(OF_BLOCK_SIZE);
}
