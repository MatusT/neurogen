// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_frameinterpolation_setup.h plus the two reconstructed-depth clears AMD
// schedules as host-side clear jobs rather than shader work.
//
// TEMPORAL RESOURCE OWNERSHIP. This pass is the sole owner of every per-frame
// reset in the module, and must run first in `prepare()`, before any scatter.
// The resources below are accumulated into by atomic max/min from many
// invocations, so a stale value from the previous frame does not get
// overwritten — it competes, and can win. They cannot be left to their writers
// to initialise.
//
// Not cleared here, deliberately: the disocclusion mask, the preliminary blend
// colour, the blend weight, the inpainting pyramid and the interpolated colour
// are each fully overwritten by their own pass every frame, so clearing them
// would only cost bandwidth.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution.

// @import ./common.wgsl

// Two interleaved u32 entries per pixel, x then y. AMD keeps these as two
// separate textures; one buffer with a stride of two halves the binding count,
// which matters against WebGPU's 8-storage-buffer-per-stage floor.
@group(0) @binding(1) var<storage, read_write> gameMotionVectorField: array<u32>;
@group(0) @binding(2) var<storage, read_write> opticalFlowMotionVectorField: array<u32>;
@group(0) @binding(3) var<storage, read_write> reconstructedDepthPrevious: array<u32>;
@group(0) @binding(4) var<storage, read_write> reconstructedDepthInterpolated: array<u32>;
@group(0) @binding(5) var<storage, read> sceneChange: OpticalFlowSceneChange;
@group(0) @binding(6) var<storage, read_write> state: FrameInterpolationState;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);

    if (all(pos == vec2<i32>(0))) {
        let resetting = sceneChange.detected != 0u || params.reset != 0u;
        state.frameIndexSinceLastReset = select(state.frameIndexSinceLastReset + 1u, 0u, resetting);
    }

    if (!fiInBounds(pos, params.renderSize)) { return; }

    let index = fiFlatIndex(pos, params.renderSize);
    gameMotionVectorField[index * 2u] = 0u;
    gameMotionVectorField[index * 2u + 1u] = 0u;

    // Nearest-wins, so the sentinel has to be the furthest representable depth
    // rather than zero — zero is the near plane and would beat every scatter.
    let farSentinel = bitcast<u32>(FI_DEPTH_FAR_SENTINEL);
    reconstructedDepthPrevious[index] = farSentinel;
    reconstructedDepthInterpolated[index] = farSentinel;

    let opticalFlowSize = fiOpticalFlowGridSize();
    if (!fiInBounds(pos, opticalFlowSize)) { return; }

    let opticalFlowIndex = fiFlatIndex(pos, opticalFlowSize);
    opticalFlowMotionVectorField[opticalFlowIndex * 2u] = 0u;
    opticalFlowMotionVectorField[opticalFlowIndex * 2u + 1u] = 0u;
}
