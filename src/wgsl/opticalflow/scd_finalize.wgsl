// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// The tail of ffx_opticalflow_compute_scd_divergence.h, as its own dispatch.
// AMD has each divergence workgroup bump a completion counter and lets
// whichever one sees itself finish last publish the verdict and reset the
// buffers. WebGPU has no cross-workgroup barrier and no way to detect the last
// workgroup, so the same work runs after the divergence dispatch has retired.
//
// Publishes the module's sceneChangeDetected output, rolls this frame's
// normalised histograms into the previous-frame buffer, and clears the
// accumulators for next frame.
//
// Dispatch: (1, 1, 1). One workgroup covers all nine histograms.

// @import ./params.wgsl
// @import ./scd.wgsl

@group(0) @binding(1) var<storage, read_write> scdHistogram: array<u32>;
@group(0) @binding(2) var<storage, read_write> scdPreviousHistogram: array<f32>;
@group(0) @binding(3) var<storage, read> scdFilteredHistogram: array<f32>;
@group(0) @binding(4) var<storage, read_write> scdTemp: array<u32>;
@group(0) @binding(5) var<storage, read_write> sceneChange: OpticalFlowSceneChange;

// No previous frame to diverge from yet, so the first frames are always
// treated as a cut.
const SCD_WARMUP_FRAMES: u32 = 5u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) localIndex: u32) {
    for (var histogram = 0u; histogram < SCD_HISTOGRAM_COUNT; histogram++) {
        let bin = histogram * SCD_HISTOGRAM_BINS + localIndex;
        scdPreviousHistogram[bin] = scdFilteredHistogram[bin];
        scdHistogram[bin] = 0u;
    }

    if (localIndex != 0u) { return; }

    // The least divergent of the three shifts wins, so a histogram that merely
    // slid a bin does not read as a cut.
    var divergence = scdTemp[0];
    for (var shift = 1u; shift < SCD_SHIFT_COUNT; shift++) {
        divergence = min(divergence, scdTemp[shift]);
    }

    for (var shift = 0u; shift < SCD_SHIFT_COUNT; shift++) {
        scdTemp[shift] = 0u;
    }

    let changed = params.frameIndex <= SCD_WARMUP_FRAMES
        || f32(divergence) / SCD_DIVERGENCE_FACTOR > SCD_SCENE_CHANGE_THRESHOLD;
    sceneChange.detected = u32(changed);
}
