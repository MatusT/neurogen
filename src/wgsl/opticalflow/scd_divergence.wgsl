// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_compute_scd_divergence.h: scores how far this frame's luma
// histograms have moved from last frame's. One workgroup per (region, shift)
// pair blurs its histogram, normalises it, and accumulates a symmetrised
// Kullback-Leibler divergence against the stored previous histogram. The
// verdict itself is drawn in scd_finalize.wgsl.
//
// AMD closes this pass with a "was I the last workgroup?" counter that lets
// one workgroup read every other's result and publish the flag. WebGPU has no
// cross-workgroup barrier, so that tail is a separate dispatch. Its writes to
// the previous-histogram and histogram buffers move with it: in AMD's version
// the shift-1 workgroup overwrites buffers the shift-0 and shift-2 workgroups
// are still reading. This pass therefore only stages its normalised histogram,
// and finalize does the swap.
//
// Dispatch: (9, 3) — x selects the region, y the shift.

// @import ./params.wgsl
// @import ./scd.wgsl

@group(0) @binding(1) var<storage, read> scdHistogram: array<u32>;
@group(0) @binding(2) var<storage, read> scdPreviousHistogram: array<f32>;
@group(0) @binding(3) var<storage, read_write> scdFilteredHistogram: array<f32>;
@group(0) @binding(4) var<storage, read_write> scdTemp: array<atomic<u32>>;

// Half of an 11-tap Gaussian, mirrored about SCD_GAUSSIAN[5].
const SCD_GAUSSIAN = array<f32, 6>(0.0088122291, 0.027143577, 0.065114059, 0.12164907, 0.17699835, 0.20056541);
const SCD_KERNEL_SHIFT: i32 = -5;
const SCD_SHIFT_DOWN: u32 = 0u;
const SCD_SHIFT_NONE: u32 = 1u;

var<workgroup> sourceHistogram: array<f32, SCD_HISTOGRAM_BINS>;
var<workgroup> filteredHistogram: array<f32, SCD_HISTOGRAM_BINS>;
var<workgroup> sumScratch: array<f32, SCD_HISTOGRAM_BINS>;
var<workgroup> divergenceScratch: array<vec2<f32>, SCD_HISTOGRAM_BINS>;

fn histogramTap(bin: i32) -> f32 {
    return sourceHistogram[clamp(bin, 0, i32(SCD_HISTOGRAM_BINS) - 1)];
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) groupId: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {
    let shift = groupId.y;
    let bin = groupId.x * SCD_HISTOGRAM_BINS + localIndex;

    sourceHistogram[localIndex] = f32(scdHistogram[bin]);
    workgroupBarrier();

    let tapBase = i32(localIndex) + SCD_KERNEL_SHIFT;
    var value = 0.0;
    value += SCD_GAUSSIAN[0] * histogramTap(tapBase + 0);
    value += SCD_GAUSSIAN[1] * histogramTap(tapBase + 1);
    value += SCD_GAUSSIAN[2] * histogramTap(tapBase + 2);
    value += SCD_GAUSSIAN[3] * histogramTap(tapBase + 3);
    value += SCD_GAUSSIAN[4] * histogramTap(tapBase + 4);
    value += SCD_GAUSSIAN[5] * histogramTap(tapBase + 5);
    value += SCD_GAUSSIAN[4] * histogramTap(tapBase + 6);
    value += SCD_GAUSSIAN[3] * histogramTap(tapBase + 7);
    value += SCD_GAUSSIAN[2] * histogramTap(tapBase + 8);
    value += SCD_GAUSSIAN[1] * histogramTap(tapBase + 9);
    value += SCD_GAUSSIAN[0] * histogramTap(tapBase + 10);

    // Keeps every bin strictly positive: the divergence below takes logs of
    // ratios of these values.
    value += 1.0;

    // The bin vacated by the shift is refilled with the same floor.
    if (shift == SCD_SHIFT_DOWN) {
        if (localIndex == 0u) {
            filteredHistogram[SCD_HISTOGRAM_BINS - 1u] = 1.0;
        } else {
            filteredHistogram[localIndex - 1u] = value;
        }
    } else if (shift == SCD_SHIFT_NONE) {
        filteredHistogram[localIndex] = value;
    } else {
        if (localIndex == SCD_HISTOGRAM_BINS - 1u) {
            filteredHistogram[0] = 1.0;
        } else {
            filteredHistogram[localIndex + 1u] = value;
        }
    }
    workgroupBarrier();

    // AMD leans on wave-width sync below 64 lanes here; every step needs its
    // own barrier, outside the lane predicate, without subgroups.
    sumScratch[localIndex] = filteredHistogram[localIndex];
    workgroupBarrier();

    for (var stride = SCD_HISTOGRAM_BINS / 2u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            sumScratch[localIndex] += sumScratch[localIndex + stride];
        }
        workgroupBarrier();
    }

    let current = filteredHistogram[localIndex] / sumScratch[0];
    let previous = scdPreviousHistogram[bin];

    // `scdPreviousHistogram` is a fresh storage buffer's zero-initialized
    // default until scd_finalize.wgsl has run at least once (frame 0, or
    // after any external reset). `previous` is then 0, making this
    // divergence value +-inf/NaN. That's intentional dead computation, not a
    // bug: scd_finalize.wgsl unconditionally forces sceneChangeDetected for
    // params.frameIndex <= SCD_WARMUP_FRAMES regardless of what lands in
    // scdTemp here, and `scdFilteredHistogram` below (what actually becomes
    // next frame's `scdPreviousHistogram`) is `current` alone — it never
    // reads `previous`, so nothing NaN/Inf-poisoned is ever persisted.
    // Symmetrised Kullback-Leibler: both directions, so a bin appearing and a
    // bin vanishing weigh the same.
    divergenceScratch[localIndex] = vec2<f32>(current * log(current / previous),
                                              previous * log(previous / current));
    workgroupBarrier();

    for (var stride = SCD_HISTOGRAM_BINS / 2u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            divergenceScratch[localIndex] += divergenceScratch[localIndex + stride];
        }
        workgroupBarrier();
    }

    if (localIndex == 0u) {
        let sum = divergenceScratch[0];
        let divergence = 1.0 - exp(-(abs(sum.x) + abs(sum.y)));
        atomicAdd(&scdTemp[shift], u32((divergence / f32(SCD_HISTOGRAM_COUNT)) * SCD_DIVERGENCE_FACTOR));
    }

    // Only the unshifted curve becomes next frame's reference.
    if (shift == SCD_SHIFT_NONE) {
        scdFilteredHistogram[bin] = current;
    }
}
