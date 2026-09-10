// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_generate_scd_histogram.h: builds the nine regional luma
// histograms for this frame. Each workgroup strides a column of its region,
// tallies into workgroup memory, then folds its 256 bins into the global
// histogram with one atomic add per bin.
//
// The global histogram is a storage buffer of atomic<u32>; AMD uses an
// R32_UINT texture UAV, which WebGPU cannot do atomics on.
//
// Dispatch: (ceil(((W/4)/3)/32), 16, 9) — z selects the region.

// @import ./common.wgsl
// @import ./params.wgsl
// @import ./scd.wgsl

@group(0) @binding(1) var<storage, read> luma: array<u32>;
@group(0) @binding(2) var<storage, read_write> scdHistogram: array<atomic<u32>>;

// LBASE: the same luma from different invocations is spread over this many
// adjacent slots to keep them off one workgroup-memory bank, then summed back.
const BANK_SPREAD: u32 = 10u;
const PIXELS_PER_STEP: i32 = 4;
// Rows advance by the full dispatch height (16 groups of 8 rows).
const ROW_STRIDE: i32 = 128;

var<workgroup> localHistogram: array<atomic<u32>, SCD_HISTOGRAM_BINS * BANK_SPREAD>;

fn loadLuma(pos: vec2<i32>, size: vec2<i32>) -> u32 {
    if (!ofInBounds(pos, size)) { return 0u; }

    return luma[ofFlatIndex(pos, size)];
}

@compute @workgroup_size(32, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>,
        @builtin(workgroup_id) groupId: vec3<u32>,
        @builtin(local_invocation_index) localIndex: u32) {
    let size = ofLumaLevelSize(0u);
    let region = size / SCD_HISTOGRAMS_PER_DIM;
    let regionId = vec2<i32>(i32(groupId.z) % SCD_HISTOGRAMS_PER_DIM, i32(groupId.z) / SCD_HISTOGRAMS_PER_DIM);
    let regionStart = region * regionId;
    let regionStop = regionStart + region;

    let spreadBase = localIndex * BANK_SPREAD;
    for (var i = 0u; i < BANK_SPREAD; i++) {
        atomicStore(&localHistogram[spreadBase + i], 0u);
    }
    workgroupBarrier();

    var coord = regionStart + vec2<i32>(PIXELS_PER_STEP * i32(globalId.x), i32(globalId.y));
    if (coord.x < regionStop.x) {
        let spread = localIndex % BANK_SPREAD;
        for (; coord.y < regionStop.y; coord.y += ROW_STRIDE) {
            for (var dx = 0; dx < PIXELS_PER_STEP; dx++) {
                let value = loadLuma(coord + vec2<i32>(dx, 0), size);
                atomicAdd(&localHistogram[value * BANK_SPREAD + spread], 1u);
            }
        }
    }
    workgroupBarrier();

    // Invocation n owns bin n: its spread slots are contiguous, so folding is
    // a straight run of BANK_SPREAD loads.
    var total = 0u;
    for (var i = 0u; i < BANK_SPREAD; i++) {
        total += atomicLoad(&localHistogram[spreadBase + i]);
    }

    atomicAdd(&scdHistogram[groupId.z * SCD_HISTOGRAM_BINS + localIndex], total);
}
