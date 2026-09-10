// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_filter_optical_flow_v5.h: a vector median over the 3x3
// neighbourhood, keeping whichever of the nine candidates sits closest to all
// the others. Unlike a component-wise median this only ever emits a vector the
// search actually produced, so it removes outliers without inventing motion.
//
// Runs after the search at every level; at level 0 its destination is the
// module's opticalFlowVectorField output.
//
// Dispatch: (ceil(flowW/16), ceil(flowH/4)) for the level's flow grid.

// @import ./common.wgsl
// @import ./params.wgsl

@group(0) @binding(1) var<storage, read> flowIn: array<vec2<i32>>;
@group(0) @binding(2) var<storage, read_write> flowOut: array<vec2<i32>>;

const FILTER_TAPS: i32 = 9;
// The winner's index rides in the low nibble so one min() carries both the
// score and which tap produced it.
const FILTER_SCORE_SHIFT: u32 = 4u;
const FILTER_INDEX_MASK: u32 = 0xfu;

@compute @workgroup_size(16, 4)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let size = ofFlowLevelSize(params.pyramidLevel);
    let pos = vec2<i32>(globalId.xy);
    if (!ofInBounds(pos, size)) { return; }

    var taps: array<vec2<i32>, FILTER_TAPS>;
    var tap = 0;
    for (var x = -1; x < 2; x++) {
        for (var y = -1; y < 2; y++) {
            let neighbour = pos + vec2<i32>(x, y);
            if (ofInBounds(neighbour, size)) {
                taps[tap] = flowIn[ofFlatIndex(neighbour, size)];
            } else {
                taps[tap] = vec2<i32>(0);
            }
            tap++;
        }
    }

    var best = 0xffffffffu;
    for (var i = 0; i < FILTER_TAPS; i++) {
        var score = 0u;
        for (var j = 0; j < FILTER_TAPS; j++) {
            let delta = taps[i] - taps[j];
            score += u32(delta.x * delta.x + delta.y * delta.y);
        }

        best = min((score << FILTER_SCORE_SHIFT) | u32(i), best);
    }

    flowOut[ofFlatIndex(pos, size)] = taps[best & FILTER_INDEX_MASK];
}
