// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_scale_optical_flow_advanced_v5.h: hands a level's motion
// vectors down to the next finer level, which then uses them as its search
// prediction. This is what makes the search hierarchical, so it runs between
// every pair of levels.
//
// A destination cell has four plausible parents — the coarse cell it falls in
// and its three neighbours across the nearest cell boundary. Each is scored by
// matching the destination cell's luma against the previous frame at that
// parent's vector, and the winner is doubled into the finer level's units. The
// workgroup's z axis carries the four candidates, so the scoring is a plain
// workgroup-memory pick with no cross-lane ops involved.
//
// Dispatch, with params.pyramidLevel set to the source (coarser) level:
// (ceil(dstW/4), ceil(dstH/4)) over the destination level's flow grid.

// @import ./luma.wgsl

@group(0) @binding(3) var<storage, read> flowSrc: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read_write> flowDst: array<vec2<i32>>;
@group(0) @binding(5) var<storage, read> sceneChange: OpticalFlowSceneChange;

const CANDIDATE_COUNT: i32 = 4;
// Luma rows compared per candidate; four rows of four pixels covers the
// destination cell's footprint at the source level's resolution.
const COMPARE_ROWS: i32 = 4;
const COMPARE_WIDTH: i32 = 4;

var<workgroup> candidateVectors: array<array<array<vec2<i32>, 4>, 4>, CANDIDATE_COUNT>;
var<workgroup> candidateLuma: array<array<array<u32, 4>, 4>, COMPARE_ROWS>;
var<workgroup> candidateSads: array<array<array<u32, 4>, 4>, CANDIDATE_COUNT>;
var<workgroup> sceneChanged: u32;

fn storeFlow(pos: vec2<i32>, motionVector: vec2<i32>, size: vec2<i32>) {
    if (!ofInBounds(pos, size)) { return; }

    flowDst[ofFlatIndex(pos, size)] = motionVector;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) globalIdIn: vec3<u32>,
        @builtin(local_invocation_id) localIdIn: vec3<u32>,
        @builtin(local_invocation_index) localIndex: u32) {
    let globalId = vec3<i32>(globalIdIn);
    let localId = vec3<i32>(localIdIn);
    let dstSize = ofFlowLevelSize(params.pyramidLevel - 1u);

    // Republished through workgroup memory so the barriers below stay in
    // uniform control flow.
    if (localIndex == 0u) { sceneChanged = sceneChange.detected; }
    if (workgroupUniformLoad(&sceneChanged) != 0u) {
        if (localId.z == 0) {
            storeFlow(globalId.xy, vec2<i32>(0), dstSize);
        }
        return;
    }

    // The four parents: the containing coarse cell plus its neighbours on
    // whichever side the destination cell sits.
    let parentOffset = vec2<i32>((localId.z % 2) - 1 + (globalId.x % 2),
                                 (localId.z / 2) - 1 + (globalId.y % 2));
    let srcPos = globalId.xy / 2 + parentOffset;
    let srcSize = ofFlowLevelSize(params.pyramidLevel);

    var candidate = vec2<i32>(0);
    if (ofInBounds(srcPos, srcSize)) {
        candidate = flowSrc[ofFlatIndex(srcPos, srcSize)];
    }
    candidateVectors[localId.z][localId.y][localId.x] = candidate * 2;

    let lumaOrigin = vec2<i32>(globalId.x * COMPARE_WIDTH, globalId.y * COMPARE_ROWS);
    for (var row = localId.z; row < COMPARE_ROWS; row += CANDIDATE_COUNT) {
        candidateLuma[row][localId.y][localId.x] = ofLoadFirstPackedLuma(lumaOrigin + vec2<i32>(0, row));
    }
    workgroupBarrier();

    var sad = 0u;
    for (var row = 0; row < COMPARE_ROWS; row++) {
        let matchPos = lumaOrigin + vec2<i32>(0, row) + candidate;
        sad += ofSad(candidateLuma[row][localId.y][localId.x], ofLoadSecondPackedLuma(matchPos));
    }
    candidateSads[localId.z][localId.y][localId.x] = sad;
    workgroupBarrier();

    if (localId.z != 0) { return; }

    var bestSad = 0xffffffffu;
    var bestCandidate = 0;
    for (var i = 0; i < CANDIDATE_COUNT; i++) {
        if (candidateSads[i][localId.y][localId.x] < bestSad) {
            bestSad = candidateSads[i][localId.y][localId.x];
            bestCandidate = i;
        }
    }

    storeFlow(globalId.xy, candidateVectors[bestCandidate][localId.y][localId.x], dstSize);
}
