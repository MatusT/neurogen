// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_compute_optical_flow_v5.h: the block-matching motion search.
// One workgroup of 64 invocations owns a 16x16 luma tile, i.e. a 2x2 patch of
// motion vector cells, and searches each cell's 8x8 block against a +-8 pixel
// window of the previous frame's luma. The vector already stored for a cell is
// the prediction handed down by the coarser level, so the window follows it.
//
// Cross-lane work is the part that does not survive the move to WebGPU. AMD
// reduces the 256 candidate SADs with ffxWaveMin over a wave64 (with a
// shared-memory fixup for wave32 parts) and sums the still-block SADs with
// ffxWaveSum. WebGPU subgroups are an optional feature, so both reductions run
// through workgroup memory with an explicit barrier between every step, in
// uniform control flow.
//
// Dispatch, per level, coarsest first: (ceil(W_level/16), ceil(H_level/16)).

// @import ./luma.wgsl

@group(0) @binding(3) var<storage, read_write> opticalFlow: array<vec2<i32>>;
@group(0) @binding(4) var<storage, read> sceneChange: OpticalFlowSceneChange;

// CompareSize: the block edge matched per motion vector, four lumas per u32.
const COMPARE_SIZE: i32 = 8;
const COMPARE_WORDS: i32 = COMPARE_SIZE / 4;
const SEARCH_RADIUS: i32 = 8;
const THREAD_COUNT: u32 = 64u;
// 2x2 motion vector cells per workgroup.
const BLOCK_COUNT: i32 = 2;

// The window spans the block plus the search radius on both sides.
const SEARCH_WIDTH: i32 = (COMPARE_SIZE + SEARCH_RADIUS * 2) / 4;
const SEARCH_HEIGHT: i32 = COMPARE_SIZE + SEARCH_RADIUS * 2;

// SAD occupies the top 16 bits of a packed candidate so a single min() picks
// the best match and the tie-break falls through to the coordinate bits.
const CANDIDATE_SAD_SHIFT: u32 = 16u;
const SEARCH_COORD_MASK: u32 = 0xfu;

var<workgroup> blockPixels: array<array<u32, COMPARE_WORDS>, COMPARE_SIZE>;
var<workgroup> searchWindow: array<u32, SEARCH_WIDTH * SEARCH_HEIGHT>;
var<workgroup> minScratch: array<u32, THREAD_COUNT>;
var<workgroup> sumScratch: array<u32, THREAD_COUNT>;
var<workgroup> sceneChanged: u32;

// Both reductions leave a trailing barrier so the scratch can be reused by the
// next cell without a caller-side barrier.
fn reduceMin(localIndex: u32, value: u32) -> u32 {
    minScratch[localIndex] = value;
    workgroupBarrier();

    for (var stride = THREAD_COUNT / 2u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            minScratch[localIndex] = min(minScratch[localIndex], minScratch[localIndex + stride]);
        }
        workgroupBarrier();
    }

    let result = minScratch[0];
    workgroupBarrier();
    return result;
}

fn reduceSum(localIndex: u32, value: u32) -> u32 {
    sumScratch[localIndex] = value;
    workgroupBarrier();

    for (var stride = THREAD_COUNT / 2u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            sumScratch[localIndex] += sumScratch[localIndex + stride];
        }
        workgroupBarrier();
    }

    let result = sumScratch[0];
    workgroupBarrier();
    return result;
}

fn loadFlow(pos: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    if (!ofInBounds(pos, size)) { return vec2<i32>(0); }

    return opticalFlow[ofFlatIndex(pos, size)];
}

fn storeFlow(pos: vec2<i32>, motionVector: vec2<i32>, size: vec2<i32>) {
    if (!ofInBounds(pos, size)) { return; }

    opticalFlow[ofFlatIndex(pos, size)] = motionVector;
}

// FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS: distance from the window centre sits
// above the raw coordinate, so equal-SAD candidates resolve to the one closest
// to the prediction instead of the top-left corner.
fn encodeSearchCoord(coord: vec2<i32>) -> u32 {
    let centreDistance = vec2<u32>(abs(coord - vec2<i32>(SEARCH_RADIUS)));
    return (centreDistance.y << 12u) | (centreDistance.x << 8u) | (u32(coord.y) << 4u) | u32(coord.x);
}

fn decodeSearchCoord(bits: u32) -> vec2<i32> {
    let coord = vec2<i32>(i32(bits & SEARCH_COORD_MASK), i32((bits >> 4u) & SEARCH_COORD_MASK));
    return coord - vec2<i32>(SEARCH_RADIUS);
}

fn packCandidate(sad: u32, searchId: vec2<i32>, lane: i32) -> u32 {
    return (sad << CANDIDATE_SAD_SHIFT) | encodeSearchCoord(vec2<i32>(searchId.x * 4 + lane, searchId.y));
}

fn loadSearchWindow(localIndex: u32, windowCentre: vec2<i32>) {
    let base = windowCentre - vec2<i32>(SEARCH_RADIUS);

    for (var id = i32(localIndex); id < SEARCH_WIDTH * SEARCH_HEIGHT; id += i32(THREAD_COUNT)) {
        let cell = vec2<i32>(id % SEARCH_WIDTH, id / SEARCH_WIDTH);
        searchWindow[id] = ofLoadSecondPackedLuma(base + vec2<i32>(cell.x * 4, cell.y));
    }

    workgroupBarrier();
}

// SADs of this invocation's block against four horizontally adjacent
// candidate offsets.
fn candidateSads(searchId: vec2<i32>) -> vec4<u32> {
    var sad = vec4<u32>(0u);

    for (var dy = 0; dy < COMPARE_SIZE; dy++) {
        let row = (searchId.y + dy) * SEARCH_WIDTH + searchId.x;
        let a0 = searchWindow[row];
        let a1 = searchWindow[row + 1];
        let a2 = searchWindow[row + 2];

        sad += ofQSad(a0, a1, blockPixels[dy][0]);
        sad += ofQSad(a1, a2, blockPixels[dy][1]);
    }

    return sad;
}

struct ThreadMapping {
    // Column of four pixels (0..3) and row (0..15) within the workgroup's tile.
    searchId: vec2<i32>,
    pxPos: vec2<i32>,
    // Which of the 2x2 motion vector cells this invocation contributes to.
    cellId: i32,
}

fn mapThreads(groupId: vec2<i32>, localIndex: u32) -> ThreadMapping {
    var mapping: ThreadMapping;
    mapping.searchId = vec2<i32>(i32(extractBits(localIndex, 0u, 2u)), i32(extractBits(localIndex, 2u, 4u)));
    mapping.cellId = i32(extractBits(localIndex, 1u, 1u) | (extractBits(localIndex, 5u, 1u) << 1u));
    mapping.pxPos = (groupId << vec2<u32>(4u)) + mapping.searchId * vec2<i32>(4, 1);
    return mapping;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) groupIdIn: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {
    let groupId = vec2<i32>(groupIdIn.xy);
    let mapping = mapThreads(groupId, localIndex);
    let flowSize = ofFlowLevelSize(params.pyramidLevel);

    // A storage load is not workgroup-uniform, and the barriers further down
    // must stay in uniform control flow, so the flag is republished through
    // workgroupUniformLoad before anything branches on it.
    if (localIndex == 0u) { sceneChanged = sceneChange.detected; }
    if (workgroupUniformLoad(&sceneChanged) != 0u) {
        // Exactly the four invocations that map one-to-one onto the cells.
        if ((mapping.searchId.y & 7) == 0 && (mapping.searchId.x & 1) == 0) {
            storeFlow(mapping.pxPos >> vec2<u32>(3u), vec2<i32>(0), flowSize);
        }
        return;
    }

    let usePrediction = params.pyramidLevel != params.pyramidLevelCount - 1u;
    let packedLuma = ofLoadFirstPackedLuma(mapping.pxPos);
    let stillSad = ofSad(packedLuma, ofLoadSecondPackedLuma(mapping.pxPos));

    let cellOrigin = groupId << vec2<u32>(1u);
    let tileOrigin = groupId << vec2<u32>(4u);

    for (var cellY = 0; cellY < BLOCK_COUNT; cellY++) {
        for (var cellX = 0; cellX < BLOCK_COUNT; cellX++) {
            let cell = vec2<i32>(cellX, cellY);
            let cellPos = cellOrigin + cell;

            var prediction = loadFlow(cellPos, flowSize);
            if (!usePrediction) { prediction = vec2<i32>(0); }

            if (mapping.cellId == cellY * BLOCK_COUNT + cellX) {
                blockPixels[mapping.searchId.y & 7][mapping.searchId.x & 1] = packedLuma;
            }

            loadSearchWindow(localIndex, tileOrigin + cell * COMPARE_SIZE + prediction);

            let qsad = candidateSads(mapping.searchId);

            // AMD stages these four candidates through a groupshared sad map
            // before reducing, but every invocation reads back only the four
            // entries it just wrote — the map is indexed by the writer's own
            // searchId, which is unique across the 64 invocations — so the
            // round trip is dropped and the lane minimum formed directly.
            let laneMin = min(
                min(packCandidate(qsad.x, mapping.searchId, 0), packCandidate(qsad.y, mapping.searchId, 1)),
                min(packCandidate(qsad.z, mapping.searchId, 2), packCandidate(qsad.w, mapping.searchId, 3)));
            let bestCandidate = reduceMin(localIndex, laneMin);

            var motionVector = prediction + decodeSearchCoord(bestCandidate);

            // Local-search fallback: at the finest level, a block that already
            // matches where it sits at least as well as anywhere in the window
            // is held still rather than dragged onto a false match.
            let cellStillSad = reduceSum(localIndex, select(0u, stillSad, mapping.cellId == cellY * BLOCK_COUNT + cellX));
            if (params.pyramidLevel == 0u && cellStillSad <= (bestCandidate >> CANDIDATE_SAD_SHIFT)) {
                motionVector = vec2<i32>(0);
            }

            // Both reductions broadcast, and the prediction came from a single
            // address, so motionVector is workgroup-uniform: one invocation
            // stores it instead of racing 64 identical writes.
            if (localIndex == 0u) {
                storeFlow(cellPos, motionVector, flowSize);
            }
        }
    }
}
