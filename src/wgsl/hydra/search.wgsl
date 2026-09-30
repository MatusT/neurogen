// SDKCore hydra_search_block_grad(false), 0x46ffc.
// One workgroup evaluates 9 coarse seeds x 25 half-pixel refinements for an 8x8 block.
// XY is half-displacement in this luminance level's pixels; Z carries seed validity.
struct SearchParams {
    nativePassIndex: u32, // Retained ABI field; unused by this shader variant.
    scaleCoarseCoordinates: u32,
    currentUvScale: f32,
    previousUvScale: f32,
}
@group(0) @binding(0) var coarseSampler: sampler;
@group(1) @binding(0) var previousLuma: texture_2d<f32>;
@group(1) @binding(1) var currentLuma: texture_2d<f32>;
@group(1) @binding(2) var coarseMotion: texture_2d<f32>;
@group(1) @binding(3) var outputMotion: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<uniform> params: SearchParams;
var<workgroup> coarseCandidates: array<vec3<f32>, 9>;
var<workgroup> rankedCandidates: array<vec2<i32>, 256>;
const SEED_OFFSETS = array(vec2(0.25), vec2(1.25, 0.25), vec2(-0.75, 0.25),
    vec2(0.25, 1.25), vec2(0.25, -0.75), vec2(1.25), vec2(-0.75),
    vec2(-0.75, 1.25), vec2(1.25, -0.75));

@compute @workgroup_size(5, 5, 9)
fn main(@builtin(workgroup_id) block: vec3<u32>, @builtin(local_invocation_id) lane: vec3<u32>,
        @builtin(local_invocation_index) laneIndex: u32) {
    let coarseSize = vec2<f32>(textureDimensions(coarseMotion));
    let coarsePosition = vec2<f32>(block.xy) * 0.5 + SEED_OFFSETS[lane.z];
    var coarseUv = clamp(coarsePosition, vec2(0.0), coarseSize * params.currentUvScale) / coarseSize;
    if (params.scaleCoarseCoordinates != 0u) {
        coarseUv = coarsePosition / (coarseSize * params.currentUvScale);
    }
    let seed = round(textureSampleLevel(coarseMotion, coarseSampler, coarseUv, 0.0).xyz * vec3(2.0, 2.0, 1.0));
    if (lane.x == 0u && lane.y == 0u) { coarseCandidates[lane.z] = seed; }
    let refinement = vec2<f32>(lane.xy) * 0.5 - 0.5;
    let texelSize = 1.0 / vec2<f32>(textureDimensions(previousLuma));
    let previousOffset = (vec2(1.0) - seed.xy - 2.0 * refinement) * texelSize;
    let currentOffset = (seed.xy + 1.0) * texelSize;
    var absoluteDifference = 0.0;
    for (var x = 0u; x < 4u; x++) {
        for (var y = 0u; y < 4u; y++) {
            let pixel = vec2<f32>(block.xy * 8u + vec2(x, y) * 2u);
            let a = textureGather(0, previousLuma, coarseSampler,
                clamp(pixel * texelSize + previousOffset, vec2(0.0), vec2(params.previousUvScale)));
            let b = textureGather(0, currentLuma, coarseSampler,
                clamp(pixel * texelSize + currentOffset, vec2(0.0), vec2(params.currentUvScale)));
            let difference = abs(a - b);
            let pairs = difference.xy + difference.zw;
            absoluteDifference += pairs.x + pairs.y;
        }
    }
    rankedCandidates[laneIndex] = vec2(i32(3072.0 * (absoluteDifference + 0.01 * length(refinement))), i32(laneIndex));
    if (laneIndex < 31u) { rankedCandidates[laneIndex + 225u] = vec2(107374182); }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (laneIndex < stride) {
            let left = rankedCandidates[laneIndex];
            let right = rankedCandidates[laneIndex + stride];
            rankedCandidates[laneIndex] = select(right, left, left.x < right.x);
        }
        workgroupBarrier();
    }
    if (laneIndex == 0u) {
        let winner = u32(rankedCandidates[0].y);
        let localWinner = winner % 25u;
        let correction = vec2<f32>(f32(localWinner % 5u), f32(localWinner / 5u)) * 0.5 - 0.5;
        textureStore(outputMotion, vec2<i32>(block.xy), vec4(coarseCandidates[winner / 25u] + vec3(correction, 0.0), 0.0));
    }
}
