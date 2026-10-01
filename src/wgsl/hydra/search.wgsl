// SDKCore hydra_search_block_grad(false), 0x46ffc.
// Coarse-to-fine optical flow: one workgroup searches an 8x8 luminance block.
// Each of its 225 lanes evaluates one of 9 seeds x 25 half-pixel refinements.
// Output XY is half-displacement in this luminance level's pixels: matching
// samples previous at pixel - XY and current at pixel + XY. Z carries seed
// validity; W is unused. The host starts the coarsest level with (0, 0, 1, 0).
struct SearchParams {
    nativePassIndex: u32, // Retained ABI field; unused by this shader variant.
    scaleCoarseCoordinates: u32, // Select the recovered coarse-UV mapping variant.
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
// (Integer cost, lane index). Pad 225 real candidates to a power of two so all
// reduction steps use the same pairwise comparison. Lower cost wins.
var<workgroup> rankedCandidates: array<vec2<i32>, 256>;

// Center and eight neighbors in coarse motion texels. The quarter-texel phase
// and ordering are retained from the recovered search (ordering affects ties).
const SEED_OFFSETS = array(
    vec2(0.25),
    vec2(1.25, 0.25),
    vec2(-0.75, 0.25),
    vec2(0.25, 1.25),
    vec2(0.25, -0.75),
    vec2(1.25),
    vec2(-0.75),
    vec2(-0.75, 1.25),
    vec2(1.25, -0.75),
);

@compute @workgroup_size(5, 5, 9)
fn main(
    @builtin(workgroup_id) block: vec3<u32>,
    @builtin(local_invocation_id) lane: vec3<u32>,
    @builtin(local_invocation_index) laneIndex: u32,
) {
    // Map this block into the next coarser motion grid, then scale the seed's
    // displacement by two to express it in this finer luminance level's pixels.
    let coarseSize = vec2<f32>(textureDimensions(coarseMotion));
    let coarsePosition = vec2<f32>(block.xy) * 0.5 + SEED_OFFSETS[lane.z];
    var coarseUv = clamp(coarsePosition, vec2(0.0), coarseSize * params.currentUvScale) / coarseSize;
    if (params.scaleCoarseCoordinates != 0u) {
        coarseUv = coarsePosition / (coarseSize * params.currentUvScale);
    }
    let seed = round(
        textureSampleLevel(coarseMotion, coarseSampler, coarseUv, 0.0).xyz * vec3(2.0, 2.0, 1.0),
    );
    // Only one lane writes each shared seed; all 25 lanes use the same value.
    if (lane.x == 0u && lane.y == 0u) {
        coarseCandidates[lane.z] = seed;
    }

    // Search the recovered [-0.5, 1.5] half-motion refinement range per axis.
    // During scoring the current footprint stays fixed; the previous footprint
    // moves by twice the refinement, i.e. the corresponding full displacement.
    let refinement = vec2<f32>(lane.xy) * 0.5 - 0.5;
    let texelSize = 1.0 / vec2<f32>(textureDimensions(previousLuma));
    let previousOffset = (vec2(1.0) - seed.xy - 2.0 * refinement) * texelSize;
    let currentOffset = (seed.xy + 1.0) * texelSize;

    // Sixteen 2x2 gathers compare 64 luminance samples. Clamping also covers
    // partial edge blocks, so every lane reaches the workgroup barriers below.
    var absoluteDifference = 0.0;
    for (var x = 0u; x < 4u; x++) {
        for (var y = 0u; y < 4u; y++) {
            let pixel = vec2<f32>(block.xy * 8u + vec2(x, y) * 2u);
            let previousSamples = textureGather(
                0, previousLuma, coarseSampler,
                clamp(pixel * texelSize + previousOffset, vec2(0.0), vec2(params.previousUvScale)),
            );
            let currentSamples = textureGather(
                0, currentLuma, coarseSampler,
                clamp(pixel * texelSize + currentOffset, vec2(0.0), vec2(params.currentUvScale)),
            );
            let difference = abs(previousSamples - currentSamples);
            let pairs = difference.xy + difference.zw;
            absoluteDifference += pairs.x + pairs.y;
        }
    }

    // A small length penalty favors smaller corrections when image costs are
    // close. Preserve the recovered integer cost scale and padding sentinel.
    let cost = i32(3072.0 * (absoluteDifference + 0.01 * length(refinement)));
    rankedCandidates[laneIndex] = vec2(cost, i32(laneIndex));
    if (laneIndex < 31u) {
        rankedCandidates[laneIndex + 225u] = vec2(107374182);
    }
    workgroupBarrier();

    // Reduce to one winner. Equal costs select the right candidate; changing
    // this tie rule or tree order can change the motion chosen in flat regions.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (laneIndex < stride) {
            let left = rankedCandidates[laneIndex];
            let right = rankedCandidates[laneIndex + stride];
            rankedCandidates[laneIndex] = select(right, left, left.x < right.x);
        }
        workgroupBarrier();
    }

    if (laneIndex == 0u) {
        // Lane X varies fastest, followed by Y and seed index Z.
        let winner = u32(rankedCandidates[0].y);
        let localWinner = winner % 25u;
        let correction = vec2<f32>(f32(localWinner % 5u), f32(localWinner / 5u)) * 0.5 - 0.5;
        textureStore(
            outputMotion,
            vec2<i32>(block.xy),
            vec4(coarseCandidates[winner / 25u] + vec3(correction, 0.0), 0.0),
        );
    }
}
