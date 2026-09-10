// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_compute_luminance_pyramid.h. AMD generates the whole luma
// pyramid in one SPD dispatch, which depends on wave intrinsics and a
// globally-coherent atomic counter to know when a tile's neighbours are ready.
// WebGPU has neither reliably, so this is the iterative replacement: one
// dispatch per destination level, each reading the level above it.
//
// Only SpdReduce4's operator carries over — a plain 2x2 box average, then
// truncated by AMD's R8_UINT store. Taps outside the source level read 0, as
// SPD's texture loads do on odd-sized levels.
//
// Dispatch, per level 1..pyramidLevelCount-1 with params.pyramidLevel set to
// the destination level: (ceil(dstW/8), ceil(dstH/8)).

// @import ./common.wgsl
// @import ./params.wgsl

@group(0) @binding(1) var<storage, read> lumaSrc: array<u32>;
@group(0) @binding(2) var<storage, read_write> lumaDst: array<u32>;

const DOWNSAMPLE_FACTOR: i32 = 2;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let dstSize = ofLumaLevelSize(params.pyramidLevel);
    let dstPos = vec2<i32>(globalId.xy);
    if (!ofInBounds(dstPos, dstSize)) { return; }

    let srcSize = ofLumaLevelSize(params.pyramidLevel - 1u);
    let srcPos = dstPos * DOWNSAMPLE_FACTOR;

    var sum = 0.0;
    for (var y = 0; y < DOWNSAMPLE_FACTOR; y++) {
        for (var x = 0; x < DOWNSAMPLE_FACTOR; x++) {
            let tap = srcPos + vec2<i32>(x, y);
            if (!ofInBounds(tap, srcSize)) { continue; }

            sum += f32(lumaSrc[ofFlatIndex(tap, srcSize)]);
        }
    }

    lumaDst[ofFlatIndex(dstPos, dstSize)] = u32(sum * 0.25);
}
