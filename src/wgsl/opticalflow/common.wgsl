// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_common.h plus the buffer-addressing guards this port needs.
// AMD keeps its optical-flow intermediates in textures, where an out-of-bounds
// load reads 0 and an out-of-bounds store is dropped, and the passes lean on
// that. WebGPU has no atomics on storage textures and no R8_UINT storage
// format, so the intermediates here are storage buffers and every access has
// to reproduce those texture semantics explicitly.

fn ofInBounds(pos: vec2<i32>, size: vec2<i32>) -> bool {
    return all(pos >= vec2<i32>(0)) && all(pos < size);
}

fn ofFlatIndex(pos: vec2<i32>, size: vec2<i32>) -> u32 {
    return u32(pos.y * size.x + pos.x);
}

// Packs four horizontally adjacent lumas into one u32, byte 0 leftmost. When
// the 4-wide window straddles a screen edge the caller has already clamped it
// back inside, so the samples are rotated and the outermost in-screen luma is
// replicated into the lanes that fell off screen.
fn ofPackLuma(width: i32, x: i32, luma0: u32, luma1: u32, luma2: u32, luma3: u32) -> u32 {
    var packed = luma0 | (luma1 << 8u) | (luma2 << 16u) | (luma3 << 24u);

    if (x < 0) {
        let filler = packed & 0xffu;
        if (x <= -1) { packed = (packed << 8u) | filler; }
        if (x <= -2) { packed = (packed << 8u) | filler; }
        if (x <= -3) { packed = (packed << 8u) | filler; }
        return packed;
    }

    if (x > width - 4) {
        let filler = packed & 0xff000000u;
        if (x >= width - 3) { packed = (packed >> 8u) | filler; }
        if (x >= width - 2) { packed = (packed >> 8u) | filler; }
        if (x >= width - 1) { packed = (packed >> 8u) | filler; }
    }

    return packed;
}

// Sum of absolute differences over the four byte lanes of two packed lumas.
fn ofSad(a: u32, b: u32) -> u32 {
    return u32(abs(i32(a & 0xffu) - i32(b & 0xffu))
             + abs(i32((a >> 8u) & 0xffu) - i32((b >> 8u) & 0xffu))
             + abs(i32((a >> 16u) & 0xffu) - i32((b >> 16u) & 0xffu))
             + abs(i32((a >> 24u) & 0xffu) - i32((b >> 24u) & 0xffu)));
}

// Four SADs of `b` against the 8-byte window (a0, a1) at byte offsets 0..3.
// The scalar stand-in for HLSL's msad4(), which WGSL has no equivalent of;
// AMD ships the same fallback for hardware without accelerated msad4.
fn ofQSad(firstWord: u32, secondWord: u32, b: u32) -> vec4<u32> {
    var a0 = firstWord;
    var a1 = secondWord;

    var sad: vec4<u32>;
    sad.x = ofSad(a0, b);

    a0 = (a0 >> 8u) | ((a1 & 0xffu) << 24u);
    a1 >>= 8u;
    sad.y = ofSad(a0, b);

    a0 = (a0 >> 8u) | ((a1 & 0xffu) << 24u);
    a1 >>= 8u;
    sad.z = ofSad(a0, b);

    a0 = (a0 >> 8u) | ((a1 & 0xffu) << 24u);
    sad.w = ofSad(a0, b);

    return sad;
}
