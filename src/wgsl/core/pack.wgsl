// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffxAsUInt32 / ffxAsFloat and ffxUnpackF32 map 1:1 onto native WGSL
// builtins — call these directly, no wrapper is defined here:
//
//   ffxAsUInt32(x)  -> bitcast<u32>(x)   (also vec2/3/4<u32> forms)
//   ffxAsFloat(x)   -> bitcast<f32>(x)   (also vec2/3/4<f32> forms)
//   ffxUnpackF32(a) -> unpack2x16float(a)

// HLSL's f32tof16 (aliased as ffxF32ToF16) converts a single f32 to the bits
// of its binary16 representation, packed into the low 16 bits of a u32 with
// the high 16 bits zero. WGSL has no scalar equivalent — pack2x16float packs
// two f32 values at once — so the second lane is padded with 0.0, whose
// binary16 encoding is 0x0000, matching f32tof16's zeroed high bits.
fn ffxF32ToF16(x: f32) -> u32 {
    return pack2x16float(vec2<f32>(x, 0.0));
}
