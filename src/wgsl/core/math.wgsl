// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// Most ffx_core scalar/vector helpers (ffxMin, ffxMax, ffxPow, ffxSaturate,
// ffxLerp, ffxRound, ffxFract, ffxBroadcastN) map 1:1 onto native WGSL
// builtins with identical semantics — call these directly instead, no
// wrapper is defined here (WGSL has no user-level overloading, so a
// per-type wrapper would just be a same-arity alias for the builtin):
//
//   ffxMin/ffxMax     -> min(x, y) / max(x, y)
//   ffxPow            -> pow(x, y)
//   ffxSaturate       -> saturate(x)
//   ffxLerp           -> mix(x, y, t)
//   ffxRound          -> round(x)
//   ffxFract          -> fract(x)
//   ffxBroadcastN(v)  -> vecN<T>(v)

// IEEE 754 binary16 max finite value; FSR3 uses 1/this as a safe tonemap
// denominator epsilon.
const FFX_FP16_MAX: f32 = 65504.0;
const FFX_TONEMAP_EPSILON: f32 = 1.0 / FFX_FP16_MAX;

// SMPTE ST 2084 (PQ) EOTF constants.
const PQ_C1: f32 = 0.835938;
const PQ_C2: f32 = 18.8516;
const PQ_C3: f32 = 18.6875;
const PQ_INV_M2: f32 = 0.0126833;
const PQ_INV_M1: f32 = 6.27739;

// Only the vec3 (RGB) form is used by the optical-flow / frame-interpolation
// passes; the ffx_core scalar and vec2 overloads are unused and skipped.
fn ffxLinearFromPQ(value: vec3<f32>) -> vec3<f32> {
    let p = pow(value, vec3<f32>(PQ_INV_M2));
    return pow(saturate(p - vec3<f32>(PQ_C1)) / (vec3<f32>(PQ_C2) - vec3<f32>(PQ_C3) * p), vec3<f32>(PQ_INV_M1));
}

// IEC 61966-2-1 (sRGB) EOTF constants.
const SRGB_THRESHOLD: f32 = 0.04045;
const SRGB_LINEAR_SCALE: f32 = 1.0 / 12.92;
const SRGB_GAMMA: f32 = 2.4;
const SRGB_A: f32 = 1.0 / 1.055;
const SRGB_B: f32 = 0.055 / 1.055;

// HLSL source picks the piecewise branch via a sign-bit multiply trick
// (ffxZeroOneSelect + ffxZeroOneIsSigned) to stay branchless on GCN/RDNA.
// WGSL's select() is already a branchless select instruction, so it's used
// directly instead of porting those two helpers.
fn ffxLinearFromSrgb(value: vec3<f32>) -> vec3<f32> {
    return select(
        pow(value * SRGB_A + SRGB_B, vec3<f32>(SRGB_GAMMA)),
        value * SRGB_LINEAR_SCALE,
        value < vec3<f32>(SRGB_THRESHOLD)
    );
}
