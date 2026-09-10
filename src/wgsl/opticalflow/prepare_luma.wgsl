// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// ffx_opticalflow_prepare_luma.h: backbuffer colour -> level-0 optical-flow
// luma, quantised to 0..255. Every later optical-flow pass reads this, never
// the colour texture.
//
// Dispatch: (ceil(ceil(W/2)/16), ceil(ceil(H/2)/16)) — each invocation covers
// a 2x2 pixel quad of the render-resolution input.

// @import ./common.wgsl
// @import ./params.wgsl
// @import ../core/math.wgsl

@group(0) @binding(1) var inputColor: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> lumaOut: array<u32>;

const TRANSFER_FUNCTION_LINEAR_LDR: u32 = 0u;
const TRANSFER_FUNCTION_PQ: u32 = 1u;
const TRANSFER_FUNCTION_SCRGB: u32 = 2u;

const REC709_LUMA_WEIGHTS = vec3<f32>(0.2126, 0.7152, 0.0722);
const REC2020_LUMA_WEIGHTS = vec3<f32>(0.2627, 0.678, 0.0593);

// CIE L* lightness curve: linear below the L* knee, cube-root above.
const CIE_L_KNEE: f32 = 216.0 / 24389.0;
const CIE_L_LINEAR_SLOPE: f32 = 24389.0 / 27.0;
const CIE_L_SCALE: f32 = 116.0;
const CIE_L_OFFSET: f32 = 16.0;
// L* spans 0..100; the optical flow wants a roughly unit-ranged signal.
const CIE_L_TO_UNIT: f32 = 0.01;

// PQ encodes absolute nits with a 10000 nit peak; scRGB is scaled so 1.0 is
// 80 nits. Both are renormalised against the display's own peak luminance.
const PQ_PEAK_NITS: f32 = 10000.0;
const SCRGB_NITS_PER_UNIT: f32 = 80.0;

const LUMA_QUANT_SCALE: f32 = 255.0;
const PIXELS_PER_THREAD: i32 = 2;

fn perceivedLuminance(luminance: f32) -> f32 {
    if (luminance <= CIE_L_KNEE) {
        return luminance * CIE_L_LINEAR_SLOPE * CIE_L_TO_UNIT;
    }

    return (pow(luminance, 1.0 / 3.0) * CIE_L_SCALE - CIE_L_OFFSET) * CIE_L_TO_UNIT;
}

fn scRgbToLinear(value: vec3<f32>, minLuminance: f32, maxLuminance: f32) -> vec3<f32> {
    let p = value - vec3<f32>(minLuminance / SCRGB_NITS_PER_UNIT);
    return p / vec3<f32>((maxLuminance - minLuminance) / SCRGB_NITS_PER_UNIT);
}

fn sceneLuma(color: vec3<f32>) -> f32 {
    let transferFunction = params.backbufferTransferFunction;

    if (transferFunction == TRANSFER_FUNCTION_LINEAR_LDR) {
        return dot(color, REC709_LUMA_WEIGHTS);
    }

    if (transferFunction == TRANSFER_FUNCTION_PQ) {
        let linear = ffxLinearFromPQ(color) * (PQ_PEAK_NITS / params.minMaxLuminance.y);
        return perceivedLuminance(dot(linear, REC2020_LUMA_WEIGHTS));
    }

    if (transferFunction == TRANSFER_FUNCTION_SCRGB) {
        let linear = scRgbToLinear(color, params.minMaxLuminance.x, params.minMaxLuminance.y);
        return perceivedLuminance(dot(linear, REC709_LUMA_WEIGHTS));
    }

    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let size = ofLumaLevelSize(0u);
    let base = vec2<i32>(globalId.xy) * PIXELS_PER_THREAD;

    for (var y = 0; y < PIXELS_PER_THREAD; y++) {
        for (var x = 0; x < PIXELS_PER_THREAD; x++) {
            let pos = base + vec2<i32>(x, y);
            if (!ofInBounds(pos, size)) { continue; }

            // AMD's R8_UINT store bounds this for free. Here the buffer is
            // u32, and both consumers assume a byte: ofPackLuma() would spill
            // an overbright pixel into its neighbours' lanes, and the SCD
            // histogram would bin it out of range. HDR input above the
            // declared peak luminance reaches this.
            let color = textureLoad(inputColor, pos, 0).rgb;
            lumaOut[ofFlatIndex(pos, size)] = u32(saturate(sceneLuma(color)) * LUMA_QUANT_SCALE);
        }
    }
}
