// The example network for the frame-interpolation module's neural extension
// point: a trained 3-8-8-1 per-pixel MLP that overwrites `blendWeight` with its
// own verdict, replacing the classical occlusion formula in
// preliminary_blend.wgsl. Original work, not a port of AMD code.
//
// It exists to prove the seam is real. Read the FROZEN CONTRACT comment at
// blendWeight's binding in preliminary_blend.wgsl for what the buffer means and
// what may write it; this pass is exactly the "optional replacement pass" that
// contract describes, and it changes none of the files it plugs between.
//
// FEATURE CONTRACT. Three scalars per pixel, all in 0..1, read straight from
// buffers the pipeline already has — no packing pass, no extra resources:
//
//   f0  disocclusionMask.x   surface visible in the previous frame (binarised)
//   f1  disocclusionMask.y   surface visible in the current frame  (binarised)
//   f2  luma(preliminaryColor), Rec.709, saturated
//
// The first two are what the classical formula sees. The third is what it
// ignores, and is why this network can distrust a bright pixel that only one
// frame saw. test/neural-reference.ts holds the same three features and the
// objective the weights were fitted to; that file is the CPU reference this
// kernel is verified against.
//
// The colour feature is the backbuffer's own encoding (linear, PQ or scRGB per
// `backbufferTransferFunction`) rather than a linearised luminance: the weights
// were trained on 0..1 samples, so what matters is that the feature is bounded
// and monotonic in brightness, not that it is photometric.
//
// SCENE CUTS. No handling here, deliberately. On the first frame after a reset
// the mask is meaningless, but final_blend.wgsl ignores blendWeight entirely on
// that frame -- it re-tests the frame index precisely because a replacement
// pass may have rewritten the weight -- so there is nothing this pass could
// usefully do that is not already done downstream.
//
// Dispatch: (ceil(W/8), ceil(H/8)) at render resolution, after
// preliminary_blend.wgsl and before the inpainting pyramid.

// @import ./primitives.wgsl

// A byte-compatible prefix of FrameInterpolationParams (its first field is the
// same vec2<i32>), so orchestration can bind the uniform buffer it already has
// instead of allocating one for this pass. Declared separately rather than
// imported: importing the frame-interpolation params would make this directory
// depend on that module, and a consumer's own network should not inherit FSR3's
// uniform.
struct NeuralBlendParams {
    renderSize: vec2<i32>,
}

@group(0) @binding(0) var<uniform> params: NeuralBlendParams;
// binding(1) is the weight buffer, declared by the primitives above.
@group(0) @binding(2) var<storage, read> disocclusionMask: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> preliminaryColor: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> blendWeight: array<f32>;

const MLP_INPUTS: u32 = 3u;
const MLP_HIDDEN: u32 = 8u;
const MLP_OUTPUTS: u32 = 1u;

// Layer offsets per the weight-buffer layout in primitives.wgsl: each layer
// spans outputs * (inputs + 1) floats, and the next starts where it ends.
const LAYER0_OFFSET: u32 = 0u;
const LAYER1_OFFSET: u32 = LAYER0_OFFSET + MLP_HIDDEN * (MLP_INPUTS + 1u);
const LAYER2_OFFSET: u32 = LAYER1_OFFSET + MLP_HIDDEN * (MLP_HIDDEN + 1u);

const REC709_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn loadFeatures(index: u32) -> NnVector {
    let mask = disocclusionMask[index];
    let luma = saturate(dot(preliminaryColor[index].rgb, REC709_LUMA));

    var features = nnZeroVector(MLP_INPUTS);
    features.values[0] = mask.x;
    features.values[1] = mask.y;
    features.values[2] = luma;
    return features;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let pos = vec2<i32>(globalId.xy);
    if (any(pos >= params.renderSize)) { return; }

    let index = u32(pos.y * params.renderSize.x + pos.x);

    let hidden0 = nnRelu(nnLinear(loadFeatures(index), LAYER0_OFFSET, MLP_HIDDEN));
    let hidden1 = nnRelu(nnLinear(hidden0, LAYER1_OFFSET, MLP_HIDDEN));
    let output = nnLinear(hidden1, LAYER2_OFFSET, MLP_OUTPUTS);

    // Sigmoid is what keeps the contract's 0..1 range an invariant of the
    // network rather than a clamp applied after it. Unlike the classical
    // formula's exact 0 and 1 it cannot quite reach the ends, so a hole keeps
    // a sliver of its own colour as coverage in the inpainting pyramid, and a
    // pixel both frames see is not bit-exactly excluded from inpainting. At
    // these weights that is 3e-4 and 2e-3 of the range respectively.
    blendWeight[index] = nnSigmoid(output.values[0]);
}
