// Reusable per-pixel neural primitives. Original work, not a port of AMD code:
// FSR4's network is a closed signed binary with no published shader source or
// weights, so there is nothing to port and this is what makes the library's
// neural extension point real.
//
// Scope: 1x1-convolution/linear layers, i.e. a network that maps one pixel's
// feature vector to one pixel's output with no spatial extent. No neighbour
// sampling, no shared memory, no transposes — which is why a layer here is a
// few lines rather than a tiled matmul.
//
// WEIGHT BUFFER LAYOUT. All of a network's layers live in one flat array<f32>
// bound at the reserved binding below. Layers are concatenated in evaluation
// order and each occupies
//
//   [offset,               offset + outputs * inputs)   row-major matrix,
//                                                       one row per output
//                                                       channel
//   [offset + outputs * inputs,  + outputs)             biases
//
// so a layer spans `outputs * (inputs + 1)` floats and the next layer's offset
// is this one's offset plus that span. A network declares those offsets as
// const expressions from its own shape; nothing else in the buffer is
// network-specific, which is what makes a different network a different asset
// rather than a different buffer format. src/wgsl/neural/weights.ts packs the
// asset into exactly this layout — the two must agree.
//
// BINDING CONVENTION. A pass built on these primitives keeps
// `group(0) binding(0)` for its uniform (the whole library's convention) and
// `group(0) binding(1)` for the weight buffer declared here, then numbers its
// own resources from 2. Swapping in another network is then a matter of binding
// a different weight buffer and dispatching a different entry point, with no
// rebinding of the pipeline's other resources.
//
// ACTIVATION PACKING. Intermediate activations never reach memory: a per-pixel
// network's hidden layers live in NnVector for the life of one invocation. A
// feature or output buffer that does cross memory is channel-interleaved,
// `(y * width + x) * channels + c`, matching how a texel's channels already sit
// together — but a network whose inputs are already in the pipeline's buffers
// (as the example one's are) reads them directly and needs no packing pass.

@group(0) @binding(1) var<storage, read> nnWeights: array<f32>;

// A layer's width bound. Fixed because WGSL has no dynamically sized locals,
// and small because these are per-pixel networks running once per pixel per
// frame: a wider layer wants a tiled matmul, not this. WGSL bounds-checks, so a
// network exceeding it does not corrupt anything — it silently computes with
// the wrong channels, which is worse to debug. Networks should assert their
// widths against this at validation time, as blend_weight_mlp.wgsl does.
const NN_MAX_CHANNELS: u32 = 8u;

struct NnVector {
    values: array<f32, NN_MAX_CHANNELS>,
    count: u32,
}

fn nnZeroVector(count: u32) -> NnVector {
    return NnVector(array<f32, NN_MAX_CHANNELS>(), count);
}

// One linear layer: `output = W * input + b`, unactivated. The input's own
// `count` is the matrix's column count, so a layer needs only where its weights
// start and how many channels come out.
fn nnLinear(input: NnVector, weightOffset: u32, outputs: u32) -> NnVector {
    // Copied into a var so the loop below can index it dynamically.
    var values = input.values;
    let biasOffset = weightOffset + outputs * input.count;

    var result = nnZeroVector(outputs);
    for (var o = 0u; o < outputs; o++) {
        var sum = nnWeights[biasOffset + o];
        for (var i = 0u; i < input.count; i++) {
            sum += nnWeights[weightOffset + o * input.count + i] * values[i];
        }

        result.values[o] = sum;
    }

    return result;
}

// Relu, chosen over gelu/tanh because it is exact: `max(x, 0)` cannot round
// differently here than in the CPU reference the kernel is verified against, so
// any disagreement between them is a real porting error rather than a
// transcendental's last bit.
fn nnRelu(input: NnVector) -> NnVector {
    var result = input;
    for (var i = 0u; i < input.count; i++) {
        result.values[i] = max(result.values[i], 0.0);
    }

    return result;
}

fn nnSigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}
