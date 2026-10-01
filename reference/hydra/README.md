# Recovered SDKCore BASIC interpolation

This is an experimental WebGPU reconstruction of the supplied-camera-transform
path in `libFrameGenerationSdkCore.so` (Hydra, algorithm 0), selected through
Huawei's `libframegeneration.so` wrapper. It is not the FrameFlow Gridwarp
extrapolator or an implementation recovered from AMD FSR.

The adjacent GLSL files are unchanged review copies from the user's
`webge-framegeneration-analysis-20260930/framegeneration-basic-vulkan/interpolation`
bundle. They were recovered from embedded Vulkan SPIR-V, not published vendor
source. The original binary's SHA-256 is
`59364a6aba3263577a61e787a03dd05ab16d5d41749bf9bfdfd23b9470cce7ec`.
The bundle's generic variables and reconstructed names are evidence aids.
Original author identifiers and an open-source license were not recovered.

The application executes the hand-cleaned WGSL in `src/wgsl/hydra`.
See the [algorithm overview](../../src/wgsl/hydra/README.md) for a walkthrough of
the shader stages and their blend logic.
`scripts/port-hydra.py` can regenerate mechanical reference translations with
`glslangValidator` and `naga`; these go to ignored `reference/hydra/translated`.
The script deliberately does not overwrite the cleaned shaders.

## Source mapping

Offsets below are original ELF file offsets, not native function addresses.

| WGSL | Recovered program / offset | Purpose |
|---|---|---|
| `fullscreen` | shared vertex `0x3cec0`; resolve vertex `0x3d2bc` | Fullscreen triangle, common orientation |
| `luma` | compute_luma_no_ui `0x3dd2c` | Previous/current luminance |
| `depth_input` | split_zs_blit `0x3d7ac`; depth_mipmap_conversion `0x3e674`; depth_mipmap_internal `0x3e000` | Float-depth adapter and average reduction |
| `search` | hydra_search_block_grad(false) `0x46ffc` | Coarse-to-fine half-displacement search |
| `match` | hydra_match_txt(false)_dmatch(none) `0x4c5e0` | Optical-flow error and source-pixel usage |
| `reproject_vertex` / `reproject_fragment` | hydra_reproject_distort `0x434b4` / `0x4337c` | Two depth-aware reprojection meshes |
| `reproject_match` | hydra_reproject_match `0x43fac` | Reprojection error |
| `reproject_blur` | hydra_reproject_blur `0x44850` | Separable error filter |
| `resolve` | hydra_resolve_io_reproject `0x4f054` | Midpoint color and fallback blend |
| `mipmap` | Native Vulkan image blits | Portable filtered mip generation |
| `usage_masks` | WebGPU adaptation | Expand atomic flags into unused-pixel masks |

[Pass graph](passes.svg) ([D2 source](passes.d2), rendered with TALA).

## Naming and simplification

WGSL names describe the recovered mathematics rather than retaining SPIR-V IDs.
The resolve's `float16_tValue174` is `opticalWeight`, directly initialized with
its clamp expression. `f16vec2Value153` becomes `fallbackWeights`,
`float16_tValue159` becomes `fallbackSum`, and `sampledValue100` becomes
`flowUvOffset`. The search uses `coarseCandidates`, `refinement`,
`absoluteDifference`, and `rankedCandidates`. Redundant copies, mutable
single-assignment locals, phi-resolution scaffolding, and disabled specialization
branches are removed. Loops, accumulators, and actual conditional updates remain
mutable. This is primarily copy propagation and control-flow simplification;
frame-dependent values are `let`, not compile-time `const`.

## Input and scheduling contract

`HydraFrameGenerator` accepts full-resolution previous/current linear LDR color
views, plus a float texture containing current device depth in its R channel.
Color must be filterable (for example `rgba8unorm` or `rgba16float`). The depth
input can be `r32float`, uses standard WebGPU [0, 1] device depth, and is a color
texture, not a `texture_depth_2d` attachment. All inputs are single-sampled and
the configured size. Motion vectors are not consumed.

Supply both column-major `previousToCurrentClip` and `currentToPreviousClip`
transforms for a moving camera. They map homogeneous WebGPU clip positions,
including depth, between the two real frames. Omission means an identity
transform (stationary camera), not automatic estimation. Previous depth is
retained internally; supply `previousDepth` explicitly for discontinuous pairs,
or `resetHistory: true` to initialize both depth histories from current depth.
After a camera cut, present a real frame and start a new continuous pair; this
port does not include a scene-cut detector.

Call `configure`, then `prepare` and `dispatch` for each real-frame pair.
`dispatch` returns an opaque `rgba16float` midpoint at t=0.5. The generator owns
and reuses that texture. Reconfiguration and destruction invalidate it. Inputs
must remain valid through dispatch and command submission. In the demo, the
refresh that renders frame n presents real frame n−1; the next refresh presents
the midpoint. Switching algorithms waits for a fresh pair before showing output.

## Adaptations and confidence

The shaders and pass connections are recovered, but the full native scheduler
has not been replayed. This port fixes the recovered shader defaults: 8×8 blocks,
5×5×9 search workgroups, average depth reduction, full UV scales, no orientation
flip, and combined optical-flow/reprojection resolve. The host uses a half-size
luminance pyramid capped at five levels. Native runtime specialization values,
resource sizes, temporal aliases, and startup policy are not all established.
The search uses linear coarse sampling, consistent with native search Init
`0x1eaf00`'s `sampler_compute_linear` binding (Ghidra address space).

Float16/int16 intermediates are widened to 32 bits to avoid optional WebGPU
features. Push constants become uniform buffers; clip Y is converted explicitly;
depth packing/stencil transport becomes float-depth loading with D24 quantization;
depth mesh sampling uses clamped nearest loads. Native image blits become render
passes. Search shared-memory seed writes have one writer per candidate, and
overlapping fragment mask writes use atomics followed by a conversion pass.
These are deliberate portability changes, not a claim of bitwise native parity.

FFT phase correlation and automatic transform-estimation fallback are omitted.
The former is disabled by the SDK's default; the latter is bypassed by WebGE's
supplied-matrix path. HDR, UI layers, MSAA, alternative search variants, and
extrapolation are outside this implementation.

## Verification and known quality gap

`npm run build` validates the WGSL and TypeScript. `npm test` runs actual Dawn GPU
tests including static odd-size images, four motion directions, clip/depth
reprojection, and depth-history reset. `npm run quality` renders deterministic
ground truth for both backends and saves PNGs, per-frame metrics and an HTML
report under `artifacts/quality`.

The thin, fast-moving marker remains a known failing quality case for Hydra.
Passing shader validation and the focused GPU tests does not mean that this
quality test passes, or establish equivalence with the original Huawei binary.
The comparison also includes FSR3 without game motion vectors and a crossfade
baseline to make the difference in input information visible.
