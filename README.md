# neurogen

WebGPU frame generation library. Ports AMD FidelityFX FSR3's open-source
frame interpolation pipeline (optical flow + motion vector reconstruction +
disocclusion masking + inpainting + blend) from HLSL to WGSL, and adds a
reusable layer of small neural-inference WGSL primitives (conv/linear,
activations, weight buffers) as a pluggable refinement stage.

AMD's actual ML frame generation (FSR4) ships only as a closed signed
binary for DX12/Windows — not portable. See `THIRD_PARTY_NOTICES.md`.

## Layout

- `src/wgsl/core/` — shared WGSL math/packing helpers.
- `src/wgsl/opticalflow/`, `src/wgsl/frameinterpolation/` — ported FSR3 passes.
- `src/wgsl/neural/` — reusable neural-shader primitives and the pluggable
  refinement stage.
- `src/orchestration/` — `FrameGenerator`, the TS entry point.
- `demo/` — minimal WebGPU scene with a frame-generation on/off toggle.

## Build

```
npm install
npm run build   # validates WGSL (naga) + type-checks
npm test
npm run demo
```

Requires `naga` on PATH: `cargo install naga-cli`.

## Profiling the demo

Run `npm run demo`, open the demo in Firefox, and record a performance profile.
Select the demo's main thread and open **Stack Chart**. The work runs inside
explicitly named functions, so sampled stacks identify each phase (search for
`neurogen`):

- `neurogenFullRender` — scene update and full real-frame rendering.
- `neurogenInterpolation` — `prepare()` and `dispatch()` for the generated frame.
- `neurogenPresentRealFrame` — presenting a real frame, including repeats.
- `neurogenPresentInterpolatedFrame` — presenting a generated frame.

Stack Chart uses sampled function calls, so a short call can fall between
samples. Reduce the recording interval (for example, to 0.1 ms) for finer
sampling, at the cost of additional profiling overhead. See Firefox's
[profiling guidance](https://firefox-source-docs.mozilla.org/devtools/performance/performance.html#tweak-profiler-default-settings).

The **Marker Chart** also records every phase under **UserTiming**, with these
labels (search for `Neurogen:`):

- `Neurogen: Full render` — scene update and full real-frame rendering.
- `Neurogen: Interpolation` — `prepare()` and `dispatch()` for the generated frame.
- `Neurogen: Present real frame` — presenting a real frame, including repeats.
- `Neurogen: Present interpolated frame` — presenting a generated frame.

Full rendering and interpolation happen on the same refresh; the generated
frame is presented on the following refresh. Interpolation still runs with
frame generation toggled off to keep its history current.

These spans measure synchronous JavaScript work and GPU command submission.
GPU execution continues asynchronously, so the spans are not GPU timings.
Entries are cleared from the page's performance buffer after emission to avoid
accumulating them during long sessions; the profiler retains its recording.
See Firefox's [User Timing marker documentation](https://firefox-source-docs.mozilla.org/tools/profiler/instrumenting-javascript.html#markers-in-content-code).
