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
