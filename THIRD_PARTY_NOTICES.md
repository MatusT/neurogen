# Third-party notices

`src/wgsl/core/`, `src/wgsl/opticalflow/`, and `src/wgsl/frameinterpolation/`
are WGSL ports of algorithms from AMD's FidelityFX SDK (FSR3 frame
interpolation), MIT licensed:

```
Copyright (C) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

Source: https://github.com/GPUOpen-Effects/FidelityFX-SDK,
`Kits/FidelityFX/api/internal/gpu/ffx_core_{hlsl,gpu_common}.h` (core helpers)
and `Kits/FidelityFX/framegeneration/fsr3/include/gpu/{opticalflow,frameinterpolation}`.

`src/wgsl/neural/` and the TypeScript orchestration layer are original work,
not derived from AMD source.
