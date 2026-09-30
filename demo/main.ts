// The page: a scene rendered at half the display's refresh rate, and a toggle
// choosing what fills the refresh in between.
//
//   refresh 0   render real frame n, generate the frame between n-1 and n
//               present real frame n-1        (off: real frame n)
//   refresh 1   present the interpolated frame (off: real frame n again)
//
// Both paths produce real frames at the same rate. The toggle only decides
// whether the intervening refresh carries new motion or a repeat, which is the
// whole comparison — and why frame generation costs a frame of latency, since
// the frame between n-1 and n cannot exist before n has been rendered.

import { FrameGenerator } from "../src/index.js";
// The one thing that is not behind the entry point: `tsc` does not copy the
// weight asset into `dist/`, so the library documents it as a file a consumer
// bundler-imports or fetches.
import blendWeightAsset from "../src/wgsl/neural/weights/blend_weight_mlp.json";
import { GBuffer } from "./gbuffer.js";
import { Presenter } from "./present.js";
import { enabledFrom, FrameGen, shown, Shown } from "./schedule.js";
import { demoScene, PROJECTION, projectionFor } from "./scene.js";
import { formatMeasurement, measureMidpointTiming } from "./verify.js";

const RENDER_SIZE: readonly [number, number] = [1280, 720];
// Display refreshes per real frame.
const REAL_FRAME_PERIOD = 2;
const FPS_WINDOW_MS = 500;

// Synchronous CPU work, including GPU command encoding/submission. WebGPU
// executes asynchronously, so these spans do not measure GPU execution time.
// Callbacks have explicit names so sampled stacks also identify each phase.
function profile<T>(name: string, work: () => T): T {
  const start = performance.now();
  try {
    return work();
  } finally {
    performance.measure(name, { start, end: performance.now() });
    // The profiler records the measure when emitted. Release the page's copy
    // so an indefinitely running render loop does not accumulate entries.
    performance.clearMeasures(name);
  }
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`missing element #${id}`);
  }

  return found as T;
}

const status = element("status");

async function main(): Promise<void> {
  if (!navigator.gpu) {
    status.textContent = "This browser has no WebGPU. Try Chrome 113+ or Firefox 141+.";
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    status.textContent = "No WebGPU adapter available.";
    return;
  }

  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (event) => {
    const message = (event as GPUUncapturedErrorEvent).error.message;
    status.textContent = `WebGPU error: ${message}`;
    console.error(message);
  });

  const canvas = element<HTMLCanvasElement>("canvas");
  [canvas.width, canvas.height] = RENDER_SIZE;

  const context = canvas.getContext("webgpu");
  if (!context) {
    status.textContent = "Could not get a webgpu canvas context.";
    return;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  run(device, context, format);
}

function run(device: GPUDevice, context: GPUCanvasContext, format: GPUTextureFormat): void {
  const [width, height] = RENDER_SIZE;
  const projection = projectionFor(RENDER_SIZE);
  const gbuffer = new GBuffer(device, RENDER_SIZE, projection);
  const presenter = new Presenter(device, format);
  const generator = new FrameGenerator({ device });
  generator.configure({ ...PROJECTION, renderWidth: width, renderHeight: height });

  const hud = {
    framegen: element("hud-framegen"),
    neural: element("hud-neural"),
    real: element("hud-real"),
    presenting: element("hud-presenting"),
    fps: element("hud-fps"),
  };
  const framegenInput = element<HTMLInputElement>("framegen");
  const neuralButton = element<HTMLButtonElement>("neural");
  const verifyButton = element<HTMLButtonElement>("verify");
  const report = element("report");

  let mode = framegenInput.checked ? FrameGen.On : FrameGen.Off;
  let neural = false;
  let refresh = 0;
  let realFrame = -1;
  // Written by every real frame after the first, which is exactly when the
  // schedule can return Shown.Interpolated.
  let interpolated!: GPUTexture;
  let enabled = enabledFrom(realFrame);
  let paused = false;
  let windowStart = performance.now();
  let presentedInWindow = 0;
  let realInWindow = 0;

  framegenInput.addEventListener("change", () => {
    mode = framegenInput.checked ? FrameGen.On : FrameGen.Off;
    enabled = enabledFrom(realFrame);
    hud.framegen.textContent = mode === FrameGen.On ? "on" : "off — real frames held";
  });
  hud.framegen.textContent = mode === FrameGen.On ? "on" : "off — real frames held";
  hud.neural.textContent = "classical occlusion formula";

  neuralButton.addEventListener("click", () => {
    // The library keeps the installed network across a reconfigure and has no
    // uninstall, so this is a one-way switch rather than a checkbox.
    generator.installNeuralBlendWeight(blendWeightAsset);
    neural = true;
    neuralButton.disabled = true;
    hud.neural.textContent = `neural (${blendWeightAsset.name})`;
  });

  verifyButton.addEventListener("click", () => {
    paused = true;
    verifyButton.disabled = true;
    report.textContent = "measuring...";

    // Its own generator and render targets, so the live loop's history is not
    // disturbed and the scene it measures is the deterministic marker one —
    // but with whichever blendWeight writer is currently installed above, so
    // the figures describe the pipeline you are actually watching.
    measureMidpointTiming(device, {
      size: RENDER_SIZE,
      neuralWeights: neural ? blendWeightAsset : undefined,
    })
      .then((measurement) => {
        report.textContent = formatMeasurement(measurement);
      })
      .catch((error: unknown) => {
        report.textContent = `measurement failed: ${String(error)}`;
      })
      .finally(() => {
        // The measurement holds the loop for a second or so. Restart the rate
        // window with it, or the next one averages over the pause and reports
        // half the frame rate the demo is actually running at.
        windowStart = performance.now();
        presentedInWindow = 0;
        realInWindow = 0;
        paused = false;
        verifyButton.disabled = false;
      });
  });

  function tick(): void {
    requestAnimationFrame(tick);
    if (paused) {
      return;
    }

    const newRealFrame = refresh % REAL_FRAME_PERIOD === 0;
    if (newRealFrame) {
      realFrame++;
      profile("Neurogen: Full render", function neurogenFullRender() {
        gbuffer.renderFrame(demoScene(projection, realFrame));
      });
      realInWindow++;

      // Kept running whether or not the toggle is on: the module's scene-change
      // detector compares this frame against the last one it processed, so a
      // gap would read as a cut and spend its next sixteen frames recovering
      // just as the viewer switched frame generation back on.
      if (realFrame > 0) {
        interpolated = profile("Neurogen: Interpolation", function neurogenInterpolation() {
          generator.prepare(gbuffer.frameInputs());
          return generator.dispatch();
        });
      }
    }

    const what = shown({ mode, rendered: newRealFrame, realFrame, enabledFrom: enabled });
    if (what === Shown.Interpolated) {
      profile("Neurogen: Present interpolated frame", function neurogenPresentInterpolatedFrame() {
        presenter.present(context.getCurrentTexture().createView(), source(what).createView());
      });
    } else {
      profile("Neurogen: Present real frame", function neurogenPresentRealFrame() {
        presenter.present(context.getCurrentTexture().createView(), source(what).createView());
      });
    }
    hud.presenting.textContent = label(what);

    refresh++;
    presentedInWindow++;
    updateRates();
  }

  function source(what: Shown): GPUTexture {
    if (what === Shown.RealPrevious) {
      return gbuffer.previousColor;
    }

    return what === Shown.Interpolated ? interpolated : gbuffer.currentColor;
  }

  function label(what: Shown): string {
    if (what === Shown.Interpolated) {
      return `interpolated, between ${realFrame - 1} and ${realFrame}`;
    }

    return `real frame ${what === Shown.RealPrevious ? realFrame - 1 : realFrame}`;
  }

  function updateRates(): void {
    const now = performance.now();
    const elapsed = now - windowStart;
    if (elapsed < FPS_WINDOW_MS) {
      return;
    }

    const perSecond = (count: number) => (count * 1000) / elapsed;
    hud.fps.textContent = `${perSecond(presentedInWindow).toFixed(0)} / ${perSecond(realInWindow).toFixed(0)}`;
    hud.real.textContent = String(realFrame);
    windowStart = now;
    presentedInWindow = 0;
    realInWindow = 0;
  }

  requestAnimationFrame(tick);
}

main().catch((error: unknown) => {
  status.textContent = `startup failed: ${String(error)}`;
  console.error(error);
});
