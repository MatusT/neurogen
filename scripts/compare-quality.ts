import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { create, globals } from "webgpu";
import { createInterpolator, type InterpolationBackend } from "../src/index.js";
import { GBuffer } from "../demo/gbuffer.js";
import { demoScene, markerAt, PROJECTION, projectionFor, type Instance } from "../demo/scene.js";
import { perspective } from "../demo/math.js";
import { readColor, measureMarker, midpointFailures } from "../demo/verify.js";
import { imageMetrics, crossfade, type ImageMetrics } from "./quality/metrics.js";
import { colorPng, errorImage } from "./quality/png.js";

Object.assign(globalThis, globals);
const gpu = create([]); // Keep Dawn's owner alive until all work and readbacks finish.
const VARIANTS = ["fsr3", "fsr3-optical-only", "hydra"] as const;
const CASES = ["static", "thin-bar", "textured-objects", "camera-pan"] as const;
type Case = typeof CASES[number];
type Variant = typeof VARIANTS[number];
const FRAME_COUNT = 24;
const FIRST_SCORED_FRAME = 18;
const CAMERA_STEP = [0.035, 0.01] as const;

interface Result {
  scene: Case;
  variant: Variant | "crossfade";
  frame: number;
  metrics: ImageMetrics;
  markerFailures?: string[];
}

function options(): { size: readonly [number, number]; output: string } {
  const args = process.argv.slice(2).filter(arg => arg !== "--");
  let size: readonly [number, number] = [1280, 720];
  let output = "artifacts/quality";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--size") {
      const match = /^(\d+)x(\d+)$/.exec(args[++i] ?? "");
      if (!match) throw new Error("Expected --size WIDTHxHEIGHT");
      size = [Number(match[1]), Number(match[2])];
      if (size.some(value => value < 64 || value > 3840)) throw new Error("Quality dimensions must be in [64, 3840]");
    } else if (args[i] === "--out" && args[i + 1]) output = args[++i];
    else throw new Error(`Unknown argument: ${args[i]}`);
  }
  return { size, output: resolve(output) };
}

function sceneAt(scene: Case, size: readonly [number, number], frame: number): Instance[] {
  const projection = projectionFor(size);
  if (scene === "thin-bar") return markerAt(projection, size, frame);
  if (scene === "textured-objects") return demoScene(projection, frame);
  const instances = demoScene(projection, 0);
  for (const instance of instances) {
    instance.previousModel = instance.model.slice();
    if (scene === "camera-pan") {
      for (const axis of [0, 1]) {
        instance.model[12 + axis] += CAMERA_STEP[axis] * (frame - 12);
        instance.previousModel[12 + axis] += CAMERA_STEP[axis] * (frame - 13);
      }
    }
  }
  return instances;
}

function cameraTransforms(size: readonly [number, number]) {
  const p = perspective(projectionFor(size));
  const forward = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  // P * translate(view-space step) * inverse(P), for this finite perspective.
  for (const axis of [0, 1]) {
    forward[8 + axis] = p[axis * 5] * CAMERA_STEP[axis] / p[14];
    forward[12 + axis] = forward[8 + axis] * p[10];
  }
  const backward = forward.slice();
  for (const index of [8, 9, 12, 13]) backward[index] *= -1;
  return { previousToCurrentClip: forward, currentToPreviousClip: backward };
}

async function sourceHash(): Promise<string> {
  const hash = createHash("sha256");
  async function walk(directory: string): Promise<void> {
    for (const entry of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(directory, entry.name);
      if (entry.isDirectory() && entry.name !== "generated") await walk(path);
      else if (entry.isFile()) { hash.update(path.replaceAll("\\", "/")); hash.update(await readFile(path)); }
    }
  }
  for (const directory of ["src", "demo", "scripts/quality"]) await walk(directory);
  hash.update(await readFile("scripts/compare-quality.ts"));
  return hash.digest("hex");
}

function html(results: Result[], metadata: unknown): string {
  const escape = (value: unknown) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");
  const figure = (scene: string, name: string, label = name) => `<figure><a href="${scene}/${name}.png"><img src="${scene}/${name}.png"></a><figcaption>${label}</figcaption></figure>`;
  const sections = CASES.map(scene => {
    const rows = [...VARIANTS, "crossfade"].map(variant => {
      const frames = results.filter(row => row.scene === scene && row.variant === variant);
      const average = (key: "mae" | "rmse" | "changedRegionMae") => {
        const values = frames.map(row => row.metrics[key]).filter((value): value is number => value !== null);
        return values.length ? (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(5) : "—";
      };
      const failures = frames.filter(row => row.markerFailures?.length).length;
      return `<tr><td>${variant}</td><td>${average("mae")}</td><td>${average("rmse")}</td><td>${average("changedRegionMae")}</td><td>${scene === "thin-bar" ? `${failures}/${frames.length} frames failed` : "—"}</td></tr>`;
    }).join("");
    const failures = results.filter(row => row.scene === scene && row.frame === FRAME_COUNT - 1 && row.markerFailures?.length)
      .map(row => `<p><b>${row.variant}</b>: ${escape(row.markerFailures!.join("; "))}</p>`).join("");
    return `<section><h2>${scene}</h2><table><tr><th>Variant</th><th>Mean MAE ↓</th><th>Mean RMSE ↓</th><th>Changed-region MAE ↓</th><th>Marker geometry</th></tr>${rows}</table>
      <div class="images">${figure(scene, "previous", "Real frame 22")}${figure(scene, "truth", "Rendered ground truth at 22.5")}${figure(scene, "current", "Real frame 23")}</div>
      <div class="images">${[...VARIANTS, "crossfade"].map(variant => figure(scene, variant)).join("")}</div>
      <div class="images">${[...VARIANTS, "crossfade"].map(variant => figure(scene, `${variant}-error`, `${variant}: absolute error ×8`)).join("")}</div>
      ${failures ? `<details><summary>Last-frame marker failures</summary>${failures}</details>` : ""}</section>`;
  }).join("");
  return `<!doctype html><meta charset="utf-8"><title>Neurogen quality comparison</title><style>
    body{font:15px system-ui;background:#141820;color:#e2e8f0;margin:24px}p{max-width:100ch}table{border-collapse:collapse}td,th{padding:8px 18px;text-align:left;border-bottom:1px solid #465065}
    .images{display:flex;gap:12px;margin-top:20px}figure{flex:1;min-width:0;margin:0}img{width:100%;image-rendering:pixelated}figcaption{padding:6px 0}section{margin-bottom:44px}pre{overflow:auto}</style>
    <h1>Interpolation quality: FSR3 and recovered Hydra</h1>
    <p>Each backend processes the same 24 real frames. Metrics score frames 18–23 against the scene rendered directly at n−0.5. Preview images show the last pair; click any image for full resolution. Lower errors are better.</p>
    <p>FSR3 receives game motion vectors; fsr3-optical-only receives zero vectors. Hydra receives depth and camera transforms, and estimates object motion from color. Neural refinement is off. Crossfade is a non-motion-compensated baseline.</p>
    <p>Metrics use linear floating-point RGB, without alpha. Changed-region metrics include pixels where the real frames or the ground truth differ by more than 1/255. PNG previews are sRGB. Error previews amplify absolute linear error by 8 before sRGB conversion. These synthetic scenes do not establish native-SDK parity or general game quality.</p>
    ${sections}<details><summary>Reproduction metadata</summary><pre>${escape(JSON.stringify(metadata, null, 2))}</pre></details>`;
}

async function main(): Promise<void> {
  const { size, output } = options();
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter available");
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => errors.push(event.error.message));
  const results: Result[] = [];
  const projection = projectionFor(size);
  const zeroMotion = device.createTexture({ size, format: "rg16float", usage: GPUTextureUsage.TEXTURE_BINDING });
  await mkdir(output, { recursive: true });
  const revision = { sourceSha256: await sourceHash(),
    commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    workingTree: execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim() };
  try {
    for (const scene of CASES) {
      console.log(`Rendering ${scene} at ${size.join("x")}...`);
      const directory = join(output, scene);
      await mkdir(directory, { recursive: true });
      device.pushErrorScope("validation");
      const real = new GBuffer(device, size, projection), truth = new GBuffer(device, size, projection);
      const generators = VARIANTS.map(variant => ({ variant,
        generator: createInterpolator({ device, backend: (variant === "hydra" ? "hydra" : "fsr3") as InterpolationBackend }) }));
      try {
        for (const { generator } of generators) generator.configure({ ...PROJECTION, renderWidth: size[0], renderHeight: size[1] });
        for (let frame = 0; frame < FRAME_COUNT; frame++) {
          real.renderFrame(sceneAt(scene, size, frame));
          if (frame === 0) continue;
          const outputs = new Map<string, GPUTexture>();
          for (const { generator, variant } of generators) {
            generator.prepare({ ...real.frameInputs(), ...(scene === "camera-pan" ? cameraTransforms(size) : {}),
              ...(variant === "fsr3-optical-only" ? { motionVectors: zeroMotion.createView() } : {}) });
            outputs.set(variant, generator.dispatch());
          }
          if (frame < FIRST_SCORED_FRAME) continue;
          truth.renderFrame(sceneAt(scene, size, frame - 0.5));
          const previous = await readColor(device, real.previousColor);
          const current = await readColor(device, real.currentColor);
          const expected = await readColor(device, truth.currentColor);
          const images = new Map<string, Float32Array>();
          for (const [variant, texture] of outputs) images.set(variant, await readColor(device, texture));
          images.set("crossfade", crossfade(previous, current));
          for (const [variant, image] of images) {
            results.push({ scene, variant: variant as Result["variant"], frame,
              metrics: imageMetrics(image, expected, previous, current),
              ...(scene === "thin-bar" ? { markerFailures: midpointFailures(measureMarker(previous, size), measureMarker(current, size), measureMarker(image, size)) } : {}) });
            if (frame === FRAME_COUNT - 1) {
              await writeFile(join(directory, `${variant}.png`), colorPng(image, ...size));
              await writeFile(join(directory, `${variant}-error.png`), colorPng(errorImage(image, expected), ...size));
            }
          }
          if (frame === FRAME_COUNT - 1) for (const [name, pixels] of [["previous", previous], ["current", current], ["truth", expected]] as const) {
            await writeFile(join(directory, `${name}.png`), colorPng(pixels, ...size));
          }
        }
      } finally {
        for (const { generator } of generators) generator.destroy();
        real.destroy(); truth.destroy();
        const validation = await device.popErrorScope();
        if (validation) errors.push(validation.message);
      }
      if (errors.length) throw new Error(errors.join("\n"));
    }
    const metadata = { size, realFrames: FRAME_COUNT, scoredFrames: [FIRST_SCORED_FRAME, FRAME_COUNT - 1],
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      ...revision,
      node: process.version, platform: process.platform, webgpu: JSON.parse(await readFile("node_modules/webgpu/package.json", "utf8")).version,
      validationErrors: errors, command: "npm run quality -- --size " + size.join("x") };
    await writeFile(join(output, "results.json"), JSON.stringify({ metadata, results }, null, 2));
    await writeFile(join(output, "report.html"), html(results, metadata));
    console.table(results.filter(row => row.frame === FRAME_COUNT - 1).map(row => ({ scene: row.scene, variant: row.variant,
      mae: row.metrics.mae.toFixed(5), changedMae: row.metrics.changedRegionMae?.toFixed(5) ?? "—", markerFailures: row.markerFailures?.length ?? "—" })));
    console.log(`Report: ${join(output, "report.html")}\nData: ${join(output, "results.json")}`);
  } finally {
    await device.queue.onSubmittedWorkDone();
    zeroMotion.destroy(); device.destroy();
  }
}

main().then(() => process.exit(0), error => { console.error(error); process.exit(1); });
