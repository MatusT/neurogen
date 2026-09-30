import { FrameGenerator, type FrameGeneratorOptions } from "./FrameGenerator.js";
import { HydraFrameGenerator } from "./HydraFrameGenerator.js";

export type InterpolationBackend = "fsr3" | "hydra";

export function createInterpolator(options: FrameGeneratorOptions & { backend: "fsr3" }): FrameGenerator;
export function createInterpolator(options: FrameGeneratorOptions & { backend: "hydra" }): HydraFrameGenerator;
export function createInterpolator(options: FrameGeneratorOptions & { backend: InterpolationBackend }): FrameGenerator | HydraFrameGenerator;
export function createInterpolator(options: FrameGeneratorOptions & { backend: InterpolationBackend }): FrameGenerator | HydraFrameGenerator {
  switch (options.backend) {
    case "fsr3": return new FrameGenerator(options);
    case "hydra": return new HydraFrameGenerator(options);
    default: throw new Error(`Unknown interpolation backend: ${String(options.backend)}`);
  }
}
