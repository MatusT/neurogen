export interface FrameGeneratorOptions {
  device: GPUDevice;
  outputFormat: GPUTextureFormat;
}

export class FrameGenerator {
  readonly device: GPUDevice;
  readonly outputFormat: GPUTextureFormat;

  constructor(options: FrameGeneratorOptions) {
    this.device = options.device;
    this.outputFormat = options.outputFormat;
  }
}
