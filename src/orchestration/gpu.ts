// The handful of raw-WebGPU shapes both stages need. Nothing here knows what a
// frame is.

export const buf = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });

// COPY_SRC on every storage buffer: the module outputs have to be readable, and
// making the intermediates readable too is what lets a failing fixture dump the
// pass that actually went wrong rather than only the last one.
export function storageBuffer(device: GPUDevice, byteLength: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

export function uniformBuffer(
  device: GPUDevice,
  data: Uint8Array<ArrayBuffer>,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);

  return buffer;
}

export function pipeline(device: GPUDevice, code: string, label: string): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    // "auto" derives the layout from the composed WGSL that is about to run, so
    // it cannot drift from it. Every pass declares bindings 0..n-1 and
    // references all of them, which is what makes the positional binding lists
    // below equivalent to writing the layout out by hand.
    layout: "auto",
    compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
  });
}

// A binding a pass declares but never reads. "auto" derives the layout by
// reachability from the entry point, so such a binding is absent from it and
// supplying it is a validation error — the slot has to be held open to keep the
// rest of the list on its declared indices. Only scd_divergence.wgsl needs
// this: it takes the module's uniform at binding 0 like every other pass, but
// its geometry is all compile-time constants from scd.wgsl.
export const UNREFERENCED = undefined;

// Entry i binds to `@binding(i)`, so a call site reads as the pass's binding
// table in declaration order.
export function bindGroup(
  device: GPUDevice,
  target: GPUComputePipeline,
  resources: readonly (GPUBindingResource | undefined)[],
): GPUBindGroup {
  const entries = resources.flatMap((resource, binding) =>
    resource === undefined ? [] : [{ binding, resource }],
  );

  return device.createBindGroup({ layout: target.getBindGroupLayout(0), entries });
}

export interface ComputeStep {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  groups: readonly [number, number, number];
}

export function step(
  device: GPUDevice,
  target: GPUComputePipeline,
  resources: readonly (GPUBindingResource | undefined)[],
  groups: readonly [number, number, number],
): ComputeStep {
  return { pipeline: target, bindGroup: bindGroup(device, target, resources), groups };
}

export function encodeSteps(pass: GPUComputePassEncoder, steps: readonly ComputeStep[]): void {
  for (const step of steps) {
    pass.setPipeline(step.pipeline);
    pass.setBindGroup(0, step.bindGroup);
    pass.dispatchWorkgroups(...step.groups);
  }
}
