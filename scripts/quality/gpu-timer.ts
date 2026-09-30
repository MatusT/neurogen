export interface GpuTiming {
  prepareMs: number;
  dispatchMs: number;
  totalMs: number;
}

export function decodeGpuTiming(timestamps: BigUint64Array): GpuTiming {
  if (timestamps.length !== 4 || timestamps[1] < timestamps[0] || timestamps[3] < timestamps[2]) {
    throw new Error("Invalid GPU timestamp results");
  }
  // WebGPU resolves timestamps in nanoseconds. Subtract as bigint before
  // converting, to preserve precision on devices with a large clock epoch.
  const prepareMs = Number(timestamps[1] - timestamps[0]) / 1e6;
  const dispatchMs = Number(timestamps[3] - timestamps[2]) / 1e6;
  return { prepareMs, dispatchMs, totalMs: prepareMs + dispatchMs };
}

/** One serial sample at a time; resolve/readback happen outside wall timing. */
export class GpuTimer {
  private readonly queries: GPUQuerySet;
  private readonly resolved: GPUBuffer;
  private readonly readback: GPUBuffer;
  readonly prepareWrites: GPUComputePassTimestampWrites;
  readonly dispatchWrites: GPUComputePassTimestampWrites;

  constructor(private readonly device: GPUDevice) {
    if (!device.features.has("timestamp-query")) throw new Error("GPU timing requires timestamp-query");
    this.queries = device.createQuerySet({ label: "benchmark-timestamps", type: "timestamp", count: 4 });
    this.resolved = device.createBuffer({ label: "benchmark-query-resolve", size: 32,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readback = device.createBuffer({ label: "benchmark-query-readback", size: 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.prepareWrites = { querySet: this.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
    this.dispatchWrites = { querySet: this.queries, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 };
  }

  async read(): Promise<GpuTiming> {
    const encoder = this.device.createCommandEncoder({ label: "benchmark-query-readback" });
    encoder.resolveQuerySet(this.queries, 0, 4, this.resolved, 0);
    encoder.copyBufferToBuffer(this.resolved, 0, this.readback, 0, 32);
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    try { return decodeGpuTiming(new BigUint64Array(this.readback.getMappedRange())); }
    finally { this.readback.unmap(); }
  }

  destroy(): void {
    this.queries.destroy();
    this.resolved.destroy();
    this.readback.destroy();
  }
}
