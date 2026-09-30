export interface TimingSummary {
  samples: number;
  meanMs: number;
  medianMs: number;
  p95Ms: number;
}

export function summarizeTimings(samples: readonly number[]): TimingSummary {
  if (!samples.length || samples.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error("Benchmark timings must be finite, nonnegative, and nonempty");
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    samples: sorted.length,
    meanMs: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    medianMs: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
  };
}

// Drain earlier rendering/other backends before starting the clock. Include
// CPU encoding, submission, GPU execution, and completion notification.
export async function timeInterpolation(queue: Pick<GPUQueue, "onSubmittedWorkDone">, interpolate: () => void): Promise<number> {
  await queue.onSubmittedWorkDone();
  const start = performance.now();
  interpolate();
  await queue.onSubmittedWorkDone();
  return performance.now() - start;
}
