export interface ImageMetrics {
  mae: number;
  rmse: number;
  psnr: number | null;
  changedRegionMae: number | null;
  changedPixels: number;
}

/** Linear RGB in [0, 1]; alpha is excluded. Null PSNR means an exact match. */
export function imageMetrics(actual: Float32Array, truth: Float32Array,
  previous: Float32Array, current: Float32Array): ImageMetrics {
  if (actual.length !== truth.length || previous.length !== truth.length || current.length !== truth.length || truth.length % 4 !== 0 || !truth.length) {
    throw new Error("Quality images must have equal, nonempty RGBA dimensions");
  }
  let absolute = 0, squared = 0, changedAbsolute = 0, changedPixels = 0;
  for (let i = 0; i < truth.length; i += 4) {
    let changed = false, pixelAbsolute = 0;
    for (let c = 0; c < 3; c++) {
      const index = i + c;
      if (!Number.isFinite(actual[index]) || !Number.isFinite(truth[index]) ||
          !Number.isFinite(previous[index]) || !Number.isFinite(current[index])) {
        throw new Error(`Nonfinite color at component ${index}`);
      }
      const error = actual[index] - truth[index];
      pixelAbsolute += Math.abs(error);
      squared += error * error;
      changed ||= Math.abs(previous[index] - current[index]) > 1 / 255 ||
        Math.abs(previous[index] - truth[index]) > 1 / 255;
    }
    absolute += pixelAbsolute;
    if (changed) { changedPixels++; changedAbsolute += pixelAbsolute; }
  }
  const components = truth.length / 4 * 3;
  const rmse = Math.sqrt(squared / components);
  return { mae: absolute / components, rmse, psnr: rmse === 0 ? null : -20 * Math.log10(rmse),
    changedRegionMae: changedPixels ? changedAbsolute / (changedPixels * 3) : null, changedPixels };
}

export function crossfade(previous: Float32Array, current: Float32Array): Float32Array {
  return previous.map((value, i) => (value + current[i]) * 0.5);
}
