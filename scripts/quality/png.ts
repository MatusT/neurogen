import { deflateSync } from "node:zlib";

function chunk(type: string, data: Buffer): Buffer {
  const body = Buffer.concat([Buffer.from(type), data]);
  let crc = 0xffffffff;
  for (const byte of body) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  const header = Buffer.alloc(4), footer = Buffer.alloc(4);
  header.writeUInt32BE(data.length);
  footer.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
  return Buffer.concat([header, body, footer]);
}

function srgb(linear: number): number {
  const value = Math.max(0, Math.min(1, linear));
  return value <= 0.0031308 ? value * 12.92 : 1.055 * value ** (1 / 2.4) - 0.055;
}

/** PNG previews use sRGB; metrics always use the original linear floating-point data. */
export function colorPng(pixels: Float32Array, width: number, height: number): Buffer {
  const rows = Buffer.alloc((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) {
    rows[y * (width * 3 + 1) + 1 + x * 3 + c] = Math.round(srgb(pixels[(y * width + x) * 4 + c]) * 255);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width); header.writeUInt32BE(height, 4);
  header[8] = 8; header[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header),
    chunk("sRGB", Buffer.from([0])), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}

export function errorImage(actual: Float32Array, truth: Float32Array): Float32Array {
  return truth.map((value, i) => i % 4 === 3 ? 1 : Math.min(1, Math.abs(value - actual[i]) * 8));
}
