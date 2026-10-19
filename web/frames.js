// Wire format: uint32 LE header length, UTF-8 JSON header, then one or more frames*bins float32 LE grids.
export function parseFrames(buffer) {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  // Float32Array needs a 4-byte aligned offset; copy when the header length breaks that.
  const start = 4 + headerLength;
  const size = header.frames * header.bins;
  const count = Math.floor((buffer.byteLength - start) / (size * 4));
  const all =
    start % 4 === 0
      ? new Float32Array(buffer, start, count * size)
      : new Float32Array(buffer.slice(start, start + count * size * 4));
  const grids = Array.from({ length: count }, (_, i) => all.subarray(i * size, (i + 1) * size));
  return { header, grids };
}
