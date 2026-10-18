// Wire format: uint32 LE header length, UTF-8 JSON header, then frames*bins float32 LE.
export function parseFrames(buffer) {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  // Float32Array needs a 4-byte aligned offset; copy when the header length breaks that.
  const start = 4 + headerLength;
  const data =
    start % 4 === 0
      ? new Float32Array(buffer, start, header.frames * header.bins)
      : new Float32Array(buffer.slice(start, start + header.frames * header.bins * 4));
  return { header, data };
}
