import assert from "node:assert/strict";
import { parseFrames } from "./frames.js";

for (const pad of ["", "x", "xy", "xyz"]) {
  for (const count of [1, 2]) {
    const header = { frames: 2, bins: 3, frame_rate: 60, pad };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const values = new Float32Array(6 * count).map((_, i) => i / 16);
    const buffer = new ArrayBuffer(4 + json.length + values.byteLength);
    new DataView(buffer).setUint32(0, json.length, true);
    new Uint8Array(buffer, 4).set(json);
    new Uint8Array(buffer, 4 + json.length).set(new Uint8Array(values.buffer));
    const parsed = parseFrames(buffer);
    assert.deepEqual(parsed.header, header);
    assert.deepEqual(
      parsed.grids.map((grid) => Array.from(grid)),
      Array.from({ length: count }, (_, i) => Array.from(values.subarray(i * 6, (i + 1) * 6))),
    );
  }
}
console.log("ok");
