import assert from "node:assert/strict";
import { parseFrames } from "./frames.js";

for (const pad of ["", "x", "xy", "xyz"]) {
  const header = { frames: 2, bins: 3, frame_rate: 60, pad };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const values = new Float32Array([0, 0.25, 0.5, 0.75, 1, 0.125]);
  const buffer = new ArrayBuffer(4 + json.length + values.byteLength);
  new DataView(buffer).setUint32(0, json.length, true);
  new Uint8Array(buffer, 4).set(json);
  new Uint8Array(buffer, 4 + json.length).set(new Uint8Array(values.buffer));
  const parsed = parseFrames(buffer);
  assert.deepEqual(parsed.header, header);
  assert.deepEqual(Array.from(parsed.data), Array.from(values));
}
console.log("ok");
