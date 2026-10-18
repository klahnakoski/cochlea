import * as THREE from "three";
import { parseFrames } from "./frames.js";

const BINS = 96;
const MAX_LENGTH = 2;
const TURN_HEIGHT = 0.6;
const RADIUS = TURN_HEIGHT / (2 * Math.sin(Math.PI / 12)); // inner face of each wedge is square
const LABEL_SIZE = 0.35;
const DECAY_PER_SECOND = 0.02;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b12);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 7, 11);
camera.lookAt(0, (TURN_HEIGHT * 7) / 2, 0);

// Dragging moves the top of the axis; the base stays put.
const helix = new THREE.Group();
scene.add(helix);
const DRAG_RADIANS_PER_PIXEL = 0.005;
const TILT_TOWARD = Math.PI / 3;
const TILT_AWAY = Math.PI / 4; // farther would show the bottom
const TILT_SIDEWAYS = Math.PI / 3;
const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!(event.buttons & 1)) return;
  helix.rotation.x = clamp(helix.rotation.x + event.movementY * DRAG_RADIANS_PER_PIXEL, -TILT_AWAY, TILT_TOWARD);
  helix.rotation.z = clamp(helix.rotation.z - event.movementX * DRAG_RADIANS_PER_PIXEL, -TILT_SIDEWAYS, TILT_SIDEWAYS);
});

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(3, 6, 4);
scene.add(sun);

// Each bar is a wedge spanning its pitch class's 30°, so width grows with radius
// and neighbours touch; one turn tall, so columns stack without gaps.
const HALF_ANGLE = Math.PI / 12;
// Corner bits: 1 = top, 2 = counter-clockwise side, 4 = outer.
const FACES = [
  [4, 5, 6, 7],
  [0, 1, 2, 3],
  [1, 3, 5, 7],
  [0, 2, 4, 6],
  [0, 1, 4, 5],
  [2, 3, 6, 7],
];
const CORNERS = FACES.flatMap(([a, b, c, d]) => [a, b, c, c, b, d]);
const angleOf = (bin) => ((bin % 12) / 12) * Math.PI * 2;
const heightOf = (bin) => (bin / 12) * TURN_HEIGHT;

const positions = new Float32Array(BINS * CORNERS.length * 3);
const colors = new Float32Array(positions.length);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

function shapeBar(bin, length) {
  const base = bin * CORNERS.length * 3;
  CORNERS.forEach((corner, i) => {
    const angle = angleOf(bin) + (corner & 2 ? HALF_ANGLE : -HALF_ANGLE);
    const radius = corner & 4 ? RADIUS + length : RADIUS;
    const y = heightOf(bin) + (corner & 1 ? TURN_HEIGHT / 2 : -TURN_HEIGHT / 2);
    positions.set([Math.cos(angle) * radius, y, Math.sin(angle) * radius], base + i * 3);
  });
}

for (let bin = 0; bin < BINS; bin++) {
  shapeBar(bin, 1);
  const color = new THREE.Color().setHSL((bin % 12) / 12, 0.75, 0.55);
  for (let i = 0; i < CORNERS.length; i++) color.toArray(colors, (bin * CORNERS.length + i) * 3);
}
// Wedge faces lie on fixed planes, so normals never change with length.
geometry.computeVertexNormals();
const bars = new THREE.Mesh(
  geometry,
  new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide }),
);
bars.frustumCulled = false;
helix.add(bars);

function label(text, angle) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ddd";
  context.font = "40px sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 32, 32);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas) }));
  sprite.scale.setScalar(LABEL_SIZE);
  const radius = RADIUS + LABEL_SIZE;
  sprite.position.set(Math.cos(angle) * radius, -TURN_HEIGHT, Math.sin(angle) * radius);
  helix.add(sprite);
}
NOTE_NAMES.forEach((name, pitch) => label(name, angleOf(pitch)));

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const info = document.getElementById("info");
const audio = new AudioContext();
let source = null;
let startTime = 0;
let spectrogram = null;
const levels = new Float32Array(BINS);
let lastRender = 0;

function stop() {
  if (source) source.stop();
  source = null;
}

async function fetchFrames(file) {
  const form = new FormData();
  form.append("audio", file);
  const response = await fetch("/analyze", { method: "POST", body: form });
  if (!response.ok) throw new Error(`analyze failed: ${response.status}`);
  return parseFrames(await response.arrayBuffer());
}

async function play(file) {
  info.textContent = "analyzing...";
  const [frames, buffer] = await Promise.all([
    fetchFrames(file),
    file.arrayBuffer().then((bytes) => audio.decodeAudioData(bytes)),
  ]);
  await audio.resume();
  stop();
  spectrogram = frames;
  info.textContent = JSON.stringify(frames.header, null, 2);
  source = audio.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.destination);
  startTime = audio.currentTime;
  source.start(startTime);
}

// Frame row at fractional position `at`, lerped between neighbours.
function frameValue(at, bin) {
  const { header, data } = spectrogram;
  const last = header.frames - 1;
  const clamped = Math.min(Math.max(at, 0), last);
  const low = Math.floor(clamped);
  const high = Math.min(low + 1, last);
  const mix = clamped - low;
  return data[low * header.bins + bin] * (1 - mix) + data[high * header.bins + bin] * mix;
}

renderer.setAnimationLoop((now) => {
  const dt = Math.min((now - lastRender) / 1000, 0.1);
  lastRender = now;
  const decay = Math.pow(DECAY_PER_SECOND, dt);
  const t = audio.currentTime - startTime + (audio.outputLatency || 0);
  for (let bin = 0; bin < BINS; bin++) {
    const target = source && t >= 0 ? frameValue(t * spectrogram.header.frame_rate, bin) : 0;
    levels[bin] = Math.max(target, levels[bin] * decay);
    shapeBar(bin, Math.max(levels[bin], 0.02) * MAX_LENGTH);
  }
  geometry.attributes.position.needsUpdate = true;
  renderer.render(scene, camera);
});

document.getElementById("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) play(file).catch((error) => (info.textContent = String(error)));
});
document.getElementById("stop").addEventListener("click", stop);
