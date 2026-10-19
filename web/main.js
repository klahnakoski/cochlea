import * as THREE from "three";
import { parseFrames } from "./frames.js";

const BINS = 96;
const TURN_HEIGHT = 1.2;
const MAX_LENGTH = 4 * TURN_HEIGHT; // a bar at level 1/4 is a cube
const RADIUS = TURN_HEIGHT / (4 * Math.sin(Math.PI / 12)); // inner face of each wedge is 1 wide, 2 tall
const LABEL_SIZE = 0.35;
// Bar colours are OKLCH: lightness is perceptual, so equal steps look equal.
const DARKEST = 0.4; // bass
const LIGHTEST = 0.9; // treble
const CHROMA = 0.12;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b12);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
const CENTRE = (TURN_HEIGHT * (BINS / 12 - 1)) / 2;
const VIEW = new THREE.Vector3(0, 5, 11); // camera offset from the centre at zoom 1
function zoom(factor) {
  camera.position.set(0, CENTRE, 0).addScaledVector(VIEW, factor);
  camera.lookAt(0, CENTRE, 0);
}
zoom(1);
document.getElementById("zoom").addEventListener("input", (event) => zoom(Number(event.target.value)));

// Dragging tilts the axis about the spiral's centre, so it stays mid-screen.
const pivot = new THREE.Group();
pivot.position.y = CENTRE;
scene.add(pivot);
const helix = new THREE.Group();
helix.position.y = -CENTRE;
pivot.add(helix);
const DRAG_RADIANS_PER_PIXEL = 0.005;
const TILT_FORWARD = Math.PI; // far enough to point the axis straight down
const TILT_SIDEWAYS = Math.PI / 2;
const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
renderer.domElement.addEventListener("pointermove", (event) => {
  if (!(event.buttons & 1)) return;
  pivot.rotation.x = clamp(pivot.rotation.x + event.movementY * DRAG_RADIANS_PER_PIXEL, -TILT_FORWARD, TILT_FORWARD);
  pivot.rotation.z = clamp(pivot.rotation.z - event.movementX * DRAG_RADIANS_PER_PIXEL, -TILT_SIDEWAYS, TILT_SIDEWAYS);
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
// OKLCH to a linear-sRGB colour (Björn Ottosson's OKLab matrices).
function oklch(lightness, chroma, hue) {
  const a = chroma * Math.cos(hue);
  const b = chroma * Math.sin(hue);
  const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const clip = (value) => Math.min(Math.max(value, 0), 1);
  return new THREE.Color().setRGB(
    clip(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    clip(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    clip(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
    THREE.LinearSRGBColorSpace,
  );
}
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
  const lightness = DARKEST + ((LIGHTEST - DARKEST) * bin) / (BINS - 1);
  const color = oklch(lightness, CHROMA, ((bin % 12) / 12) * Math.PI * 2);
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
const suppress = document.getElementById("suppress");
const audio = new AudioContext();
let source = null;
let startTime = 0;
let spectrogram = null;
let currentFile = null;

function stop() {
  if (source) source.stop();
  source = null;
}

async function fetchFrames(file) {
  const form = new FormData();
  form.append("audio", file);
  const response = await fetch(`/analyze?suppress=${suppress.checked ? 1 : 0}`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`analyze failed: ${response.status}`);
  return parseFrames(await response.arrayBuffer());
}

async function play(file) {
  info.textContent = "analyzing...";
  currentFile = file;
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

renderer.setAnimationLoop(() => {
  // what the speaker plays now left the context outputLatency ago
  const t = audio.currentTime - startTime - (audio.outputLatency || 0);
  for (let bin = 0; bin < BINS; bin++) {
    const level = source && t >= 0 ? frameValue(t * spectrogram.header.frame_rate, bin) : 0;
    shapeBar(bin, Math.max(level, 0.02) * MAX_LENGTH);
  }
  geometry.attributes.position.needsUpdate = true;
  renderer.render(scene, camera);
});

document.getElementById("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) play(file).catch((error) => (info.textContent = String(error)));
});
document.getElementById("stop").addEventListener("click", stop);
// re-analyze the playing file; playback continues, frames swap when they arrive
suppress.addEventListener("change", () => {
  if (!currentFile) return;
  const file = currentFile;
  fetchFrames(file)
    .then((frames) => {
      if (file === currentFile) spectrogram = frames;
    })
    .catch((error) => (info.textContent = String(error)));
});
