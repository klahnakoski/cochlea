import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { parseFrames } from "./frames.js";

const BINS = 96;
const RADIUS = 3;
const MAX_LENGTH = 2;
const TURN_HEIGHT = 0.6;
const DECAY_PER_SECOND = 0.02;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b12);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 7, 11);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, (TURN_HEIGHT * 7) / 2, 0);
controls.maxPolarAngle = Math.PI / 2; // never look from below
controls.enablePan = false;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(3, 6, 4);
scene.add(sun);

const bars = new THREE.InstancedMesh(
  new THREE.BoxGeometry(1, 0.12, 0.12).translate(0.5, 0, 0),
  new THREE.MeshStandardMaterial(),
  BINS,
);
bars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
const angleOf = (bin) => ((bin % 12) / 12) * Math.PI * 2;
const heightOf = (bin) => (bin / 12) * TURN_HEIGHT;
for (let bin = 0; bin < BINS; bin++) {
  bars.setColorAt(bin, new THREE.Color().setHSL((bin % 12) / 12, 0.75, 0.55));
}
scene.add(bars);

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
  sprite.position.set(Math.cos(angle) * (RADIUS - 0.5), 0, Math.sin(angle) * (RADIUS - 0.5));
  scene.add(sprite);
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

const matrix = new THREE.Matrix4();
const rotation = new THREE.Quaternion();
const position = new THREE.Vector3();
const scale = new THREE.Vector3();
const up = new THREE.Vector3(0, 1, 0);

renderer.setAnimationLoop((now) => {
  const dt = Math.min((now - lastRender) / 1000, 0.1);
  lastRender = now;
  const decay = Math.pow(DECAY_PER_SECOND, dt);
  const t = audio.currentTime - startTime + (audio.outputLatency || 0);
  for (let bin = 0; bin < BINS; bin++) {
    const target = source && t >= 0 ? frameValue(t * spectrogram.header.frame_rate, bin) : 0;
    levels[bin] = Math.max(target, levels[bin] * decay);
    const angle = angleOf(bin);
    rotation.setFromAxisAngle(up, -angle);
    position.set(Math.cos(angle) * RADIUS, heightOf(bin), Math.sin(angle) * RADIUS);
    scale.set(Math.max(levels[bin], 0.02) * MAX_LENGTH, 1, 1);
    bars.setMatrixAt(bin, matrix.compose(position, rotation, scale));
  }
  bars.instanceMatrix.needsUpdate = true;
  controls.update();
  renderer.render(scene, camera);
});

document.getElementById("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) play(file).catch((error) => (info.textContent = String(error)));
});
document.getElementById("stop").addEventListener("click", stop);
