import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b12);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 4, 8);

const controls = new OrbitControls(camera, renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(3, 6, 4);
scene.add(sun);

// One box per pitch class, around a ring.
for (let pitch = 0; pitch < 12; pitch++) {
  const angle = (pitch / 12) * Math.PI * 2;
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 1, 0.5),
    new THREE.MeshStandardMaterial({ color: new THREE.Color().setHSL(pitch / 12, 0.7, 0.55) }),
  );
  box.position.set(Math.cos(angle) * 3, 0.5, Math.sin(angle) * 3);
  scene.add(box);
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

const info = document.getElementById("info");
const audio = new AudioContext();
let source = null;

function stop() {
  if (source) source.stop();
  source = null;
}

async function analyze(file) {
  const form = new FormData();
  form.append("audio", file);
  const response = await fetch("/analyze", { method: "POST", body: form });
  info.textContent = JSON.stringify(await response.json(), null, 2);
}

async function play(file) {
  const buffer = await audio.decodeAudioData(await file.arrayBuffer());
  await audio.resume();
  stop();
  source = audio.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.destination);
  source.start();
}

document.getElementById("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  analyze(file);
  play(file);
});
document.getElementById("stop").addEventListener("click", stop);
