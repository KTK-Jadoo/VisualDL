import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------- Scene ----------
const container = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0f12);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 20, 60);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(30, 50, 30);
scene.add(new THREE.AmbientLight(0xffffff, 0.25), light);

// ---------- Input plane ----------
function makeInputPlane() {
    const geom = new THREE.PlaneGeometry(14, 14, 1, 1);
    const texture = new THREE.DataTexture(new Uint8Array(28 * 28 * 4), 28, 28, THREE.RGBAFormat);
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(-22, 0, 0);
    scene.add(mesh);
    return { mesh, texture };
}
const inputPlane = makeInputPlane();

// ---------- Conv1 feature map tiles ----------
const FM_W = 24, FM_H = 24;       // example after conv (depends on your model)
const CONV1_CHANNELS = 8;         // tiny, clear, fast
const tiles = [];
(function makeConv1Tiles() {
    const cols = 4, rows = Math.ceil(CONV1_CHANNELS / cols);
    for (let c = 0; c < CONV1_CHANNELS; c++) {
        const tex = new THREE.DataTexture(new Uint8Array(FM_W * FM_H * 4), FM_W, FM_H, THREE.RGBAFormat);
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.95 });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), mat);

        const ci = c % cols, ri = Math.floor(c / cols);
        mesh.position.set(ci * 11 - (cols - 1) * 5.5, (rows - 1) * 5.5 - ri * 11, 0);
        scene.add(mesh);
        tiles.push({ mesh, tex });
    }
})();

// ---------- Output bars ----------
const classCount = 10;
const barGroup = new THREE.Group();
for (let i = 0; i < classCount; i++) {
    const m = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshPhongMaterial({ color: 0x66ccff })
    );
    m.position.set(18 + i * 1.4, -10, 0);
    barGroup.add(m);
}
scene.add(barGroup);

// ---------- Worker ----------
const worker = new Worker(new URL('./net.worker.js', import.meta.url), { type: 'module' });

worker.onmessage = (e) => {
    const { type, payload } = e.data;
    if (type === 'activations') {
        const { inputRGBA, conv1RGBA, probs } = payload;

        // input texture
        inputPlane.texture.image.data.set(inputRGBA);
        inputPlane.texture.needsUpdate = true;

        // conv1 tiles
        for (let c = 0; c < Math.min(CONV1_CHANNELS, conv1RGBA.length); c++) {
            tiles[c].tex.image.data.set(conv1RGBA[c]);
            tiles[c].tex.needsUpdate = true;
        }

        // output bars
        probs.forEach((p, i) => {
            const h = Math.max(0.2, p * 12);
            const mesh = barGroup.children[i];
            mesh.scale.set(1, h, 1);
            mesh.position.y = -10 + h / 2;
        });
    }
};

// ---------- Draw on input canvas ----------
const inputCanvas = document.getElementById('inputCanvas');
const ictx = inputCanvas.getContext('2d');
let drawing = false;
inputCanvas.addEventListener('mousedown', () => { drawing = true; });
window.addEventListener('mouseup', () => { drawing = false; sendFrame(); });
inputCanvas.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    const rect = inputCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left));
    const y = Math.floor((e.clientY - rect.top));
    ictx.fillStyle = '#fff';
    ictx.fillRect(x - 1, y - 1, 3, 3);
});

// Push frames to worker at a modest cadence:
function sendFrame() {
    const img = ictx.getImageData(0, 0, 28, 28);
    worker.postMessage({ type: 'infer', payload: img.data.buffer }, [img.data.buffer]);
}

// ---------- Loop ----------
function onResize() {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}
window.addEventListener('resize', onResize);

function tick() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
}
tick();

// Kick a first blank frame
sendFrame();
