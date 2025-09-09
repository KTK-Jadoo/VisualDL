// src/main.js
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

/* =========== Responsive pipeline layout (with depth) =========== */
const LAYOUT = {
  gapInputToC1: 20,   // X distance Input -> Conv1
  gapC1ToC2:    18,   // X distance Conv1 -> Conv2
  gapC2ToBars:  22,   // X distance Conv2 -> Bars
  depthStep:    14,   // Z distance between stages (prevents edge-on collapse)
  baseY:         0,   // Y baseline for input/conv groups
  barsY:        -8,   // Y offset for bars
};

function computeLeftAnchor() {
  const approxWidth = 14 + LAYOUT.gapInputToC1 + 28 + LAYOUT.gapC1ToC2 + 40 + LAYOUT.gapC2ToBars + 20;
  return -approxWidth * 0.45;
}

function layoutPipeline() {
  const left = computeLeftAnchor();

  // X positions
  const xInput = left;
  const xC1    = xInput + LAYOUT.gapInputToC1;
  const xC2    = xC1    + LAYOUT.gapC1ToC2;
  const xBars  = xC2    + LAYOUT.gapC2ToBars;

  // Z positions (diagonal march into the scene)
  const zInput = 0;
  const zC1    = zInput - LAYOUT.depthStep;
  const zC2    = zC1    - LAYOUT.depthStep;
  const zBars  = zC2    - LAYOUT.depthStep;

  inputPlane.mesh.position.set(xInput, LAYOUT.baseY, zInput);
  group1.position.set(xC1, LAYOUT.baseY, zC1);
  group2.position.set(xC2, LAYOUT.baseY, zC2);
  barGroup.position.set(xBars, LAYOUT.barsY, zBars);
}

// frame camera on the diagonal row
function framePipelineFront() {
  const centerX = computeLeftAnchor() + LAYOUT.gapInputToC1 + LAYOUT.gapC1ToC2 * 0.5;
  const centerZ = -LAYOUT.depthStep * 1.5;
  camera.position.set(centerX, 22, centerZ + 60);
  controls.target.set(centerX, 0, centerZ);
  controls.update();
}

/* =========== Renderer / Scene / Camera =========== */
const container = document.getElementById('app');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0f12);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 18, 60);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

/* Lights */
scene.add(new THREE.AmbientLight(0xffffff, 0.25));
const key = new THREE.DirectionalLight(0xffffff, 1.0); key.position.set(30, 50, 30); scene.add(key);
const rimL = new THREE.DirectionalLight(0xffffff, 0.35); rimL.position.set(-40, 20, -20); scene.add(rimL);

/* Ground grid */
const grid = new THREE.GridHelper(160, 40, 0x222222, 0x2a2a2a);
grid.material.opacity = 0.15; grid.material.transparent = true;
grid.position.y = -12;
scene.add(grid);

/* =========== Label Renderer =========== */
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(window.innerWidth, window.innerHeight);
labelRenderer.domElement.style.position = 'fixed';
labelRenderer.domElement.style.inset = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
document.body.appendChild(labelRenderer.domElement);

/* =========== Helpers =========== */
function makeOpaqueTexture(w,h){
  const data = new Uint8Array(w*h*4);
  for(let i=0;i<w*h;i++){ const j=i*4; data[j]=data[j+1]=data[j+2]=0; data[j+3]=255; }
  const tex = new THREE.DataTexture(data,w,h,THREE.RGBAFormat);
  tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
  return tex;
}
function makeLabel(text){
  const div = document.createElement('div');
  div.textContent = text;
  div.style.padding = '2px 6px';
  div.style.font = '12px/1.2 system-ui, -apple-system, Segoe UI, Roboto';
  div.style.color = '#e6e6e6';
  div.style.background = 'rgba(0,0,0,.45)';
  div.style.border = '1px solid rgba(255,255,255,.15)';
  div.style.borderRadius = '6px';
  div.style.backdropFilter = 'blur(4px)';
  return new CSS2DObject(div);
}
function addBackplate(group, w, h){
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25, depthWrite: false })
  );
  plate.position.set(0,0,-0.2);
  group.add(plate);
}
function setGroupOpacity(group, alpha){
  group.traverse(obj=>{
    if (obj.isMesh && obj.material) {
      obj.material.transparent = true;
      obj.material.opacity = alpha;
      obj.material.needsUpdate = true;
    }
  });
}

/* =========== Input Plane =========== */
function makeInputPlane(){
  const geom = new THREE.PlaneGeometry(14,14);
  const texture = makeOpaqueTexture(28,28);
  const mat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);

  const inputLabel = makeLabel('Input (28×28)');
  inputLabel.position.set(0, 8, 0);
  mesh.add(inputLabel);

  return { mesh, texture, mat };
}
const inputPlane = makeInputPlane();

/* =========== Conv1 =========== */
const FM1_W=24, FM1_H=24, CONV1_C=8;
const tiles1 = [];
const group1 = new THREE.Group();
(function(){
  const cols=4, rows=Math.ceil(CONV1_C/cols);
  for(let c=0;c<CONV1_C;c++){
    const tex=makeOpaqueTexture(FM1_W, FM1_H);
    const mat=new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(10,10), mat);
    const ci=c%cols, ri=Math.floor(c/cols);
    mesh.position.set(ci*11 - (cols-1)*5.5, (rows-1)*5.5 - ri*11, 0);
    tiles1.push({mesh, tex}); group1.add(mesh);
  }
})();
addBackplate(group1, 28, 24);
const conv1Label = makeLabel('Conv1 — 8 @ 24×24 (edges)');
conv1Label.position.set(0,6,1);
group1.add(conv1Label);
scene.add(group1);

/* =========== Conv2 =========== */
const FM2_W=10, FM2_H=10, CONV2_C=16;
const tiles2 = [];
const group2 = new THREE.Group();
(function(){
  const cols=8, rows=Math.ceil(CONV2_C/cols);
  for(let c=0;c<CONV2_C;c++){
    const tex=makeOpaqueTexture(FM2_W, FM2_H);
    const mat=new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(6,6), mat);
    const ci=c%cols, ri=Math.floor(c/cols);
    mesh.position.set(ci*6.6 - (cols-1)*3.3, (rows-1)*3.3 - ri*6.6, 0);
    tiles2.push({mesh, tex}); group2.add(mesh);
  }
})();
addBackplate(group2, 40, 16);
const conv2Label = makeLabel('Conv2 — 16 @ 10×10 (parts)');
conv2Label.position.set(0,4,1);
group2.add(conv2Label);
scene.add(group2);

/* =========== Probability Bars =========== */
const classCount = 10;
const barGroup = new THREE.Group();
const valueLabels = [];
for(let i=0;i<classCount;i++){
  const m = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshPhongMaterial({ color: 0x66ccff }));
  // digit label
  const digit = makeLabel(String(i)); digit.element.style.padding='1px 4px'; digit.element.style.fontSize='11px';
  digit.position.set(0, -0.8, 0); m.add(digit);
  // numeric value label
  const val = makeLabel('0.00'); val.element.style.padding='1px 4px'; val.element.style.fontSize='11px';
  val.position.set(0, 0.7, 0); m.add(val);
  valueLabels[i] = val;

  barGroup.add(m);
}
for(let i=0;i<classCount;i++) barGroup.children[i].position.x = i*1.4;
const probHeader = makeLabel('Class Probabilities');
probHeader.position.set(-2, 8, 0);
barGroup.add(probHeader);
scene.add(barGroup);

/* =========== Data-flow pulses =========== */
const pulses = [];
function makePulse(a, b){
  const geom = new THREE.BufferGeometry().setFromPoints([a.clone(), b.clone()]);
  const mat  = new THREE.LineBasicMaterial({ color: 0x66ccff, transparent:true, opacity: 0.9 });
  const line = new THREE.Line(geom, mat);
  line.userData.t0 = performance.now();
  scene.add(line);
  pulses.push(line);
}
function updatePulses(){
  const now = performance.now();
  for (let i=pulses.length-1;i>=0;i--){
    const L = pulses[i];
    const age = now - L.userData.t0;
    const life = 550;
    const k = 1 - Math.min(1, age/life);
    L.material.opacity = k*k;
    if (age > life){ scene.remove(L); pulses.splice(i,1); }
  }
}

/* =========== Worker Bridge =========== */
const worker = new Worker(new URL('./net.worker.js', import.meta.url), { type: 'module' });

let shiftHeld=false;
worker.onmessage = (e) => {
  const { type, payload } = e.data || {};
  if (type === 'error'){ console.error('[Worker ERROR]', payload); return; }
  if (type !== 'activations') return;

  const { inputRGBA, conv1RGBA, conv2RGBA, conv2TopK, probs } = payload;

  // input
  inputPlane.texture.image.data.set(inputRGBA);
  inputPlane.texture.needsUpdate = true;

  // conv1 tiles
  for (let c=0;c<Math.min(CONV1_C, conv1RGBA.length); c++){
    tiles1[c].tex.image.data.set(conv1RGBA[c]);
    tiles1[c].tex.needsUpdate = true;
  }

  // conv2 tiles
  for (let c=0;c<Math.min(CONV2_C, conv2RGBA.length); c++){
    tiles2[c].tex.image.data.set(conv2RGBA[c]);
    tiles2[c].tex.needsUpdate = true;
  }

  // Top-K isolation for Conv2 while Shift is held
  if (shiftHeld && conv2TopK){
    const show = new Set(conv2TopK);
    tiles2.forEach((t, idx)=>{ t.mesh.visible = show.has(idx); });
  } else {
    tiles2.forEach(t=> t.mesh.visible = true);
  }

  // bars + numeric labels
  probs.forEach((p,i)=>{
    const h = Math.max(0.2, p*12);
    const mesh = barGroup.children[i];
    mesh.scale.y = THREE.MathUtils.lerp(mesh.scale.y || 1, h, 0.25);
    mesh.position.y = -10 + mesh.scale.y/2;
    const lbl = valueLabels[i];
    lbl.element.textContent = p.toFixed(2);
    lbl.position.y = mesh.scale.y + 0.2;
  });

  // Pulse along pipeline
  const pA = inputPlane.mesh.getWorldPosition(new THREE.Vector3());
  const pB = group1.getWorldPosition(new THREE.Vector3());
  const pC = group2.getWorldPosition(new THREE.Vector3());
  const pD = barGroup.getWorldPosition(new THREE.Vector3());
  makePulse(pA, pB); makePulse(pB, pC); makePulse(pC, pD);
};

/* =========== Hi-res Drawing Pad (224×224) → 28×28 =========== */
const inputCanvas = document.getElementById('inputCanvas');
const ictx = inputCanvas.getContext('2d', { willReadFrequently: true });
ictx.fillStyle = '#000'; ictx.fillRect(0,0,inputCanvas.width,inputCanvas.height);

const down = document.createElement('canvas'); down.width = 28; down.height = 28;
const dctx = down.getContext('2d', { willReadFrequently: true });

let drawing=false, last=null, frozen=false, stepOnce=false;

ictx.lineCap='round'; ictx.lineJoin='round'; ictx.strokeStyle='#fff'; ictx.lineWidth=16;

function toLocal(e){ const r=inputCanvas.getBoundingClientRect(); return {x:e.clientX-r.left, y:e.clientY-r.top}; }
function sendFrame(){
  if (frozen && !stepOnce) return;
  stepOnce = false;
  dctx.clearRect(0,0,28,28);
  dctx.drawImage(inputCanvas, 0, 0, 28, 28);
  const img = dctx.getImageData(0,0,28,28);
  worker.postMessage({ type:'infer', payload: img.data.buffer }, [img.data.buffer]);
}

// Clear button + hotkey
function clearInput() {
  ictx.fillStyle = '#000';
  ictx.fillRect(0, 0, inputCanvas.width, inputCanvas.height);
  dctx.clearRect(0, 0, 28, 28);
  const img = dctx.getImageData(0, 0, 28, 28);
  worker.postMessage({ type: 'infer', payload: img.data.buffer }, [img.data.buffer]);
}
document.getElementById('clearBtn').addEventListener('click', clearInput);

inputCanvas.addEventListener('mousedown', e=>{ drawing=true; last=toLocal(e); });
window.addEventListener('mouseup', ()=>{ drawing=false; last=null; sendFrame(); });
inputCanvas.addEventListener('mousemove', e=>{
  if (!drawing) return;
  const p=toLocal(e);
  ictx.beginPath(); ictx.moveTo(last.x,last.y); ictx.lineTo(p.x,p.y); ictx.stroke();
  last=p; sendFrame();
});

/* =========== Conv1 Receptive Field Highlight (hover) =========== */
const raycaster = new THREE.Raycaster();
const mouseNDC = new THREE.Vector2();
const rfHighlight = new THREE.Mesh(
  new THREE.PlaneGeometry(2.5, 2.5), // 5/28 of 14 = 2.5 units on the input plane
  new THREE.MeshBasicMaterial({ color: 0xffff00, transparent:true, opacity:0.25, depthWrite:false })
);
rfHighlight.visible = false;
inputPlane.mesh.add(rfHighlight);

function updateRFHighlight(intersect){
  const obj = intersect.object;
  const tileIdx = tiles1.findIndex(t => t.mesh === obj);
  if (tileIdx < 0) { rfHighlight.visible = false; return; }
  const uv = intersect.uv; // (0..1)
  const px = Math.max(0, Math.min(FM1_W-1, Math.floor(uv.x * FM1_W)));
  const py = Math.max(0, Math.min(FM1_H-1, Math.floor((1-uv.y) * FM1_H)));
  // Conv1 k=5,s=1 => a 5x5 patch on input; map to local plane coords (-7..+7)
  const planeHalf = 7, scale = 14/28;
  const cx = px + 2.5, cy = py + 2.5;
  const lx = (cx * scale) - planeHalf;
  const ly = planeHalf - (cy * scale);
  rfHighlight.position.set(lx, ly, 0.02);
  rfHighlight.visible = true;
}

window.addEventListener('mousemove', (e)=>{
  mouseNDC.x =  (e.clientX / window.innerWidth) * 2 - 1;
  mouseNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObjects(tiles1.map(t=>t.mesh), false);
  if (hits.length) updateRFHighlight(hits[0]); else rfHighlight.visible = false;
});

/* =========== Hotkeys =========== */
window.addEventListener('keydown', (e)=>{
  if (e.key === '1') group1.visible = !group1.visible;
  if (e.key === '2') group2.visible = !group2.visible;

  if (e.key === 'F' || e.key === 'f'){
    const d1 = camera.position.distanceTo(group1.getWorldPosition(new THREE.Vector3()));
    const d2 = camera.position.distanceTo(group2.getWorldPosition(new THREE.Vector3()));
    const focus = d1 < d2 ? group1 : group2;
    const other = d1 < d2 ? group2 : group1;
    setGroupOpacity(focus, 1.0);
    setGroupOpacity(other, 0.15);
    setTimeout(()=>{ setGroupOpacity(group1,1); setGroupOpacity(group2,1); }, 1500);
  }

  if (e.key === ' ') { e.preventDefault(); frozen = !frozen; }
  if (e.key === 'ArrowRight'){ stepOnce = true; sendFrame(); }
  if (e.key === 'c' || e.key === 'C'){ clearInput(); }

  // Camera presets along the diagonal
  if (e.key === 'q' || e.key === 'Q'){ // front-ish
    const centerX = computeLeftAnchor() + LAYOUT.gapInputToC1 + LAYOUT.gapC1ToC2 * 0.5;
    const centerZ = -LAYOUT.depthStep * 1.5;
    camera.position.set(centerX, 22, centerZ + 60);
    controls.target.set(centerX, 0, centerZ);
  }
  if (e.key === 'w' || e.key === 'W'){ // three-quarter
    const centerX = computeLeftAnchor() + LAYOUT.gapInputToC1 + LAYOUT.gapC1ToC2;
    const centerZ = -LAYOUT.depthStep * 2.0;
    camera.position.set(centerX + 32, 26, centerZ + 46);
    controls.target.set(centerX, 0, centerZ);
  }
  if (e.key === 'e' || e.key === 'E'){ // top
    const centerX = computeLeftAnchor() + LAYOUT.gapInputToC1 + LAYOUT.gapC1ToC2 * 0.5;
    const centerZ = -LAYOUT.depthStep * 1.5;
    camera.position.set(centerX, 80, centerZ + 0.0001);
    controls.target.set(centerX, 0, centerZ);
  }

  if (e.key === 'Shift') shiftHeld = true;
});
window.addEventListener('keyup', (e)=>{ if (e.key === 'Shift') shiftHeld = false; });

/* =========== Resize & Render =========== */
function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  layoutPipeline();
}
window.addEventListener('resize', onResize);

function tick(){
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  updatePulses();
  requestAnimationFrame(tick);
}

/* =========== Boot =========== */
layoutPipeline();        // compute positions
framePipelineFront();    // center camera on pipeline
tick();
sendFrame();             // make textures visible immediately
