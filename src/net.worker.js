// src/net.worker.js
import * as ort from 'onnxruntime-web';

// ORT wasm config (CDN serves correct MIME + CORS for Pages)
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.numThreads = 1;

let session = null;
let ready = false;

// --------- Viridis LUT (same as before) ----------
const viridis = new Uint8Array(256 * 4);
(function buildViridis() {
  const stops = [
    [0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],[0.207,0.372,0.553],
    [0.164,0.471,0.558],[0.128,0.567,0.551],[0.135,0.659,0.518],[0.267,0.748,0.441],
    [0.478,0.821,0.318],[0.741,0.873,0.150],[0.993,0.906,0.144],[0.993,0.746,0.239],
    [0.988,0.552,0.349],[0.940,0.364,0.432],[0.839,0.190,0.502],[0.616,0.040,0.475]
  ];
  const L = stops.length;
  const lerp = (a,b,t)=>a+(b-a)*t;
  for(let i=0;i<256;i++){
    const t = i/255, x=t*(L-1), i0=Math.floor(x), i1=Math.min(L-1,i0+1), f=x-i0;
    const r=lerp(stops[i0][0],stops[i1][0],f), g=lerp(stops[i0][1],stops[i1][1],f), b=lerp(stops[i0][2],stops[i1][2],f);
    const j=i*4; viridis[j]=Math.round(r*255); viridis[j+1]=Math.round(g*255); viridis[j+2]=Math.round(b*255); viridis[j+3]=255;
  }
})();

function rgbaFromGrayWithLUT(arr, w, h, lut=viridis) {
  const out = new Uint8Array(w*h*4);
  for (let i=0;i<w*h;i++){
    const idx = Math.max(0, Math.min(255, (arr[i]*255)|0)) * 4;
    const j = i*4;
    out[j]   = lut[idx];
    out[j+1] = lut[idx+1];
    out[j+2] = lut[idx+2];
    out[j+3] = 255;
  }
  return out;
}

function packFeatureMaps(t /* [1,C,H,W] */) {
  const C=t.dims[1], H=t.dims[2], W=t.dims[3];
  const out = new Array(C);
  const means = new Float32Array(C);
  for (let c=0;c<C;c++){
    const chan = new Float32Array(W*H);
    let sum=0, mn=Infinity, mx=-Infinity;
    for (let y=0;y<H;y++){
      for (let x=0;x<W;x++){
        const v = t.data[((c*H + y)*W) + x];
        chan[y*W + x] = v; sum += v; if (v<mn) mn=v; if (v>mx) mx=v;
      }
    }
    means[c] = sum/(W*H);
    const rng=(mx-mn)||1; for (let i=0;i<chan.length;i++) chan[i]=(chan[i]-mn)/rng;
    out[c] = rgbaFromGrayWithLUT(chan, W, H);
  }
  const idx = [...Array(C).keys()].sort((a,b)=>means[b]-means[a]);
  return { rgba: out, topk: idx.slice(0, Math.min(6,C)) };
}
function softmaxFromLogits(arr){
  const m = Math.max(...arr);
  const exps = arr.map(v=>Math.exp(v-m));
  const s = exps.reduce((a,b)=>a+b,0)||1;
  return exps.map(v=>v/s);
}

// ---- Init when main thread provides the model URL ----
async function init(modelUrl) {
  try {
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    ready = true;
    postMessage({ type: 'ready' });
  } catch (err) {
    postMessage({ type: 'error', payload: `Init failed: ${err}` });
  }
}

// ---- Message handler ----
onmessage = async (e) => {
  const { type, modelUrl, payload } = e.data || {};

  if (type === 'init') {
    return init(modelUrl);
  }

  if (type === 'infer') {
    if (!ready) return; // ignore until model is ready
    try {
      // payload: RGBA Uint8 (28x28)
      const rgba = new Uint8Array(payload);
      const gray = new Float32Array(28*28);
      for (let i=0;i<28*28;i++){
        const r=rgba[i*4], g=rgba[i*4+1], b=rgba[i*4+2];
        gray[i] = (0.2126*r + 0.7152*g + 0.0722*b)/255;
      }
      const input = new ort.Tensor('float32', gray, [1,1,28,28]);

      const results = await session.run({ input });
      const probsT = results['probs'];   // [1,10]
      const conv1T = results['conv1'];   // [1,8,24,24]
      const conv2T = results['conv2'];   // [1,16,10,10]

      const inputRGBA = rgbaFromGrayWithLUT(gray, 28, 28);
      const { rgba: conv1RGBA } = packFeatureMaps(conv1T);
      const { rgba: conv2RGBA, topk: conv2TopK } = packFeatureMaps(conv2T);
      const probs = softmaxFromLogits(Array.from(probsT.data));

      postMessage(
        { type: 'activations', payload: { inputRGBA, conv1RGBA, conv2RGBA, conv2TopK, probs } },
        [ inputRGBA.buffer, ...conv1RGBA.map(a=>a.buffer), ...conv2RGBA.map(a=>a.buffer) ]
      );
    } catch (err) {
      postMessage({ type: 'error', payload: `Infer failed: ${err}` });
    }
  }
};
