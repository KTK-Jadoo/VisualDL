// net.worker.js
import * as ort from 'onnxruntime-web';

let session;
let ready = false;

// Example: MNIST-like tiny CNN exported to ONNX with named outputs: conv1, probs
async function init() {
    session = await ort.InferenceSession.create('/public/model.onnx', {
        executionProviders: ['wasm'], // swap to 'webgl' if using tf.js; ort-web uses WASM + SIMD
        graphOptimizationLevel: 'all'
    });
    ready = true;
}
init();

function rgbaFromGrayFloat32(arr, w, h, scale = 255) {
    // maps [0,1] to RGBA (A=255)
    const out = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
        const v = Math.max(0, Math.min(1, arr[i])) * scale;
        const j = i * 4;
        out[j] = out[j + 1] = out[j + 2] = v | 0;
        out[j + 3] = 255;
    }
    return out;
}

onmessage = async (e) => {
    const { type, payload } = e.data;
    if (type !== 'infer' || !ready) return;

    // payload is RGBA(28x28) Uint8
    const rgba = new Uint8Array(payload);
    // Convert to normalized grayscale 1x1x28x28
    const gray = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // simple luminance from RGBA
        const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
        gray[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    }

    const input = new ort.Tensor('float32', gray, [1, 1, 28, 28]);
    const results = await session.run({ 'input': input });

    // Expect results: 
    //   conv1: [1, C, H, W]
    //   probs: [1, 10]
    const conv1 = results['conv1'];
    const probs = results['probs'];

    const C = conv1.dims[1], H = conv1.dims[2], W = conv1.dims[3];
    const conv1RGBA = [];
    // Per-channel min/max normalize for display
    for (let c = 0; c < C; c++) {
        const chan = new Float32Array(W * H);
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const idx = ((0 * C + c) * H + y) * W + x;
                chan[y * W + x] = conv1.data[idx];
            }
        }
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < chan.length; i++) { const v = chan[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        const rng = (mx - mn) || 1.0;
        for (let i = 0; i < chan.length; i++) chan[i] = (chan[i] - mn) / rng;
        conv1RGBA.push(rgbaFromGrayFloat32(chan, W, H));
    }

    const probsArr = Array.from(probs.data);
    // Normalize softmax in case model outputs logits
    const maxP = Math.max(...probsArr);
    const exps = probsArr.map(v => Math.exp(v - maxP));
    const sum = exps.reduce((a, b) => a + b, 0);
    const soft = exps.map(v => v / sum);

    const inputRGBA = rgbaFromGrayFloat32(gray, 28, 28);

    postMessage({
        type: 'activations',
        payload: { inputRGBA, conv1RGBA, probs: soft }
    }, [inputRGBA.buffer, ...conv1RGBA.map(a => a.buffer)]);
};
