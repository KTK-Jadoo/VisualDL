# VisualDL · 3D CNN Viewer

An educational and aesthetic visualization of a small convolutional neural network that runs entirely in the browser. The application provides a high-resolution drawing pad, performs model inference in a Web Worker using a WebAssembly backend, and renders the input, intermediate feature maps, and class probabilities in a Three.js scene that you can orbit and inspect.

The intent is clarity and responsiveness. Clarity is achieved by making invisible tensors spatial and color-mapped. Responsiveness is achieved by separating rendering from inference and by updating GPU textures in place rather than recreating geometry or materials each frame.

---

## Live demo

This repository is designed to be served from a GitHub Pages project site. Add your deployment URL here once you enable Pages for the `gh-pages` branch.

---

## Design rationale

### Inference in a Web Worker
Keeping inference off the main thread preserves smooth camera controls and pointer interactions while you draw. The Worker receives only the 28×28 RGBA buffer as a transferable `ArrayBuffer` and returns compact RGBA tiles and probabilities. This avoids blocking layout and avoids large structured clones.

### WebAssembly execution
`onnxruntime-web` with the wasm execution provider runs in every modern browser without native dependencies. This choice maximizes portability and makes the demo a single static site. Graph optimizations are enabled to reduce runtime overhead.

### Color-mapped activations
Per-channel min–max normalization followed by a small viridis lookup table exposes structure that grayscale often hides. The lookup is a 256×1 RGBA table precomputed once in the Worker. Updating `DataTexture.image.data` with LUT-mapped bytes is inexpensive and visually expressive.

### Stable, explicit layout
The pipeline is laid out left-to-right and also steps backward along the Z-axis. This diagonal arrangement prevents the familiar “stacking into a line” that occurs when the camera is aligned with a single axis. The layout is computed from a small set of gaps, and the camera presets frame that row consistently on load.

### Update textures, not meshes
Input, Conv1, and Conv2 are planes textured with `THREE.DataTexture`. On each inference, only the pixel buffers change and `needsUpdate` is set. Meshes, geometries, and materials are created once and reused. This produces consistent frame times and avoids garbage collection spikes.

### Deployment-safe model loading
The main thread computes the correct absolute URL for `model.onnx` using `import.meta.env.BASE_URL` and passes it to the Worker. The Worker never assumes the site root. This detail is necessary when serving from a project path such as `/VisualDL/`.

---

## What happens when you draw

1. You draw on a 224×224 HTML canvas. For inference, the canvas is downsampled to 28×28. Luminance is computed per pixel and normalized to the unit interval.

2. The 28×28 RGBA buffer is transferred to the Worker. Transfer avoids copying and keeps the event loop responsive while you draw continuous strokes.

3. The Worker runs the ONNX model and reads three named outputs that this demo expects  
   a. `conv1` with shape `[1, 8, 24, 24]`  
   b. `conv2` with shape `[1, 16, 10, 10]`  
   c. `probs` with shape `[1, 10]`  

4. Each feature map channel is normalized independently, converted through the viridis lookup table to RGBA, and sent back. The main thread writes these bytes into existing `DataTexture` objects attached to the tiles.

5. Class probabilities update a small bar chart. Each bar shows the digit label below and the numeric probability above. The tallest bar represents the current prediction.

6. A brief pulse line animates from Input to Conv1 to Conv2 to the Bars to indicate dataflow without distracting from values.

---

## Interaction model

1. Rotate and zoom with the mouse. OrbitControls are enabled with damping for a steady feel.

2. Hover a Conv1 tile to highlight the corresponding 5×5 receptive-field footprint on the input plane. The footprint size is a UI assumption aligned with this repo’s training script; if you swap in a different model, adjust the footprint accordingly.

3. Hold Shift to show only the most active Conv2 channels for the current frame. The Worker computes per-channel means and returns the top indices.

4. Press Space to freeze updates and Right Arrow to step a single inference. This is useful when demonstrating cause and effect from small input edits.

5. Press C or click the Clear button to wipe the drawing pad and send a blank frame.

6. Press Q, W, or E to jump to front, three-quarter, or top views that frame the pipeline’s diagonal layout.

---

## Project structure

1. `index.html` mounts the Three.js canvas, hosts a compact HUD with instructions, and places the 224×224 drawing pad and Clear button.

2. `public/model.onnx` contains the exported ONNX model. Files in `public` are served at the site root in both development and production.

3. `src/main.js` builds the Three.js scene, creates textures and labels, manages input and layout, and communicates with the Worker. It performs no heavy tensor computation.

4. `src/net.worker.js` configures `onnxruntime-web`, loads the model from the absolute URL provided by the main thread, runs inference, normalizes and color-maps activations, and posts results back.

---

## Development

1. Install dependencies  
   `npm install`

2. Run locally  
   `npm run dev`  
   Vite serves the app with hot reload. The ONNX model is read from `public/model.onnx`.

3. Build and preview  
   `npm run build`  
   `npm run preview`  
   The `dist` directory contains a static site.

---

## Deployment

1. Vite is configured with `base: '/VisualDL/'` so that assets resolve under the repository path used by GitHub Pages.

2. The deployment script builds the site, copies `index.html` to `404.html` for SPA routing, and pushes `dist` to the `gh-pages` branch.  
   `npm run deploy`

3. In the repository settings, enable GitHub Pages for the `gh-pages` branch root. The site is then reachable at  
   `https://<your-username>.github.io/VisualDL/`

---

## Mathematical background

### Convolution and feature maps
A convolutional layer applies learned kernels to local neighborhoods. For output channel \(c_{\text{out}}\) at spatial index \((i,j)\) with kernel size \(k\) and stride \(s\),
\[
y[c_{\text{out}}, i, j] \;=\; \sum_{c_{\text{in}}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1}
w[c_{\text{out}}, c_{\text{in}}, u, v]\; x[c_{\text{in}}, i\cdot s + u, j\cdot s + v] \;+\; b[c_{\text{out}}].
\]
The demo visualizes two convolutional stages by showing each output channel as a small image tile after per-channel normalization.

### Receptive field intuition
The receptive field describes how many input pixels influence a single activation. If layer \(l\) has kernel size \(k_l\) and stride \(s_l\), the receptive field \(r_l\) and effective jump \(j_l\) evolve from \(r_0=1, j_0=1\) by
\[
r_l \;=\; r_{l-1} + (k_l - 1)\, j_{l-1}, \qquad
j_l \;=\; j_{l-1}\, s_l.
\]
The UI’s input overlay shows a 5×5 footprint for Conv1 to match the intended first kernel size for this demo. If you substitute a different model, adjust the overlay to the correct \(k_1\).

### Softmax for class probabilities
Given logits \(z_k\), probabilities are obtained by
\[
p_k \;=\; \frac{e^{z_k}}{\sum_j e^{z_j}}.
\]
Softmax is applied in the Worker to ensure that the bar chart always reflects a normalized distribution even if the model exports logits.

### Normalization for display
Each channel \(a\) is mapped to the unit interval before color mapping. With minimum \(m\) and maximum \(M\) over spatial positions,
\[
\hat{a}(i,j) \;=\; \frac{a(i,j) - m}{M - m + \varepsilon},
\]
with a small \(\varepsilon\) to avoid division by zero. Independent normalization is deliberate so that a high-variance channel does not visually suppress lower-variance channels.

---

## Troubleshooting

1. Panels do not update  
   Open the browser Network panel and confirm that `model.onnx` resolves with status 200 from the project path. The Worker logs a readiness message once the session is created. If you see a WebAssembly “magic number” error, verify that `onnxruntime-web` is pulled from a CDN and not served through an incorrect MIME type.

2. The scene appears empty after deployment  
   The camera is framed after layout. If you change `LAYOUT.gap*` or `depthStep` substantially, call the framing helper again at boot or adjust the preset positions.

3. Tiles look blurred  
   Ensure `NearestFilter` is used for both minification and magnification on all `DataTexture` objects so that small feature maps remain crisp.

---

## Notes on extensibility

1. You can replace `public/model.onnx` with any ONNX model that exposes named outputs for the intermediate tensors you wish to visualize. If the shapes or names differ, update the Worker’s `session.run` keys and the tiling geometry accordingly.

2. The receptive-field overlay and the top-K selection logic are independent of a specific dataset. They are driven by returned shapes and simple statistics and can be reused for other models with minimal changes.

3. The viridis lookup can be replaced by any 256-entry RGBA lookup. For qualitative palettes, ensure monotonic luminance to preserve structure.

---

## License and acknowledgment

This repository is intended for instruction and demonstration. The implementation follows common practices for Three.js visualization and `onnxruntime-web` inference and packages them into a single static site so that learners can explore CNN internals with nothing more than a browser.
