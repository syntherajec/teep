/**
 * removeBg.js — Background Removal Module (v2 — Premium Edge Quality)
 *
 * Pipeline:
 *  1. Try U²Net ONNX (if model file present)
 *  2. Fallback: advanced GrabCut-style color + luminance clustering
 *
 * Post-processing (always applied, both paths):
 *  A. Guided-filter-style alpha matting (edge sharpening)
 *  B. White-fringe / color-spill removal (de-matting)
 *  C. Morphological erosion to clean stray border pixels
 *  D. Feathered edge softening for natural look
 *
 * Returns: { imageData: ImageData, usedFallback: boolean }
 */

let u2netSession = null;
let modelLoaded  = false;
let modelTried   = false;

/* ─── U²Net loader ──────────────────────────────────────────────────────── */
export async function loadU2Net(onStep) {
  if (modelLoaded || modelTried) return modelLoaded;
  modelTried = true;
  try {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime not available');
    onStep?.('Loading U2Net model…');
    u2netSession = await ort.InferenceSession.create('./models/u2net.onnx');
    modelLoaded = true;
    onStep?.('U2Net model loaded!');
    return true;
  } catch (e) {
    console.warn('[removeBg] U2Net not available, using fallback:', e.message);
    return false;
  }
}

/* ─── Public entry point ─────────────────────────────────────────────────── */
export async function removeBackground(imageData, onStep) {
  const loaded = await loadU2Net(onStep);

  let rawAlpha;
  let usedFallback;

  if (loaded && u2netSession) {
    try {
      rawAlpha     = await inferU2Net(imageData, onStep);
      usedFallback = false;
    } catch (e) {
      console.warn('[removeBg] ONNX inference failed, using fallback:', e);
      onStep?.('Using smart edge-detection fallback…');
      rawAlpha     = computeFallbackAlpha(imageData);
      usedFallback = true;
    }
  } else {
    onStep?.('Using smart edge-detection fallback…');
    rawAlpha     = computeFallbackAlpha(imageData);
    usedFallback = true;
  }

  onStep?.('Refining edges & removing white fringe…');
  const refined = postProcessAlpha(rawAlpha, imageData, onStep);

  return { imageData: refined, usedFallback };
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 1 — U²Net ONNX inference
   Returns Float32Array alpha [0..1] length = W×H
═══════════════════════════════════════════════════════════════════════════ */
async function inferU2Net(imageData, onStep) {
  const { width, height } = imageData;
  const SIZE = 320;

  onStep?.('Preprocessing for U2Net…');
  const resized = resizeImageData(imageData, SIZE, SIZE);

  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const tensor = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    tensor[i]              = ((resized[i * 4]     / 255) - mean[0]) / std[0];
    tensor[i + SIZE * SIZE]    = ((resized[i * 4 + 1] / 255) - mean[1]) / std[1];
    tensor[i + SIZE * SIZE * 2] = ((resized[i * 4 + 2] / 255) - mean[2]) / std[2];
  }

  onStep?.('Running U2Net inference…');
  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]);
  const feeds = {};
  feeds[u2netSession.inputNames[0]] = inputTensor;
  const results = await u2netSession.run(feeds);
  const output  = results[u2netSession.outputNames[0]].data;

  // Normalise to [0..1]
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < output.length; i++) { if (output[i] < mn) mn = output[i]; if (output[i] > mx) mx = output[i]; }
  const range = mx - mn || 1;

  // Upscale mask back to original dimensions using bilinear
  const alpha = new Float32Array(width * height);
  const scaleX = SIZE / width;
  const scaleY = SIZE / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = x * scaleX;
      const sy = y * scaleY;
      const x0 = Math.min(Math.floor(sx), SIZE - 2);
      const y0 = Math.min(Math.floor(sy), SIZE - 2);
      const fx = sx - x0, fy = sy - y0;
      const v  = (output[y0 * SIZE + x0]         * (1-fx)*(1-fy)
                + output[y0 * SIZE + x0 + 1]       * fx*(1-fy)
                + output[(y0+1) * SIZE + x0]       * (1-fx)*fy
                + output[(y0+1) * SIZE + x0 + 1]   * fx*fy);
      alpha[y * width + x] = (v - mn) / range;
    }
  }
  return alpha;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 2 — Advanced fallback: multi-sample background modelling
   Returns Float32Array alpha [0..1]
═══════════════════════════════════════════════════════════════════════════ */
function computeFallbackAlpha(imageData) {
  const { width, height, data } = imageData;
  const N = width * height;

  /* --- Sample background colours from border band + corners --- */
  const borderSamples = [];
  const BAND = Math.max(2, Math.round(Math.min(width, height) * 0.03));
  for (let x = 0; x < width; x++) {
    for (let b = 0; b < BAND; b++) {
      pushSample(borderSamples, data, (b * width + x) * 4);
      pushSample(borderSamples, data, (((height - 1 - b) * width) + x) * 4);
    }
  }
  for (let y = BAND; y < height - BAND; y++) {
    for (let b = 0; b < BAND; b++) {
      pushSample(borderSamples, data, (y * width + b) * 4);
      pushSample(borderSamples, data, (y * width + (width - 1 - b)) * 4);
    }
  }

  /* --- Compute background colour model (mean + variance) --- */
  const bgR = avg(borderSamples, 0);
  const bgG = avg(borderSamples, 1);
  const bgB = avg(borderSamples, 2);

  const varR = variance(borderSamples, 0, bgR);
  const varG = variance(borderSamples, 1, bgG);
  const varB = variance(borderSamples, 2, bgB);

  // Adaptive threshold: tighter when bg is uniform, looser when noisy
  const bgNoise = Math.sqrt((varR + varG + varB) / 3);
  const BASE_THRESH = 35;
  const threshold = Math.max(BASE_THRESH, Math.min(90, BASE_THRESH + bgNoise * 1.5));

  /* --- Per-pixel foreground probability --- */
  const alpha = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Perceptual weighted colour distance
    const dist = Math.sqrt(
      (r - bgR) ** 2 * 0.2126 +
      (g - bgG) ** 2 * 0.7152 +
      (b - bgB) ** 2 * 0.0722
    );
    alpha[i] = Math.min(1, dist / threshold);
  }

  /* --- Guided bilateral-style alpha refinement --- */
  const blurred = separableGaussian(alpha, width, height, 3);

  // Sharpen transition zone
  for (let i = 0; i < N; i++) {
    const a = blurred[i];
    // S-curve: compress near-transparent and near-opaque, keep transition smooth
    blurred[i] = sCurve(a, 0.15, 0.85);
  }

  return blurred;
}

function pushSample(arr, data, offset) {
  arr.push([data[offset], data[offset + 1], data[offset + 2]]);
}
function avg(samples, ch) { return samples.reduce((s, c) => s + c[ch], 0) / (samples.length || 1); }
function variance(samples, ch, mean) { return samples.reduce((s, c) => s + (c[ch] - mean) ** 2, 0) / (samples.length || 1); }

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 3 — Post-processing pipeline (both paths)
═══════════════════════════════════════════════════════════════════════════ */
function postProcessAlpha(rawAlpha, imageData, onStep) {
  const { width, height, data } = imageData;
  const N = width * height;

  /* 3-A. Guided-filter-style edge sharpening
         Use local mean/variance of the source luminance to pull alpha
         toward hard edges in the source image */
  const luma = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    luma[i] = (data[i * 4] * 0.2126 + data[i * 4 + 1] * 0.7152 + data[i * 4 + 2] * 0.0722) / 255;
  }

  const GF_RADIUS = 4;
  const GF_EPS    = 0.01;
  const guidedAlpha = guidedFilter(rawAlpha, luma, width, height, GF_RADIUS, GF_EPS);

  /* 3-B. Morphological "clean border" erosion
         Slightly erode the mask to eat away stray 1-pixel white halos */
  const eroded = morphErode(guidedAlpha, width, height, 1);

  /* 3-C. Second guided pass (tighter radius) to re-sharpen after erosion */
  const sharpened = guidedFilter(eroded, luma, width, height, 2, 0.001);

  /* 3-D. White-fringe / colour de-matting
         For every semi-transparent pixel, remove the estimated background
         colour contribution using alpha pre-multiply / un-pre-multiply
         (Knockback / "Despill") */
  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  // Estimate background colour from near-zero-alpha pixels (border)
  let bgR = 0, bgG = 0, bgB = 0, bgCnt = 0;
  for (let i = 0; i < N; i++) {
    if (sharpened[i] < 0.08) {
      bgR += data[i * 4];
      bgG += data[i * 4 + 1];
      bgB += data[i * 4 + 2];
      bgCnt++;
    }
  }
  if (bgCnt === 0) { bgR = 255; bgG = 255; bgB = 255; bgCnt = 1; } // assume white
  bgR /= bgCnt; bgG /= bgCnt; bgB /= bgCnt;

  for (let i = 0; i < N; i++) {
    let a = sharpened[i];

    // Apply S-curve to make alpha crisper (avoid grey fringe)
    a = sCurve(a, 0.12, 0.88);

    if (a <= 0) {
      output.data[i * 4 + 3] = 0;
      continue;
    }
    if (a >= 1) {
      output.data[i * 4 + 3] = 255;
      continue;
    }

    // De-fringe: un-premultiply background colour from semi-transparent pixels
    // Formula: C_fg = (C_composite - C_bg * (1 - alpha)) / alpha
    const r0 = data[i * 4];
    const g0 = data[i * 4 + 1];
    const b0 = data[i * 4 + 2];

    const oneMinusA = 1 - a;
    let fr = (r0 - bgR * oneMinusA) / a;
    let fg = (g0 - bgG * oneMinusA) / a;
    let fb = (b0 - bgB * oneMinusA) / a;

    // Clamp — small overflows from noise, not true signal
    output.data[i * 4]     = Math.min(255, Math.max(0, Math.round(fr)));
    output.data[i * 4 + 1] = Math.min(255, Math.max(0, Math.round(fg)));
    output.data[i * 4 + 2] = Math.min(255, Math.max(0, Math.round(fb)));
    output.data[i * 4 + 3] = Math.min(255, Math.max(0, Math.round(a * 255)));
  }

  return output;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════════════════════ */

/**
 * Fast separable Gaussian blur on a Float32Array alpha map
 */
function separableGaussian(alpha, w, h, radius) {
  const kernel = buildGaussKernel(radius);
  const klen   = kernel.length;
  const khalf  = Math.floor(klen / 2);
  const tmp    = new Float32Array(alpha.length);
  const out    = new Float32Array(alpha.length);

  // Horizontal pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, wsum = 0;
      for (let k = 0; k < klen; k++) {
        const nx = x + k - khalf;
        if (nx >= 0 && nx < w) { sum += alpha[y * w + nx] * kernel[k]; wsum += kernel[k]; }
      }
      tmp[y * w + x] = sum / wsum;
    }
  }
  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, wsum = 0;
      for (let k = 0; k < klen; k++) {
        const ny = y + k - khalf;
        if (ny >= 0 && ny < h) { sum += tmp[ny * w + x] * kernel[k]; wsum += kernel[k]; }
      }
      out[y * w + x] = sum / wsum;
    }
  }
  return out;
}

function buildGaussKernel(radius) {
  const size = radius * 2 + 1;
  const k    = new Float32Array(size);
  const sigma = Math.max(radius / 2, 0.5);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < size; i++) k[i] /= sum;
  return k;
}

/**
 * Guided filter — aligns a floating alpha map to edges in a guide image
 * He et al. 2013 simplified box-filter implementation
 */
function guidedFilter(alpha, guide, w, h, r, eps) {
  const N = w * h;

  // Box-filter mean helpers
  const boxMean = (arr) => {
    const out = new Float32Array(N);
    const tmp = new Float32Array(N);
    const d   = r * 2 + 1;
    // Horizontal
    for (let y = 0; y < h; y++) {
      let sum = 0, cnt = 0;
      for (let x = 0; x < Math.min(r + 1, w); x++) { sum += arr[y * w + x]; cnt++; }
      for (let x = 0; x < w; x++) {
        if (x + r < w)   { sum += arr[y * w + x + r]; cnt++; }
        if (x - r - 1 >= 0) { sum -= arr[y * w + x - r - 1]; cnt--; }
        tmp[y * w + x] = sum / cnt;
      }
    }
    // Vertical
    for (let x = 0; x < w; x++) {
      let sum = 0, cnt = 0;
      for (let y = 0; y < Math.min(r + 1, h); y++) { sum += tmp[y * w + x]; cnt++; }
      for (let y = 0; y < h; y++) {
        if (y + r < h)   { sum += tmp[(y + r) * w + x]; cnt++; }
        if (y - r - 1 >= 0) { sum -= tmp[(y - r - 1) * w + x]; cnt--; }
        out[y * w + x] = sum / cnt;
      }
    }
    return out;
  };

  const meanI  = boxMean(guide);
  const meanP  = boxMean(alpha);
  const corrI  = boxMean(guide.map((v, i) => v * v));
  const corrIP = boxMean(guide.map((v, i) => v * alpha[i]));

  const A = new Float32Array(N);
  const B = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const varI = corrI[i] - meanI[i] * meanI[i];
    const covIP = corrIP[i] - meanI[i] * meanP[i];
    A[i] = covIP / (varI + eps);
    B[i] = meanP[i] - A[i] * meanI[i];
  }

  const meanA = boxMean(A);
  const meanB = boxMean(B);

  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = Math.min(1, Math.max(0, meanA[i] * guide[i] + meanB[i]));
  }
  return out;
}

/**
 * Morphological erosion — shrinks bright (foreground) regions
 * Removes thin "halo" pixels at the edge of the mask
 */
function morphErode(alpha, w, h, radius) {
  const out = new Float32Array(alpha.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let minVal = 1;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            const v = alpha[ny * w + nx];
            if (v < minVal) minVal = v;
          }
        }
      }
      // Only erode in the transition band (leave solid interior alone)
      const a = alpha[y * w + x];
      out[y * w + x] = a > 0.95 ? a : Math.min(a, minVal + 0.05);
    }
  }
  return out;
}

/**
 * Smooth S-curve remapping [0..1] → [0..1]
 * Pixels below `lo` → pushed toward 0; above `hi` → pushed toward 1
 * Transition zone stays smooth
 */
function sCurve(v, lo, hi) {
  if (v <= lo) return v * (0.5 / lo) * (v / lo);     // fast falloff to 0
  if (v >= hi) {
    const t = (v - hi) / (1 - hi);
    return hi + (1 - hi) * (1 - (1 - t) * (1 - t));  // fast rise to 1
  }
  // Smooth in transition band — keep linear
  return lo + (v - lo) * ((hi - lo) / (hi - lo));    // = v in band (no-op)
}

/**
 * Resize raw RGBA data to target dimensions using nearest-neighbour
 * (fast — used only for the 320×320 U²Net thumbnail)
 */
function resizeImageData(imageData, tw, th) {
  const { width: sw, height: sh, data } = imageData;
  const out = new Uint8ClampedArray(tw * th * 4);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(Math.floor(x * sw / tw), sw - 1);
      const sy = Math.min(Math.floor(y * sh / th), sh - 1);
      const si = (sy * sw + sx) * 4;
      const di = (y * tw + x) * 4;
      out[di]     = data[si];
      out[di + 1] = data[si + 1];
      out[di + 2] = data[si + 2];
      out[di + 3] = data[si + 3];
    }
  }
  return out;
}
