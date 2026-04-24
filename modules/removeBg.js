/**
 * removeBg.js — PRO SHARP + NO WHITE HALO
 * Target: Logo / Merch / Teepublic
 */

let u2netSession = null;
let modelLoaded = false;
let modelTried = false;

export async function loadU2Net(onStep) {
  if (modelLoaded || modelTried) return modelLoaded;
  modelTried = true;

  try {
    if (typeof ort === 'undefined') throw new Error('ONNX missing');

    onStep?.('Loading AI model...');
    u2netSession = await ort.InferenceSession.create('./models/u2net.onnx');

    modelLoaded = true;
    onStep?.('Model ready');
    return true;
  } catch (e) {
    console.warn('AI failed → fallback');
    return false;
  }
}

export async function removeBackground(imageData, onStep) {
  const loaded = await loadU2Net(onStep);

  let rawAlpha;

  if (loaded && u2netSession) {
    try {
      onStep?.('AI processing...');
      rawAlpha = await inferU2Net(imageData);
    } catch {
      rawAlpha = computeFallbackAlpha(imageData);
    }
  } else {
    rawAlpha = computeFallbackAlpha(imageData);
  }

  onStep?.('Cleaning edge...');
  return {
    imageData: postProcessPro(rawAlpha, imageData),
    usedFallback: !loaded
  };
}

/* ================= AI ================= */
async function inferU2Net(imageData) {
  const { width, height } = imageData;
  const SIZE = 320;

  const resized = resizeNN(imageData, SIZE, SIZE);

  const tensor = new Float32Array(3 * SIZE * SIZE);

  for (let i = 0; i < SIZE * SIZE; i++) {
    tensor[i] = resized[i * 4] / 255;
    tensor[i + SIZE * SIZE] = resized[i * 4 + 1] / 255;
    tensor[i + SIZE * SIZE * 2] = resized[i * 4 + 2] / 255;
  }

  const input = new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]);
  const feeds = {};
  feeds[u2netSession.inputNames[0]] = input;

  const res = await u2netSession.run(feeds);
  const out = res[u2netSession.outputNames[0]].data;

  let min = Infinity, max = -Infinity;
  for (let i = 0; i < out.length; i++) {
    if (out[i] < min) min = out[i];
    if (out[i] > max) max = out[i];
  }

  const range = max - min || 1;

  const alpha = new Float32Array(width * height);
  const sx = SIZE / width;
  const sy = SIZE / height;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = Math.floor(x * sx);
      const py = Math.floor(y * sy);

      const v = out[py * SIZE + px];
      alpha[y * width + x] = (v - min) / range;
    }
  }

  return alpha;
}

/* ================= FALLBACK ================= */
function computeFallbackAlpha(imageData) {
  const { width, height, data } = imageData;
  const N = width * height;

  let bgR = 0, bgG = 0, bgB = 0, count = 0;

  for (let x = 0; x < width; x++) {
    const i1 = x * 4;
    const i2 = ((height - 1) * width + x) * 4;

    bgR += data[i1];
    bgG += data[i1 + 1];
    bgB += data[i1 + 2];

    bgR += data[i2];
    bgG += data[i2 + 1];
    bgB += data[i2 + 2];

    count += 2;
  }

  bgR /= count;
  bgG /= count;
  bgB /= count;

  const alpha = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    const dist = Math.sqrt(
      (r - bgR) ** 2 +
      (g - bgG) ** 2 +
      (b - bgB) ** 2
    );

    alpha[i] = dist > 35 ? 1 : 0;
  }

  return alpha;
}

/* ================= CORE FIX ================= */
function postProcessPro(rawAlpha, imageData) {
  const { width, height, data } = imageData;
  const N = width * height;

  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  const alpha = new Float32Array(N);

  /* 1. HARD + SMOOTH EDGE (ANTI JAGGED TANPA BLUR) */
  for (let i = 0; i < N; i++) {
    const a = rawAlpha[i];

    if (a > 0.6) alpha[i] = 1;
    else if (a < 0.3) alpha[i] = 0;
    else {
      // smooth tipis di edge saja
      alpha[i] = (a - 0.3) / (0.6 - 0.3);
    }
  }

  /* 2. REMOVE WHITE HALO (PALING PENTING) */
  let bgR = 255, bgG = 255, bgB = 255;

  for (let i = 0; i < N; i++) {
    const a = alpha[i];

    if (a === 0) {
      output.data[i * 4 + 3] = 0;
      continue;
    }

    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];

    // 🔥 REMOVE BACKGROUND COLOR BLEED
    const invA = 1 / Math.max(a, 0.0001);

    output.data[i * 4] = clamp255((r - bgR * (1 - a)) * invA);
    output.data[i * 4 + 1] = clamp255((g - bgG * (1 - a)) * invA);
    output.data[i * 4 + 2] = clamp255((b - bgB * (1 - a)) * invA);
    output.data[i * 4 + 3] = clamp255(a * 255);
  }

  return output;
}

/* ================= UTIL ================= */
function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

function resizeNN(imageData, tw, th) {
  const { width: sw, height: sh, data } = imageData;

  const out = new Uint8ClampedArray(tw * th * 4);

  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const sx = Math.floor(x * sw / tw);
      const sy = Math.floor(y * sh / th);

      const si = (sy * sw + sx) * 4;
      const di = (y * tw + x) * 4;

      out[di] = data[si];
      out[di + 1] = data[si + 1];
      out[di + 2] = data[si + 2];
      out[di + 3] = data[si + 3];
    }
  }

  return out;
}
