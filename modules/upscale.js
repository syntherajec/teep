/**
 * upscale.js — AI Upscale Module
 * Uses Real-ESRGAN ONNX if available, otherwise high-quality bicubic fallback
 *
 * upscaleImage() returns: { imageData: ImageData, usedFallback: boolean }
 */

const sessions = {};
const tried    = {};

const MODEL_PATHS = {
  x2:    './models/realesrgan-x2.onnx',
  x4:    './models/realesrgan-x4.onnx',
  anime: './models/realesrgan-anime-x4.onnx',
};

async function loadModel(key, onStep) {
  if (sessions[key]) return sessions[key];
  if (tried[key])    return null;
  tried[key] = true;
  try {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime not available');
    onStep?.(`Loading ${key} model…`);
    sessions[key] = await ort.InferenceSession.create(MODEL_PATHS[key]);
    onStep?.(`${key} model loaded!`);
    return sessions[key];
  } catch (e) {
    console.warn(`[upscale] ${key} model not available:`, e.message);
    return null;
  }
}

/**
 * Upscale an ImageData by the given scale factor
 * @param {ImageData} imageData
 * @param {number} scale - 2, 4, or 8
 * @param {string} modelKey - 'x2', 'x4', 'anime'
 * @param {function} onStep
 * @returns {Promise<{ imageData: ImageData, usedFallback: boolean }>}
 */
export async function upscaleImage(imageData, scale, modelKey, onStep) {
  const ortKey  = scale === 2 ? 'x2' : (modelKey === 'anime' ? 'anime' : 'x4');
  const session = await loadModel(ortKey, onStep);

  if (session) {
    try {
      const result = await upscaleONNX(imageData, scale, session, onStep);
      return { imageData: result, usedFallback: false };
    } catch (e) {
      console.warn('[upscale] ONNX inference failed, using bicubic:', e);
      onStep?.(`Using bicubic upscale ${scale}x (fallback)…`);
      return { imageData: upscaleBicubic(imageData, scale, onStep), usedFallback: true };
    }
  } else {
    onStep?.(`Using bicubic upscale ${scale}x (fallback)…`);
    return { imageData: upscaleBicubic(imageData, scale, onStep), usedFallback: true };
  }
}

async function upscaleONNX(imageData, scale, session, onStep) {
  const { width, height } = imageData;
  const TILE = 256;
  const modelScale = 4;

  onStep?.('Preparing tiles…');
  const outW = width  * modelScale;
  const outH = height * modelScale;
  const outCanvas = document.createElement('canvas');
  outCanvas.width = outW; outCanvas.height = outH;
  const outCtx = outCanvas.getContext('2d');

  // Preserve alpha channel separately
  const sourceHasAlpha = hasTransparency(imageData);
  let alphaPixels = null;
  if (sourceHasAlpha) {
    const srcAlphaCanvas = document.createElement('canvas');
    srcAlphaCanvas.width = width; srcAlphaCanvas.height = height;
    const srcAlphaCtx = srcAlphaCanvas.getContext('2d');
    const alphaGray = srcAlphaCtx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const a = imageData.data[i * 4 + 3];
      alphaGray.data[i * 4]     = a;
      alphaGray.data[i * 4 + 1] = a;
      alphaGray.data[i * 4 + 2] = a;
      alphaGray.data[i * 4 + 3] = 255;
    }
    srcAlphaCtx.putImageData(alphaGray, 0, 0);
    const scaledAlphaCanvas = document.createElement('canvas');
    scaledAlphaCanvas.width = outW; scaledAlphaCanvas.height = outH;
    const scaledAlphaCtx = scaledAlphaCanvas.getContext('2d');
    scaledAlphaCtx.imageSmoothingEnabled = true;
    scaledAlphaCtx.imageSmoothingQuality = 'high';
    scaledAlphaCtx.drawImage(srcAlphaCanvas, 0, 0, outW, outH);
    alphaPixels = scaledAlphaCtx.getImageData(0, 0, outW, outH);
  }

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width; srcCanvas.height = height;
  srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

  const tilesX = Math.ceil(width  / TILE);
  const tilesY = Math.ceil(height / TILE);
  const total  = tilesX * tilesY;
  let   done   = 0;

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const sx = tx * TILE, sy = ty * TILE;
      const tw = Math.min(TILE, width  - sx);
      const th = Math.min(TILE, height - sy);

      const tileCanvas = document.createElement('canvas');
      tileCanvas.width = tw; tileCanvas.height = th;
      tileCanvas.getContext('2d').drawImage(srcCanvas, sx, sy, tw, th, 0, 0, tw, th);
      const tileData = tileCanvas.getContext('2d').getImageData(0, 0, tw, th);

      const tensor = new Float32Array(3 * th * tw);
      for (let i = 0; i < tw * th; i++) {
        tensor[i]               = tileData.data[i * 4]     / 255;
        tensor[i + tw * th]     = tileData.data[i * 4 + 1] / 255;
        tensor[i + tw * th * 2] = tileData.data[i * 4 + 2] / 255;
      }

      const input  = new ort.Tensor('float32', tensor, [1, 3, th, tw]);
      const feeds  = {};
      feeds[session.inputNames[0]] = input;
      const result = await session.run(feeds);
      const out    = result[session.outputNames[0]].data;

      const ow = tw * modelScale, oh = th * modelScale;
      const tOut = document.createElement('canvas');
      tOut.width = ow; tOut.height = oh;
      const tOutCtx = tOut.getContext('2d');
      const outPx   = tOutCtx.createImageData(ow, oh);
      for (let i = 0; i < ow * oh; i++) {
        outPx.data[i * 4]     = Math.min(255, Math.max(0, out[i]               * 255));
        outPx.data[i * 4 + 1] = Math.min(255, Math.max(0, out[i + ow * oh]     * 255));
        outPx.data[i * 4 + 2] = Math.min(255, Math.max(0, out[i + ow * oh * 2] * 255));
        outPx.data[i * 4 + 3] = 255;
      }
      tOutCtx.putImageData(outPx, 0, 0);
      outCtx.drawImage(tOut, sx * modelScale, sy * modelScale);

      done++;
      onStep?.(`Upscaling tile ${done}/${total}…`);
    }
  }

  // Re-apply upscaled alpha if source had transparency
  if (sourceHasAlpha && alphaPixels) {
    onStep?.('Restoring alpha channel…');
    const composited = outCtx.getImageData(0, 0, outW, outH);
    for (let i = 0; i < outW * outH; i++) {
      composited.data[i * 4 + 3] = alphaPixels.data[i * 4];
    }
    outCtx.putImageData(composited, 0, 0);
  }

  if (scale !== modelScale) {
    const finalW = width  * scale;
    const finalH = height * scale;
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = finalW; finalCanvas.height = finalH;
    const finalCtx = finalCanvas.getContext('2d');
    finalCtx.imageSmoothingEnabled = true;
    finalCtx.imageSmoothingQuality = 'high';
    finalCtx.drawImage(outCanvas, 0, 0, finalW, finalH);
    return finalCtx.getImageData(0, 0, finalW, finalH);
  }

  return outCtx.getImageData(0, 0, outW, outH);
}

/**
 * Check if ImageData contains any non-opaque pixels
 */
function hasTransparency(imageData) {
  const data = imageData.data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) return true;
  }
  return false;
}

/**
 * High-quality bicubic upscale fallback — preserves alpha
 */
function upscaleBicubic(imageData, scale, onStep) {
  const { width, height } = imageData;
  const outW = width  * scale;
  const outH = height * scale;

  onStep?.(`Bicubic ${scale}x: ${width}×${height} → ${outW}×${outH}…`);

  const srcCanvas = document.createElement('canvas');
  srcCanvas.width = width; srcCanvas.height = height;
  const srcCtx = srcCanvas.getContext('2d');
  srcCtx.clearRect(0, 0, width, height);
  srcCtx.putImageData(imageData, 0, 0);

  const outCanvas = document.createElement('canvas');
  outCanvas.width = outW; outCanvas.height = outH;
  const outCtx = outCanvas.getContext('2d');
  outCtx.imageSmoothingEnabled = true;
  outCtx.imageSmoothingQuality = 'high';
  outCtx.clearRect(0, 0, outW, outH);
  outCtx.drawImage(srcCanvas, 0, 0, outW, outH);

  if (scale >= 2) {
    onStep?.('Sharpening…');
    const sharpened = sharpenImageData(outCtx.getImageData(0, 0, outW, outH));
    outCtx.putImageData(sharpened, 0, 0);
  }

  return outCtx.getImageData(0, 0, outW, outH);
}

/**
 * Unsharp mask sharpening — preserves alpha channel
 */
function sharpenImageData(imageData) {
  const { width, height, data } = imageData;
  const out = new ImageData(new Uint8ClampedArray(data), width, height);
  const kernel = [
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0,
  ];
  const kSize = 3;
  const kHalf = 1;

  for (let y = kHalf; y < height - kHalf; y++) {
    for (let x = kHalf; x < width - kHalf; x++) {
      let r = 0, g = 0, b = 0;
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          const px = (y + ky - kHalf) * width + (x + kx - kHalf);
          const k  = kernel[ky * kSize + kx];
          r += data[px * 4]     * k;
          g += data[px * 4 + 1] * k;
          b += data[px * 4 + 2] * k;
        }
      }
      const i = (y * width + x) * 4;
      out.data[i]     = Math.min(255, Math.max(0, r));
      out.data[i + 1] = Math.min(255, Math.max(0, g));
      out.data[i + 2] = Math.min(255, Math.max(0, b));
      // Alpha preserved from copied data
    }
  }
  return out;
}
