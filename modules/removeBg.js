/**
 * removeBg.js — Background Removal Module
 * Uses U2Net ONNX if available, otherwise smart edge-detection fallback
 *
 * removeBackground() returns: { imageData: ImageData, usedFallback: boolean }
 */

let u2netSession = null;
let modelLoaded  = false;
let modelTried   = false;

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

/**
 * Remove background from ImageData
 * @returns {{ imageData: ImageData, usedFallback: boolean }}
 */
export async function removeBackground(imageData, onStep) {
  const loaded = await loadU2Net(onStep);

  if (loaded && u2netSession) {
    try {
      const result = await removeBackgroundONNX(imageData, onStep);
      return { imageData: result, usedFallback: false };
    } catch (e) {
      console.warn('[removeBg] ONNX inference failed, using fallback:', e);
      onStep?.('Using smart edge-detection fallback…');
      return { imageData: removeBackgroundFallback(imageData), usedFallback: true };
    }
  } else {
    onStep?.('Using smart edge-detection fallback…');
    return { imageData: removeBackgroundFallback(imageData), usedFallback: true };
  }
}

async function removeBackgroundONNX(imageData, onStep) {
  const { width, height } = imageData;
  const size = 320;

  onStep?.('Preprocessing for U2Net…');
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = width; tmpCanvas.height = height;
  tmpCanvas.getContext('2d').putImageData(imageData, 0, 0);
  ctx.drawImage(tmpCanvas, 0, 0, size, size);
  const resized = ctx.getImageData(0, 0, size, size);

  // Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];
  const tensor = new Float32Array(1 * 3 * size * size);
  for (let i = 0; i < size * size; i++) {
    tensor[i]               = ((resized.data[i*4]   / 255) - mean[0]) / std[0];
    tensor[i + size*size]   = ((resized.data[i*4+1] / 255) - mean[1]) / std[1];
    tensor[i + size*size*2] = ((resized.data[i*4+2] / 255) - mean[2]) / std[2];
  }

  onStep?.('Running U2Net inference…');
  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, size, size]);
  const feeds = {};
  feeds[u2netSession.inputNames[0]] = inputTensor;
  const results = await u2netSession.run(feeds);
  const output  = results[u2netSession.outputNames[0]].data;

  onStep?.('Applying mask…');
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = size; maskCanvas.height = size;
  const maskCtx = maskCanvas.getContext('2d');
  const maskData = maskCtx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.min(255, Math.max(0, output[i] * 255));
    maskData.data[i*4]   = v;
    maskData.data[i*4+1] = v;
    maskData.data[i*4+2] = v;
    maskData.data[i*4+3] = 255;
  }
  maskCtx.putImageData(maskData, 0, 0);

  const outCanvas = document.createElement('canvas');
  outCanvas.width = width; outCanvas.height = height;
  const outCtx = outCanvas.getContext('2d');
  outCtx.putImageData(imageData, 0, 0);

  const scaledMask = document.createElement('canvas');
  scaledMask.width = width; scaledMask.height = height;
  scaledMask.getContext('2d').drawImage(maskCanvas, 0, 0, width, height);
  const scaledMaskData = scaledMask.getContext('2d').getImageData(0, 0, width, height);

  const outData = outCtx.getImageData(0, 0, width, height);
  for (let i = 0; i < width * height; i++) {
    outData.data[i*4+3] = scaledMaskData.data[i*4];
  }
  outCtx.putImageData(outData, 0, 0);
  return outCtx.getImageData(0, 0, width, height);
}

/**
 * Smart fallback: GrabCut-style color clustering + edge refinement
 */
function removeBackgroundFallback(imageData) {
  const { width, height, data } = imageData;
  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  const bgColors = [];
  const samplePoints = [
    [0,0],[width-1,0],[0,height-1],[width-1,height-1],
    [Math.floor(width/2),0],[Math.floor(width/2),height-1],
    [0,Math.floor(height/2)],[width-1,Math.floor(height/2)],
  ];
  for (const [x,y] of samplePoints) {
    const i = (y*width + x)*4;
    bgColors.push([data[i], data[i+1], data[i+2]]);
  }

  const bgR = bgColors.reduce((s,c)=>s+c[0],0)/bgColors.length;
  const bgG = bgColors.reduce((s,c)=>s+c[1],0)/bgColors.length;
  const bgB = bgColors.reduce((s,c)=>s+c[2],0)/bgColors.length;

  const bgVar = bgColors.reduce((s,c)=>{
    return s + Math.abs(c[0]-bgR) + Math.abs(c[1]-bgG) + Math.abs(c[2]-bgB);
  },0) / bgColors.length;

  const threshold = Math.max(30, Math.min(80, bgVar * 2 + 20));

  const alpha = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
    const dist = Math.sqrt(
      (r - bgR)**2 * 0.299 +
      (g - bgG)**2 * 0.587 +
      (b - bgB)**2 * 0.114
    );
    alpha[i] = Math.min(1, dist / threshold);
  }

  const blurred = gaussianBlurAlpha(alpha, width, height, 2);

  for (let i = 0; i < width * height; i++) {
    const a = Math.min(1, Math.max(0, blurred[i]));
    const sharpened = a < 0.3 ? a * 0.3 : a > 0.7 ? 0.7 + (a - 0.7) * 1.5 : a;
    output.data[i*4+3] = Math.min(255, Math.max(0, sharpened * 255));
  }

  return output;
}

function gaussianBlurAlpha(alpha, w, h, radius) {
  const out = new Float32Array(alpha.length);
  const kernel = buildGaussianKernel(radius);
  const klen = kernel.length;
  const khalf = Math.floor(klen / 2);

  const tmp = new Float32Array(alpha.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, wsum = 0;
      for (let k = 0; k < klen; k++) {
        const nx = x + k - khalf;
        if (nx >= 0 && nx < w) {
          sum  += alpha[y*w+nx] * kernel[k];
          wsum += kernel[k];
        }
      }
      tmp[y*w+x] = sum / wsum;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, wsum = 0;
      for (let k = 0; k < klen; k++) {
        const ny = y + k - khalf;
        if (ny >= 0 && ny < h) {
          sum  += tmp[ny*w+x] * kernel[k];
          wsum += kernel[k];
        }
      }
      out[y*w+x] = sum / wsum;
    }
  }
  return out;
}

function buildGaussianKernel(radius) {
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  const sigma = radius / 2;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x*x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return kernel;
}
