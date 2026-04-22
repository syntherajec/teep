/**
 * resize.js — Resize / Crop Module
 * Handles aspect ratio cropping and dimension constraints
 * Optimized for TeePublic minimum 4500px requirements
 */

const RATIO_MAP = {
  '1:1':  [1, 1],
  '3:4':  [3, 4],
  '4:3':  [4, 3],
  '4:5':  [4, 5],
  '9:16': [9, 16],
  '16:9': [16, 9],
};

/**
 * Resize and/or crop an ImageData
 * @param {ImageData} imageData
 * @param {string} ratio - 'original', '1:1', '16:9', etc.
 * @param {number} maxW - Max width (0 = no limit)
 * @param {function} onStep
 * @returns {ImageData}
 */
export function resizeImage(imageData, ratio, maxW, onStep) {
  let { width, height } = imageData;

  // Step 1: Crop to aspect ratio (center crop)
  let cropX = 0, cropY = 0;
  let cropW = width, cropH = height;

  if (ratio !== 'original' && RATIO_MAP[ratio]) {
    const [rw, rh] = RATIO_MAP[ratio];
    const targetAspect = rw / rh;
    const srcAspect    = width / height;

    if (srcAspect > targetAspect) {
      // Wider than target: crop sides
      cropH = height;
      cropW = Math.round(height * targetAspect);
      cropX = Math.round((width - cropW) / 2);
      cropY = 0;
    } else {
      // Taller than target: crop top/bottom
      cropW = width;
      cropH = Math.round(width / targetAspect);
      cropX = 0;
      cropY = Math.round((height - cropH) / 2);
    }
    onStep?.(`Cropping to ${ratio} (${cropW}×${cropH})…`);
  }

  // Step 2: Apply maxW constraint
  let outW = cropW, outH = cropH;
  if (maxW > 0 && outW > maxW) {
    const factor = maxW / outW;
    outW = maxW;
    outH = Math.round(outH * factor);
    onStep?.(`Scaling to max ${maxW}px wide…`);
  }

  // If no changes needed
  if (cropX === 0 && cropY === 0 && cropW === width && cropH === height && outW === width && outH === height) {
    return imageData;
  }

  // Step 3: Render onto canvas
  const srcCanvas = document.createElement('canvas');
  srcCanvas.width  = width;
  srcCanvas.height = height;
  srcCanvas.getContext('2d').putImageData(imageData, 0, 0);

  const outCanvas = document.createElement('canvas');
  outCanvas.width  = outW;
  outCanvas.height = outH;
  const ctx = outCanvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  // Draw cropped+resized
  ctx.drawImage(srcCanvas, cropX, cropY, cropW, cropH, 0, 0, outW, outH);

  onStep?.(`Resized → ${outW}×${outH}`);
  return ctx.getImageData(0, 0, outW, outH);
}

/**
 * Get output dimensions without actually processing
 */
export function getOutputDimensions(width, height, ratio, maxW) {
  let cropW = width, cropH = height;

  if (ratio !== 'original' && RATIO_MAP[ratio]) {
    const [rw, rh] = RATIO_MAP[ratio];
    const targetAspect = rw / rh;
    const srcAspect    = width / height;
    if (srcAspect > targetAspect) {
      cropH = height;
      cropW = Math.round(height * targetAspect);
    } else {
      cropW = width;
      cropH = Math.round(width / targetAspect);
    }
  }

  let outW = cropW, outH = cropH;
  if (maxW > 0 && outW > maxW) {
    const factor = maxW / outW;
    outW = maxW;
    outH = Math.round(outH * factor);
  }
  return { width: outW, height: outH };
}
