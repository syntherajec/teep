/**
 * zip.js — ZIP creation and download utilities
 * Uses JSZip library (loaded globally from libs/)
 */

/**
 * Format bytes to human readable
 */
export function formatBytes(bytes, decimals = 1) {
  if (bytes === 0) return '0 B';
  const k     = 1024;
  const dm    = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i     = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Convert ImageData to a PNG/JPEG Blob
 */
export function imageDataToBlob(imageData, hasAlpha, quality = 0.95) {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    canvas.width  = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);

    const mime = hasAlpha ? 'image/png' : 'image/jpeg';
    canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('toBlob failed')),
      mime,
      quality
    );
  });
}

/**
 * Create a ZIP containing all processed results
 * @param {Array} results - [{filename, imageData, hasAlpha}]
 * @param {function} onStep
 * @returns {Promise<Blob>}
 */
export async function createZip(results, onStep) {
  if (typeof JSZip === 'undefined') {
    throw new Error('JSZip library not loaded');
  }

  const zip = new JSZip();
  const folder = zip.folder('pixelforge_output');

  for (let i = 0; i < results.length; i++) {
    const { filename, imageData, hasAlpha } = results[i];
    onStep?.(`Encoding ${i + 1}/${results.length}: ${filename}`);

    try {
      const blob = await imageDataToBlob(imageData, hasAlpha);
      const buf  = await blob.arrayBuffer();
      folder.file(filename, buf);
    } catch (e) {
      console.warn(`[zip] Failed to encode ${filename}:`, e);
    }
  }

  onStep?.('Generating ZIP archive…');
  return await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  });
}

/**
 * Trigger a browser download for a Blob
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, 1000);
}
