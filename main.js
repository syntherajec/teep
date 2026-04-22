/**
 * main.js — PixelForge
 * Orchestrates upload UI, queue, and full processing pipeline.
 * ES module; all processing is 100% client-side.
 */

import { removeBackground } from './modules/removeBg.js';
import { upscaleImage }    from './modules/upscale.js';
import { resizeImage }                                     from './modules/resize.js';
import { createZip, downloadBlob, formatBytes }           from './modules/zip.js';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  files:      [],   // File[]
  results:    [],   // {filename, imageData, hasAlpha}[]
  processing: false,
  zipBlob:    null,
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone       = $('dropZone');
const fileInput      = $('fileInput');
const fileListEl     = $('fileList');
const fileItemsEl    = $('fileItems');
const fileCountEl    = $('fileCount');
const clearFilesBtn  = $('clearFiles');

const removeBgCheck  = $('removeBgCheck');
const removeBgNote   = $('removeBgNote');
const upscaleCheck   = $('upscaleCheck');
const upscaleNote    = $('upscaleNote');
const upscaleSelect  = $('upscaleSelect');
const upscaleModel   = $('upscaleModel');
const resizeCheck    = $('resizeCheck');
const ratioSelect    = $('ratioSelect');
const maxWidthInput  = $('maxWidth');

const startBtn       = $('startBtn');
const downloadBtn    = $('downloadBtn');

const progressWrap   = $('progressWrap');
const progressBarEl  = $('progressBar');
const progressLabel  = $('progressLabel');
const progressCount  = $('progressCount');
const progressSub    = $('progressSub');

const queueWrap      = $('queueWrap');
const queueListEl    = $('queueList');

// ─── Drag & Drop / File Input ─────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles([...e.dataTransfer.files]);
});

fileInput.addEventListener('change', () => {
  addFiles([...fileInput.files]);
  fileInput.value = '';
});

clearFilesBtn.addEventListener('click', () => {
  state.files = [];
  renderFileList();
  updateStartBtn();
  resetProcessingUI();
});

// ─── File Management ──────────────────────────────────────────────────────────
function addFiles(newFiles) {
  const images = newFiles.filter(f => f.type.startsWith('image/'));
  if (!images.length) { showToast('No image files detected', 'error'); return; }

  const existing = new Set(state.files.map(f => f.name + f.size));
  const added    = images.filter(f => !existing.has(f.name + f.size));

  state.files.push(...added);
  renderFileList();
  updateStartBtn();

  if (added.length < images.length) {
    showToast(`${images.length - added.length} duplicate(s) skipped`, 'info');
  } else {
    showToast(`${added.length} image(s) added`, 'success');
  }
}

function renderFileList() {
  if (state.files.length === 0) {
    fileListEl.style.display = 'none';
    return;
  }
  fileListEl.style.display = 'block';
  fileCountEl.textContent  = `${state.files.length} file${state.files.length !== 1 ? 's' : ''} selected`;

  fileItemsEl.innerHTML = '';
  state.files.forEach((file, idx) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.id = `file-item-${idx}`;

    const objUrl = URL.createObjectURL(file);
    item.innerHTML = `
      <img class="file-thumb" src="${objUrl}" alt="" />
      <div class="file-info">
        <div class="file-name" title="${escHtml(file.name)}">${escHtml(file.name)}</div>
        <div class="file-size">${formatBytes(file.size)}</div>
      </div>
      <span class="file-status waiting" id="fstatus-${idx}">Ready</span>
    `;
    // Revoke object URL after image loads to free memory
    item.querySelector('img').addEventListener('load', () => URL.revokeObjectURL(objUrl), { once: true });
    fileItemsEl.appendChild(item);
  });
}

function setFileStatus(idx, cls, label) {
  const el = $(`fstatus-${idx}`);
  if (el) { el.className = `file-status ${cls}`; el.textContent = label; }
}

// ─── Option Wiring ────────────────────────────────────────────────────────────
upscaleSelect.addEventListener('change', () => {
  const v = parseInt(upscaleSelect.value);
  if (v === 2) upscaleModel.value = 'x2';
  else if (v >= 4 && upscaleModel.value === 'x2') upscaleModel.value = 'x4';
});

// ─── Start / Download Buttons ─────────────────────────────────────────────────
function updateStartBtn() {
  startBtn.disabled = state.files.length === 0 || state.processing;
}

startBtn.addEventListener('click', () => {
  if (!state.processing && state.files.length > 0) runProcessing();
});

downloadBtn.addEventListener('click', () => {
  if (state.zipBlob) downloadBlob(state.zipBlob, `pixelforge_${Date.now()}.zip`);
});

// ─── Main Pipeline ────────────────────────────────────────────────────────────
async function runProcessing() {
  state.processing = true;
  state.results    = [];
  state.zipBlob    = null;

  startBtn.disabled    = true;
  downloadBtn.disabled = true;

  progressWrap.style.display = 'block';
  queueWrap.style.display    = 'block';

  // Snapshot options
  const opts = {
    removeBg:    removeBgCheck.checked,
    upscale:     upscaleCheck.checked,
    upscaleVal:  parseInt(upscaleSelect.value),
    upscaleKey:  upscaleModel.value,
    resize:      resizeCheck.checked,
    ratio:       ratioSelect.value,
    maxW:        parseInt(maxWidthInput.value) || 0,
  };

  buildQueueUI(state.files);
  setProgress(0, state.files.length, 'Starting queue…');
  progressBarEl.style.width = '0%';

  let errorCount = 0;

  for (let i = 0; i < state.files.length; i++) {
    const file = state.files[i];

    setQueueState(i, 'active', 'Loading…');
    setFileStatus(i, 'active', 'Processing');
    setProgress(i, state.files.length, truncate(file.name, 40));

    try {
      const result = await processFile(file, opts, msg => {
        setQueueStep(i, msg);
        setProgressSub(msg);
      });
      state.results.push(result);
      setQueueState(i, 'done', '✓ Done');
      setFileStatus(i, 'done', 'Done');
    } catch (err) {
      console.error(`[pipeline] ${file.name}:`, err);
      setQueueState(i, 'error', '✗ Error');
      setFileStatus(i, 'error', 'Error');
      errorCount++;
    }

    const pct = Math.round(((i + 1) / state.files.length) * 100);
    progressBarEl.style.width = pct + '%';
    progressCount.textContent = `${i + 1} / ${state.files.length}`;
  }

  // ── ZIP ───────────────────────────────────────────────────────────────────
  if (state.results.length > 0) {
    setProgressSub('Packaging ZIP…');
    try {
      state.zipBlob = await createZip(state.results, setProgressSub);
      downloadBtn.disabled = false;
      progressLabel.textContent = `✓ Complete — ${state.results.length} image(s) ready`;
      progressBarEl.style.width = '100%';

      const msg = errorCount > 0
        ? `${state.results.length} done, ${errorCount} failed — ZIP ready`
        : `All ${state.results.length} image(s) processed — ZIP ready!`;
      showToast(msg, 'success');
    } catch (err) {
      console.error('[zip]', err);
      showToast('ZIP creation failed: ' + err.message, 'error');
    }
  } else {
    showToast('No images were processed successfully', 'error');
    progressLabel.textContent = 'Processing failed';
  }

  state.processing = false;
  startBtn.disabled = false;
}

// ─── Per-file Pipeline ────────────────────────────────────────────────────────
async function processFile(file, opts, onStep) {
  // ① Load → ImageData
  onStep('Loading image…');
  let imageData = await fileToImageData(file);
  let hasAlpha  = false;

  // ② Remove background
  if (opts.removeBg) {
    onStep('Removing background…');
    imageData = await removeBackground(imageData, onStep);
    hasAlpha  = true;
    // Show fallback note if model not available
    if (removeBgNote) removeBgNote.style.display = 'block';
  }

  // ③ AI Upscale
  if (opts.upscale && opts.upscaleVal > 1) {
    onStep(`Upscaling ${opts.upscaleVal}x…`);
    imageData = await upscaleImage(imageData, opts.upscaleVal, opts.upscaleKey, onStep);
    // Show fallback note if model not available
    if (upscaleNote) upscaleNote.style.display = 'block';
  }

  // ④ Resize / Aspect Ratio
  if (opts.resize) {
    onStep('Resizing…');
    imageData = resizeImage(imageData, opts.ratio, opts.maxW, onStep);
  }

  // Build output filename
  const ext      = hasAlpha ? '.png' : getExt(file.name);
  const base     = file.name.replace(/\.[^/.]+$/, '');
  const suffix   = buildSuffix(opts);
  const filename = `${base}${suffix}${ext}`;

  return { filename, imageData, hasAlpha };
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function fileToImageData(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      try {
        const c = document.createElement('canvas');
        c.width  = img.naturalWidth;
        c.height = img.naturalHeight;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        resolve(ctx.getImageData(0, 0, c.width, c.height));
      } catch(e) { URL.revokeObjectURL(url); reject(e); }
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Image load failed')); };
    img.src = url;
  });
}

function buildSuffix(opts) {
  const p = [];
  if (opts.removeBg)                       p.push('nobg');
  if (opts.upscale && opts.upscaleVal > 1) p.push(`${opts.upscaleVal}x`);
  if (opts.resize && opts.ratio !== 'original') p.push(opts.ratio.replace(':', 'x'));
  return p.length ? '_' + p.join('_') : '_processed';
}

function getExt(filename) {
  const m = filename.match(/\.[^/.]+$/);
  return m ? m[0].toLowerCase() : '.png';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// ─── Queue UI ─────────────────────────────────────────────────────────────────
function buildQueueUI(files) {
  queueListEl.innerHTML = '';
  files.forEach((file, i) => {
    const el = document.createElement('div');
    el.className = 'queue-item';
    el.id = `qitem-${i}`;
    el.innerHTML = `
      <span class="q-num">${String(i + 1).padStart(2, '0')}</span>
      <span class="q-name">${escHtml(file.name)}</span>
      <span class="q-step" id="qstep-${i}">Waiting</span>
      <span class="q-icon" id="qicon-${i}">·</span>
    `;
    queueListEl.appendChild(el);
  });
}

function setQueueState(idx, cls, label) {
  const item = $(`qitem-${idx}`);
  const icon = $(`qicon-${idx}`);
  const step = $(`qstep-${idx}`);
  if (!item) return;
  item.className = `queue-item q-${cls}`;
  if (cls === 'active') {
    icon.innerHTML = '<span class="spinner"></span>';
    step.textContent = label;
    item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {
    icon.textContent = cls === 'done' ? '✓' : '✗';
    step.textContent = label;
  }
}

function setQueueStep(idx, msg) {
  const el = $(`qstep-${idx}`);
  if (el) el.textContent = msg.length > 32 ? msg.slice(0, 32) + '…' : msg;
}

// ─── Progress UI ──────────────────────────────────────────────────────────────
function setProgress(current, total, label) {
  progressLabel.textContent = label;
  progressCount.textContent = `${current} / ${total}`;
}

function setProgressSub(msg) {
  if (progressSub) progressSub.textContent = msg;
}

function resetProcessingUI() {
  progressWrap.style.display = 'none';
  queueWrap.style.display    = 'none';
  progressBarEl.style.width  = '0%';
  downloadBtn.disabled       = true;
}

// ─── Toast ────────────────────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(() => {
    t.style.transition = 'opacity 0.5s ease';
    t.style.opacity    = '0';
    setTimeout(() => t.remove(), 600);
  }, 4000);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
updateStartBtn();
