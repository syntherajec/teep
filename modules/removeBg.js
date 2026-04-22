/**
 * removeBg.js — Background Removal Module (v3 — Sharp & Clean)
 *
 * Key improvements:
 *  • Detects existing alpha channel → skips bg removal, only cleans fringe
 *  • Guided filter radius reduced to 2 (was 4) → no interior blurring
 *  • Conservative erosion: only touches pixels already <0.4 alpha
 *  • De-fringe only in transition band, never on solid pixels
 *
 * Returns: { imageData: ImageData, usedFallback: boolean }
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
    modelLoaded  = true;
    onStep?.('U2Net model loaded!');
    return true;
  } catch (e) {
    console.warn('[removeBg] U2Net not available, using fallback:', e.message);
    return false;
  }
}

export async function removeBackground(imageData, onStep) {
  if (hasExistingAlpha(imageData)) {
    onStep?.('Alpha channel detected — cleaning fringe only…');
    return { imageData: cleanExistingAlpha(imageData), usedFallback: false };
  }

  const loaded = await loadU2Net(onStep);
  let rawAlpha, usedFallback;

  if (loaded && u2netSession) {
    try {
      rawAlpha     = await inferU2Net(imageData, onStep);
      usedFallback = false;
    } catch (e) {
      console.warn('[removeBg] ONNX failed, fallback:', e);
      onStep?.('Using smart edge-detection fallback…');
      rawAlpha     = computeFallbackAlpha(imageData);
      usedFallback = true;
    }
  } else {
    onStep?.('Using smart edge-detection fallback…');
    rawAlpha     = computeFallbackAlpha(imageData);
    usedFallback = true;
  }

  onStep?.('Refining edges & removing fringe…');
  return { imageData: postProcessAlpha(rawAlpha, imageData), usedFallback };
}

/* ── Existing-alpha detection ─────────────────────────────────────────── */
function hasExistingAlpha(imageData) {
  const { data } = imageData;
  const total = data.length / 4;
  const step  = Math.max(1, Math.floor(total / 40000));
  let semi = 0, checked = 0;
  for (let i = 0; i < total; i += step) {
    if (data[i * 4 + 3] < 250) semi++;
    checked++;
  }
  return (semi / checked) > 0.005;
}

/* ── Clean existing alpha: de-fringe only, preserve solid pixels ──────── */
function cleanExistingAlpha(imageData) {
  const { width, height, data } = imageData;
  const N      = width * height;
  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  // Estimate background colour from nearly-transparent pixels
  let bgR = 0, bgG = 0, bgB = 0, bgCnt = 0;
  for (let i = 0; i < N; i++) {
    const a = data[i * 4 + 3];
    if (a > 5 && a < 60) {
      bgR += data[i*4]; bgG += data[i*4+1]; bgB += data[i*4+2]; bgCnt++;
    }
  }
  if (bgCnt < 10) { bgR = 255; bgG = 255; bgB = 255; bgCnt = 1; }
  bgR /= bgCnt; bgG /= bgCnt; bgB /= bgCnt;

  for (let i = 0; i < N; i++) {
    const origA = data[i * 4 + 3];
    if (origA >= 252) { output.data[i*4+3] = 255; continue; }
    if (origA <=   3) { output.data[i*4+3] = 0;   continue; }

    const a       = origA / 255;
    const sharpA  = sCurve(a, 0.10, 0.90);
    if (sharpA <= 0) { output.data[i*4+3] = 0;   continue; }
    if (sharpA >= 1) { output.data[i*4+3] = 255; continue; }

    const r0 = data[i*4], g0 = data[i*4+1], b0 = data[i*4+2];
    const inv = 1 / sharpA;
    output.data[i*4]   = clamp255((r0 - bgR*(1-sharpA)) * inv);
    output.data[i*4+1] = clamp255((g0 - bgG*(1-sharpA)) * inv);
    output.data[i*4+2] = clamp255((b0 - bgB*(1-sharpA)) * inv);
    output.data[i*4+3] = clamp255(sharpA * 255);
  }
  return output;
}

/* ── U²Net ONNX inference → Float32Array alpha [0..1] ────────────────── */
async function inferU2Net(imageData, onStep) {
  const { width, height } = imageData;
  const SIZE = 320;
  onStep?.('Preprocessing for U2Net…');
  const resized = resizeNN(imageData, SIZE, SIZE);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const tensor = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    tensor[i]               = ((resized[i*4]   / 255) - mean[0]) / std[0];
    tensor[i + SIZE*SIZE]   = ((resized[i*4+1] / 255) - mean[1]) / std[1];
    tensor[i + SIZE*SIZE*2] = ((resized[i*4+2] / 255) - mean[2]) / std[2];
  }
  onStep?.('Running U2Net inference…');
  const inp = new ort.Tensor('float32', tensor, [1, 3, SIZE, SIZE]);
  const feeds = {}; feeds[u2netSession.inputNames[0]] = inp;
  const res = await u2netSession.run(feeds);
  const out = res[u2netSession.outputNames[0]].data;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < out.length; i++) { if (out[i]<mn) mn=out[i]; if (out[i]>mx) mx=out[i]; }
  const range = mx - mn || 1;
  const alpha = new Float32Array(width * height);
  const sx = SIZE/width, sy = SIZE/height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = x*sx, py = y*sy;
      const x0 = Math.min(Math.floor(px), SIZE-2), y0 = Math.min(Math.floor(py), SIZE-2);
      const fx = px-x0, fy = py-y0;
      const v = out[y0*SIZE+x0]*(1-fx)*(1-fy) + out[y0*SIZE+x0+1]*fx*(1-fy)
              + out[(y0+1)*SIZE+x0]*(1-fx)*fy  + out[(y0+1)*SIZE+x0+1]*fx*fy;
      alpha[y*width+x] = (v-mn)/range;
    }
  }
  return alpha;
}

/* ── Fallback: border-sample background model ─────────────────────────── */
function computeFallbackAlpha(imageData) {
  const { width, height, data } = imageData;
  const N    = width * height;
  const BAND = Math.max(3, Math.round(Math.min(width, height) * 0.04));
  const samp = [];
  for (let x = 0; x < width; x++) {
    for (let b = 0; b < BAND; b++) {
      push3(samp, data, (b*width+x)*4);
      push3(samp, data, ((height-1-b)*width+x)*4);
    }
  }
  for (let y = BAND; y < height-BAND; y++) {
    for (let b = 0; b < BAND; b++) {
      push3(samp, data, (y*width+b)*4);
      push3(samp, data, (y*width+(width-1-b))*4);
    }
  }
  const bgR = mean3(samp,0), bgG = mean3(samp,1), bgB = mean3(samp,2);
  const noise = Math.sqrt((var3(samp,0,bgR)+var3(samp,1,bgG)+var3(samp,2,bgB))/3);
  const THRESH = Math.max(30, Math.min(85, 32 + noise*1.4));
  const alpha = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const dist = Math.sqrt((data[i*4]-bgR)**2*0.2126+(data[i*4+1]-bgG)**2*0.7152+(data[i*4+2]-bgB)**2*0.0722);
    alpha[i] = Math.min(1, dist/THRESH);
  }
  const blurred = separableGaussian(alpha, width, height, 2);
  for (let i = 0; i < N; i++) blurred[i] = sCurve(blurred[i], 0.18, 0.82);
  return blurred;
}

/* ── Post-processing: guided filter + de-fringe ───────────────────────── */
function postProcessAlpha(rawAlpha, imageData) {
  const { width, height, data } = imageData;
  const N    = width * height;
  const luma = new Float32Array(N);
  for (let i = 0; i < N; i++)
    luma[i] = (data[i*4]*0.2126 + data[i*4+1]*0.7152 + data[i*4+2]*0.0722) / 255;

  const guided = guidedFilter(rawAlpha, luma, width, height, 2, 0.002);
  const eroded = conservativeErode(guided, width, height);
  const output = new ImageData(new Uint8ClampedArray(data), width, height);

  let bgR=0,bgG=0,bgB=0,bgCnt=0;
  for (let i=0;i<N;i++) if (eroded[i]<0.07) { bgR+=data[i*4];bgG+=data[i*4+1];bgB+=data[i*4+2];bgCnt++; }
  if (bgCnt<10) { bgR=255;bgG=255;bgB=255;bgCnt=1; }
  bgR/=bgCnt; bgG/=bgCnt; bgB/=bgCnt;

  for (let i=0;i<N;i++) {
    const a = sCurve(eroded[i], 0.12, 0.88);
    if (a<=0) { output.data[i*4+3]=0; continue; }
    if (a>=1) { output.data[i*4+3]=255; continue; }
    const r0=data[i*4],g0=data[i*4+1],b0=data[i*4+2], inv=1/a;
    output.data[i*4]   = clamp255((r0-bgR*(1-a))*inv);
    output.data[i*4+1] = clamp255((g0-bgG*(1-a))*inv);
    output.data[i*4+2] = clamp255((b0-bgB*(1-a))*inv);
    output.data[i*4+3] = clamp255(a*255);
  }
  return output;
}

/* ── Shared helpers ───────────────────────────────────────────────────── */
function clamp255(v) { return v<0?0:v>255?255:Math.round(v); }
function push3(a,d,o) { a.push([d[o],d[o+1],d[o+2]]); }
function mean3(s,c) { return s.reduce((a,x)=>a+x[c],0)/(s.length||1); }
function var3(s,c,m) { return s.reduce((a,x)=>a+(x[c]-m)**2,0)/(s.length||1); }

function sCurve(v, lo, hi) {
  if (v<=0) return 0; if (v>=1) return 1;
  if (v<=lo) return (v/lo)**2 * lo * 0.5;
  if (v>=hi) { const t=(v-hi)/(1-hi); return hi+(1-hi)*(1-(1-t)**2); }
  return v;
}

function separableGaussian(alpha, w, h, radius) {
  const kernel=buildGaussKernel(radius), klen=kernel.length, kh=radius;
  const tmp=new Float32Array(alpha.length), out=new Float32Array(alpha.length);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    let s=0,ws=0;
    for (let k=0;k<klen;k++) { const nx=x+k-kh; if(nx>=0&&nx<w){s+=alpha[y*w+nx]*kernel[k];ws+=kernel[k];} }
    tmp[y*w+x]=s/ws;
  }
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    let s=0,ws=0;
    for (let k=0;k<klen;k++) { const ny=y+k-kh; if(ny>=0&&ny<h){s+=tmp[ny*w+x]*kernel[k];ws+=kernel[k];} }
    out[y*w+x]=s/ws;
  }
  return out;
}

function buildGaussKernel(radius) {
  const size=radius*2+1, k=new Float32Array(size), sigma=Math.max(radius/2,0.5);
  let sum=0;
  for (let i=0;i<size;i++) { const x=i-radius; k[i]=Math.exp(-(x*x)/(2*sigma*sigma)); sum+=k[i]; }
  for (let i=0;i<size;i++) k[i]/=sum;
  return k;
}

function guidedFilter(alpha, guide, w, h, r, eps) {
  const N=w*h;
  const boxMean=(arr)=>{
    const tmp=new Float32Array(N), out=new Float32Array(N);
    for (let y=0;y<h;y++) {
      let s=0,c=0;
      for (let x=0;x<Math.min(r+1,w);x++){s+=arr[y*w+x];c++;}
      for (let x=0;x<w;x++){
        if(x+r<w){s+=arr[y*w+x+r];c++;} if(x-r-1>=0){s-=arr[y*w+x-r-1];c--;}
        tmp[y*w+x]=s/c;
      }
    }
    for (let x=0;x<w;x++) {
      let s=0,c=0;
      for (let y=0;y<Math.min(r+1,h);y++){s+=tmp[y*w+x];c++;}
      for (let y=0;y<h;y++){
        if(y+r<h){s+=tmp[(y+r)*w+x];c++;} if(y-r-1>=0){s-=tmp[(y-r-1)*w+x];c--;}
        out[y*w+x]=s/c;
      }
    }
    return out;
  };
  const mI=boxMean(guide), mP=boxMean(alpha);
  const cI=boxMean(guide.map((v,i)=>v*v)), cIP=boxMean(guide.map((v,i)=>v*alpha[i]));
  const A=new Float32Array(N), B=new Float32Array(N);
  for (let i=0;i<N;i++) {
    const vI=cI[i]-mI[i]*mI[i], cvIP=cIP[i]-mI[i]*mP[i];
    A[i]=cvIP/(vI+eps); B[i]=mP[i]-A[i]*mI[i];
  }
  const mA=boxMean(A), mB=boxMean(B), out=new Float32Array(N);
  for (let i=0;i<N;i++) out[i]=Math.min(1,Math.max(0,mA[i]*guide[i]+mB[i]));
  return out;
}

function conservativeErode(alpha, w, h) {
  const out=new Float32Array(alpha.length);
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    const a=alpha[y*w+x];
    if (a>=0.4) { out[y*w+x]=a; continue; }
    let minV=a;
    for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++) {
      const nx=x+dx,ny=y+dy;
      if (nx>=0&&nx<w&&ny>=0&&ny<h) { const v=alpha[ny*w+nx]; if(v<minV) minV=v; }
    }
    out[y*w+x]=Math.max(0, a - Math.min(0.08, a-minV));
  }
  return out;
}

function resizeNN(imageData, tw, th) {
  const {width:sw,height:sh,data}=imageData;
  const out=new Uint8ClampedArray(tw*th*4);
  for (let y=0;y<th;y++) for (let x=0;x<tw;x++) {
    const sx=Math.min(Math.floor(x*sw/tw),sw-1), sy=Math.min(Math.floor(y*sh/th),sh-1);
    const si=(sy*sw+sx)*4, di=(y*tw+x)*4;
    out[di]=data[si]; out[di+1]=data[si+1]; out[di+2]=data[si+2]; out[di+3]=data[si+3];
  }
  return out;
}
