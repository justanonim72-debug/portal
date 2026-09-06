const app = document.querySelector('#app');
const video = document.querySelector('#camera');
const canvas = document.querySelector('#stage');
const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
const startBtn = document.querySelector('#startBtn');
const cameraBtn = document.querySelector('#cameraBtn');
const filterBtn = document.querySelector('#filterBtn');
const debugBtn = document.querySelector('#debugBtn');
const recBtn = document.querySelector('#recBtn');
const filterName = document.querySelector('#filterName');
const bootStatus = document.querySelector('#bootStatus');
const statusText = document.querySelector('#statusText');
const statusDot = document.querySelector('#statusDot');
const fpsEl = document.querySelector('#fps');
const hint = document.querySelector('#hint');
const toast = document.querySelector('#toast');
const recBadge = document.querySelector('#recBadge');
const recTime = document.querySelector('#recTime');

const filterCanvas = document.createElement('canvas');
const filterCtx = filterCanvas.getContext('2d', { alpha: false, desynchronized: true });
const inferenceCanvas = document.createElement('canvas');
const inferenceCtx = inferenceCanvas.getContext('2d', { alpha: false, desynchronized: true });
const recordCanvas = document.createElement('canvas');
const recordCtx = recordCanvas.getContext('2d', { alpha: false, desynchronized: true });

const MP_VERSION = '0.10.35';
const MP_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const PRESETS = [
  { name: 'THERMAL HOLO', css: 'contrast(1.38) saturate(3.4) sepia(.30) hue-rotate(145deg) brightness(1.06)', a: 'rgba(68,232,255,.96)', b: 'rgba(255,90,220,.72)', g: 'rgba(55,220,255,.9)', scan: 1, grid: 1, glitch: 0 },
  { name: 'XRAY GLITCH', css: 'grayscale(1) invert(.94) contrast(1.45) brightness(1.08)', a: 'rgba(220,246,255,.98)', b: 'rgba(104,186,255,.7)', g: 'rgba(160,220,255,.86)', scan: 1, grid: 0, glitch: 1 },
  { name: 'VOID NEON', css: 'invert(.82) hue-rotate(188deg) saturate(2.5) contrast(1.3)', a: 'rgba(155,92,255,.98)', b: 'rgba(58,233,255,.76)', g: 'rgba(100,92,255,.9)', scan: 1, grid: 0, glitch: 0 },
  { name: 'HOLOGRAM+', css: 'grayscale(.16) sepia(.7) saturate(5.2) hue-rotate(126deg) contrast(1.18) brightness(1.08)', a: 'rgba(70,235,255,.98)', b: 'rgba(255,110,225,.68)', g: 'rgba(68,225,255,.92)', scan: 1, grid: 1, glitch: 0 },
  { name: 'RAW SHIELD', css: 'none', a: 'rgba(128,241,255,.98)', b: 'rgba(255,255,255,.55)', g: 'rgba(108,233,255,.84)', scan: 0, grid: 0, glitch: 0 }
];
const LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

let trackingMode = 'boot';
let worker = null;
let workerReady = false;
let workerPending = false;
let workerCost = 45;
let workerWatchdog = null;
let fallbackLandmarker = null;
let fallbackBusy = false;
let fallbackLast = 0;
let backend = '…';
let modelReady = false;

let stream = null;
let facing = 'user';
let running = false;
let debug = false;
let presetIndex = 0;
let session = 0;
let videoFrameId = null;
let rafId = null;
let lastSent = 0;
let toastTimer = null;

let lastResult = { landmarks: [] };
let lastSeen = 0;
let lastTwoSeen = -Infinity;
let mode = null;
let target = null;
let display = null;
let history = [];
let sideSign = 0;
let flipVotes = 0;
let twoRef = null;
let portalAlpha = 0;

let filterAt = 0;
let lastRender = performance.now();
let fpsFrames = 0;
let fpsAt = performance.now();
let particles = [];
let particleAt = 0;

let recorder = null;
let recordStream = null;
let chunks = [];
let recording = false;
let recordStarted = 0;
let recordMime = '';
let recordAt = 0;

startBtn.disabled = true;
app.dataset.facing = facing;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: 0 });
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: 0 });
const mul = (a, s) => ({ x: a.x * s, y: a.y * s, z: 0 });
const dot = (a, b) => a.x * b.x + a.y * b.y;
const mid = (a, b) => ({ x: (a.x + b.x) * .5, y: (a.y + b.y) * .5, z: 0 });
const avg = arr => ({ x: arr.reduce((s,p)=>s+p.x,0)/arr.length, y: arr.reduce((s,p)=>s+p.y,0)/arr.length, z: 0 });
const norm = (v, fallback = {x:1,y:0,z:0}) => { const n = Math.hypot(v.x,v.y); return n > 1e-5 ? {x:v.x/n,y:v.y/n,z:0} : {...fallback}; };
const lerp = (a,b,t) => ({ x:a.x+(b.x-a.x)*t, y:a.y+(b.y-a.y)*t, z:0 });

function setStatus(text, state='warn') { statusText.textContent = text; statusDot.className = state; }
function pop(text) { toast.textContent = text; toast.classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(()=>toast.classList.remove('show'), 1800); }
function withTimeout(promise, ms, label) { return Promise.race([promise, new Promise((_, reject)=>setTimeout(()=>reject(new Error(`${label} timeout`)), ms))]); }

function markReady(label, modeName) {
  trackingMode = modeName;
  modelReady = true;
  startBtn.disabled = false;
  bootStatus.textContent = label;
  setStatus('Siap', 'ready');
}

async function startFallback(reason) {
  if (modelReady || trackingMode === 'fallback-loading') return;
  trackingMode = 'fallback-loading';
  if (worker) { worker.terminate(); worker = null; }
  clearTimeout(workerWatchdog);
  bootStatus.textContent = `Worker dilewati (${reason}). Menyiapkan fallback…`;
  setStatus('Fallback', 'warn');
  try {
    const mod = await withTimeout(import(MP_MODULE), 12000, 'module');
    const vision = await withTimeout(mod.FilesetResolver.forVisionTasks(MP_WASM), 12000, 'WASM');
    fallbackLandmarker = await withTimeout(mod.HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'CPU' },
      runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: .46,
      minHandPresenceConfidence: .48,
      minTrackingConfidence: .56
    }), 15000, 'model');
    backend = 'CPU fallback';
    markReady('Tracking siap · CPU fallback · Portal V6', 'fallback');
  } catch (error) {
    trackingMode = 'failed';
    bootStatus.textContent = `Tracking gagal: ${error?.message || error}`;
    setStatus('Model gagal', 'error');
    startBtn.disabled = false;
    startBtn.textContent = 'Coba lagi';
    startBtn.onclick = () => location.reload();
  }
}

function initTracking() {
  trackingMode = 'worker-loading';
  bootStatus.textContent = 'Menyiapkan tracking worker…';
  setStatus('Memuat model', 'warn');
  try {
    worker = new Worker('./tracking-worker-v6.js');
  } catch (error) {
    startFallback('worker tidak tersedia');
    return;
  }
  worker.onmessage = event => {
    const m = event.data || {};
    if (m.type === 'progress') {
      bootStatus.textContent = m.text || `Tracking: ${m.stage || 'loading'}…`;
      return;
    }
    if (m.type === 'ready') {
      clearTimeout(workerWatchdog);
      workerReady = true;
      backend = `${m.backend || 'CPU'} worker`;
      markReady(`Tracking siap · ${backend} · Portal V6`, 'worker');
      return;
    }
    if (m.type === 'fatal') {
      startFallback(m.message || 'worker init gagal');
      return;
    }
    if (m.type !== 'result' || m.session !== session) return;
    workerPending = false;
    workerCost = m.cost || workerCost;
    if (m.error) console.warn('Worker detection:', m.error);
    updateFromResult({ landmarks: m.landmarks || [] }, performance.now());
  };
  worker.onerror = error => {
    console.error(error);
    workerPending = false;
    startFallback('worker error');
  };
  worker.postMessage({ type: 'init' });
  workerWatchdog = setTimeout(() => {
    if (!modelReady) startFallback('worker > 12 detik');
  }, 12000);
}

async function openCamera() {
  if (!modelReady) { pop('Tracking belum siap.'); return; }
  if (recording) stopRecording();
  stopLoops(); session++; workerPending = false;
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream = null; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio:false, video:{ facingMode:{ideal:facing}, width:{ideal:960,max:1280}, height:{ideal:540,max:720}, frameRate:{ideal:60,min:24,max:60} } });
    video.srcObject = stream;
    if (!video.videoWidth || !video.videoHeight) await new Promise(r=>video.addEventListener('loadedmetadata', r, {once:true}));
    await video.play();
    app.dataset.facing = facing;
    resize(); resetPortal();
    running = true; app.dataset.state = 'live';
    setStatus('CARI TANGAN', 'warn'); hint.innerHTML = 'Buka <b>jempol + telunjuk</b>'; hint.classList.add('visible');
    rafId = requestAnimationFrame(renderLoop);
    startDetectionPump();
  } catch (error) {
    console.error(error);
    bootStatus.textContent = !window.isSecureContext ? 'Kamera butuh HTTPS.' : 'Izin kamera gagal.';
    setStatus('Kamera gagal', 'error');
    pop(!window.isSecureContext ? 'Buka lewat HTTPS.' : 'Izinkan kamera lalu coba lagi.');
  }
}

function stopLoops() {
  running = false;
  if (rafId !== null) cancelAnimationFrame(rafId); rafId = null;
  if (videoFrameId !== null) {
    if (video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(videoFrameId); else cancelAnimationFrame(videoFrameId);
  }
  videoFrameId = null;
}
function resetPortal() {
  lastResult={landmarks:[]}; lastSeen=0; lastTwoSeen=-Infinity; mode=null; target=null; display=null; history=[]; sideSign=0; flipVotes=0; twoRef=null; portalAlpha=0; particles=[]; filterAt=0; lastSent=0; lastRender=performance.now(); fpsFrames=0; fpsAt=performance.now();
}
function resize() {
  const w=Math.max(1,innerWidth),h=Math.max(1,innerHeight); canvas.width=Math.round(w); canvas.height=Math.round(h); canvas.style.width=`${w}px`; canvas.style.height=`${h}px`; recordCanvas.width=canvas.width; recordCanvas.height=canvas.height;
  const vw=video.videoWidth||960,vh=video.videoHeight||540,fs=Math.min(1,360/Math.max(vw,vh)); filterCanvas.width=Math.max(1,Math.round(vw*fs)); filterCanvas.height=Math.max(1,Math.round(vh*fs));
  const is=Math.min(1,320/Math.max(vw,vh)); inferenceCanvas.width=Math.max(1,Math.round(vw*is)); inferenceCanvas.height=Math.max(1,Math.round(vh*is));
}
function cover(sw,sh,dw,dh){const s=Math.max(dw/sw,dh/sh),w=sw*s,h=sh*s;return{s,x:(dw-w)*.5,y:(dh-h)*.5,w,h};}

async function sendWorkerFrame(now) {
  if (!running || !workerReady || workerPending || !worker || video.readyState < 2) return;
  const period = clamp(workerCost*.72+10, 42, 82); if (now-lastSent < period) return;
  workerPending = true; lastSent = now; const ss = session;
  try {
    const vw=video.videoWidth||960,vh=video.videoHeight||540,max=448,s=Math.min(1,max/Math.max(vw,vh)),rw=Math.max(1,Math.round(vw*s)),rh=Math.max(1,Math.round(vh*s));
    const bitmap = await createImageBitmap(video, { resizeWidth:rw, resizeHeight:rh, resizeQuality:'low' });
    if (!running || ss !== session) { bitmap.close?.(); workerPending=false; return; }
    worker.postMessage({ type:'frame', bitmap, timestamp:performance.now(), session:ss }, [bitmap]);
  } catch (error) { workerPending=false; console.warn(error); }
}

function runFallbackFrame(now) {
  if (!running || !fallbackLandmarker || fallbackBusy || video.readyState < 2 || now-fallbackLast < 90) return;
  fallbackBusy=true; fallbackLast=now;
  try {
    inferenceCtx.save(); inferenceCtx.setTransform(1,0,0,1,0,0); inferenceCtx.clearRect(0,0,inferenceCanvas.width,inferenceCanvas.height); inferenceCtx.drawImage(video,0,0,inferenceCanvas.width,inferenceCanvas.height); inferenceCtx.restore();
    const r=fallbackLandmarker.detectForVideo(inferenceCanvas, now); updateFromResult({landmarks:r.landmarks||[]}, performance.now());
  } catch(error){ console.warn(error); } finally { fallbackBusy=false; }
}
function startDetectionPump() {
  const pump = now => {
    if (!running) return;
    if (trackingMode === 'worker') sendWorkerFrame(now); else if (trackingMode === 'fallback') runFallbackFrame(now);
    videoFrameId = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(pump) : requestAnimationFrame(pump);
  };
  videoFrameId = video.requestVideoFrameCallback ? video.requestVideoFrameCallback(pump) : requestAnimationFrame(pump);
}

function toScreen(lm) {
  const vw=video.videoWidth||1,vh=video.videoHeight||1,t=cover(vw,vh,canvas.width,canvas.height),nx=facing==='user'?1-lm.x:lm.x;
  return {x:t.x+nx*vw*t.s,y:t.y+lm.y*vh*t.s,z:lm.z||0};
}
function geom(l) {
  const p=i=>toScreen(l[i]);
  const w=p(0),tm=p(2),ti=p(3),tt=p(4),im=p(5),ip=p(6),id=p(7),it=p(8),mm=p(9),mt=p(12),rm=p(13),rt=p(16),pm=p(17),pt=p(20),c=avg([w,im,mm,rm,pm]);
  return {l,w,tm,ti,tt,im,ip,id,it,mm,mt,rm,rt,pm,pt,c,cx:c.x};
}
function area(points){let a=0;for(let i=0;i<points.length;i++){const j=(i+1)%points.length;a+=points[i].x*points[j].y-points[j].x*points[i].y;}return Math.abs(a*.5);}
function angleOrder(points){const c=avg(points);return [...points].sort((a,b)=>Math.atan2(a.y-c.y,a.x-c.x)-Math.atan2(b.y-c.y,b.x-c.x));}
function rotations(points){const out=[];for(const base of[points,[...points].reverse()])for(let k=0;k<points.length;k++)out.push(Array.from({length:points.length},(_,i)=>base[(k+i)%points.length]));return out;}
function align(points,ref){const q=angleOrder(points);if(!ref)return q.map(p=>({...p}));let best=q,score=Infinity;for(const v of rotations(q)){let s=0;for(let i=0;i<v.length;i++)s+=dist(v[i],ref[i]);if(s<score){score=s;best=v;}}return best.map(p=>({...p}));}

function buildTwo(hands) {
  if(hands.length<2)return null;const s=[...hands].sort((a,b)=>a.cx-b.cx),L=s[0],R=s[1];
  if(dist(L.c,R.c)<65||dist(L.tt,L.it)<20||dist(R.tt,R.it)<20)return null;
  const corners=align([L.tt,L.it,R.tt,R.it],twoRef);if(area(corners)<900)return null;twoRef=corners.map(p=>({...p}));
  const c=avg(corners),bow=clamp(Math.sqrt(area(corners))*.045,4,18),nodes=[];
  for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4],m=mid(a,b),out=norm(sub(m,c));nodes.push({...a},add(m,mul(out,bow)));}
  return {mode:'two',nodes};
}
function chooseOneSide(h,c,n,len) {
  const root=mid(h.ti,h.ip),tipDir=sub(c,root); let signed=dot(tipDir,n);
  if(Math.abs(signed)<8){const secondary=avg([h.mt,h.rt,h.pt]);signed=dot(sub(secondary,c),n)*-.25;}
  const raw=signed>=0?1:-1;
  if(!sideSign){sideSign=raw;return raw;}
  if(raw!==sideSign&&Math.abs(signed)>Math.max(12,len*.16)){if(++flipVotes>=5){sideSign=raw;flipVotes=0;}}else flipVotes=0;
  return sideSign;
}
function buildOne(h) {
  const p0={...h.tt},p1={...h.it},len=dist(p0,p1);if(len<26)return null;
  const c=mid(p0,p1),u=norm(sub(p1,p0)),baseN={x:-u.y,y:u.x,z:0},n=mul(baseN,chooseOneSide(h,c,baseN,len));
  const depth=clamp(len*1.32,72,Math.min(canvas.width,canvas.height)*.40);
  const sec=avg([h.mt,h.rt,h.pt]);const bend=clamp(dot(sub(sec,c),u)*.06,-len*.08,len*.08);
  const farC=add(add(c,mul(n,depth)),mul(u,bend));
  const farHalf=len*.38;
  const sideDepth=depth*.46;
  const p2=add(add(p1,mul(n,sideDepth)),mul(u,len*.05));
  const p3=add(farC,mul(u,farHalf));
  const p4=add(farC,mul(u,-farHalf));
  const p5=add(add(p0,mul(n,sideDepth)),mul(u,-len*.05));
  const nodes=[p0,p1,p2,p3,p4,p5];
  return area(nodes)<800?null:{mode:'one',nodes};
}
function median(v){const a=[...v].sort((x,y)=>x-y);return a[Math.floor(a.length/2)];}
function stabilize(nodes,newMode,now) {
  if(mode!==newMode||!target||target.length!==nodes.length)history=[];mode=newMode;history.push(nodes.map(p=>({...p})));if(history.length>3)history.shift();
  let f=nodes;if(history.length>=3)f=nodes.map((_,i)=>({x:median(history.map(q=>q[i].x)),y:median(history.map(q=>q[i].y)),z:0}));
  if(target&&target.length===f.length)f=f.map((p,i)=>{const d=dist(target[i],p),max=newMode==='one'?(i<2?86:62):86;return d<=max?p:lerp(target[i],p,max/d);});
  target=f;lastSeen=now;if(!display||display.length!==f.length)display=f.map(p=>({...p}));
}
function updateFromResult(r,now) {
  lastResult=r;const hands=(r.landmarks||[]).slice(0,2).map(geom);let portal=null;
  if(hands.length>=2){portal=buildTwo(hands);if(portal)lastTwoSeen=now;}
  if(!portal&&hands.length===1&&now-lastTwoSeen>140)portal=buildOne(hands[0]);
  if(portal)stabilize(portal.nodes,portal.mode,now);
}
function smoothDisplay(dt) {
  if(!target)return;if(!display||display.length!==target.length){display=target.map(p=>({...p}));return;}
  display=target.map((p,i)=>{const d=dist(display[i],p);const anchor=mode==='one'&&i<2;const response=anchor?(d<10?22:d<35?36:52):(d<10?12:d<35?22:36);return lerp(display[i],p,1-Math.exp(-response*dt));});
}
function smoothPath(nodes,tension=.82) {
  const p=new Path2D();if(nodes.length<3)return p;p.moveTo(nodes[0].x,nodes[0].y);
  for(let i=0;i<nodes.length;i++){const p0=nodes[(i-1+nodes.length)%nodes.length],p1=nodes[i],p2=nodes[(i+1)%nodes.length],p3=nodes[(i+2)%nodes.length];const c1={x:p1.x+(p2.x-p0.x)*tension/6,y:p1.y+(p2.y-p0.y)*tension/6},c2={x:p2.x-(p3.x-p1.x)*tension/6,y:p2.y-(p3.y-p1.y)*tension/6};p.bezierCurveTo(c1.x,c1.y,c2.x,c2.y,p2.x,p2.y);}p.closePath();return p;
}

function filterFrame(now) {
  if(!target||portalAlpha<.02||video.readyState<2||now-filterAt<45)return;filterAt=now;const pr=PRESETS[presetIndex];filterCtx.save();filterCtx.setTransform(1,0,0,1,0,0);filterCtx.clearRect(0,0,filterCanvas.width,filterCanvas.height);filterCtx.filter=pr.css;if(facing==='user'){filterCtx.translate(filterCanvas.width,0);filterCtx.scale(-1,1);}filterCtx.drawImage(video,0,0,filterCanvas.width,filterCanvas.height);filterCtx.restore();
}
function drawGrid(path,a,pr){if(!pr.grid)return;ctx.save();ctx.globalAlpha=a*.10;ctx.strokeStyle=pr.a;ctx.lineWidth=.7;ctx.clip(path);for(let x=0;x<canvas.width;x+=26){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}for(let y=0;y<canvas.height;y+=26){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}ctx.restore();}
function drawParticles(nodes,a,now,pr){if(a>.55&&now-particleAt>75&&particles.length<14){particleAt=now;const edge=Math.floor(Math.random()*nodes.length),p=nodes[edge],c=avg(nodes),n=norm(sub(p,c)),speed=14+Math.random()*24;particles.push({x:p.x,y:p.y,vx:n.x*speed,vy:n.y*speed,born:now,life:280+Math.random()*340,size:.8+Math.random()*1.2});}const dt=Math.min(.05,Math.max(.001,(now-lastRender)/1000)),alive=[];ctx.save();for(const p of particles){const age=now-p.born;if(age>=p.life)continue;p.x+=p.vx*dt;p.y+=p.vy*dt;ctx.globalAlpha=a*(1-age/p.life)*.65;ctx.fillStyle=pr.a;ctx.beginPath();ctx.arc(p.x,p.y,p.size,0,Math.PI*2);ctx.fill();alive.push(p);}ctx.restore();particles=alive;}
function drawPortal(nodes,a,now) {
  if(!nodes||area(nodes)<700||a<=.01)return;const pr=PRESETS[presetIndex],path=smoothPath(nodes,mode==='one'?1.0:.78);
  ctx.save();ctx.globalAlpha=a;ctx.clip(path);const t=cover(filterCanvas.width,filterCanvas.height,canvas.width,canvas.height);ctx.drawImage(filterCanvas,t.x,t.y,t.w,t.h);
  if(pr.scan){const ys=nodes.map(p=>p.y),min=Math.min(...ys),max=Math.max(...ys),y=min+((now*.0004)%1)*Math.max(1,max-min),g=ctx.createLinearGradient(0,y-14,0,y+14);g.addColorStop(0,'rgba(150,245,255,0)');g.addColorStop(.5,'rgba(230,253,255,.16)');g.addColorStop(1,'rgba(150,245,255,0)');ctx.fillStyle=g;ctx.fillRect(0,y-16,canvas.width,32);}ctx.restore();
  drawGrid(path,a,pr);
  if(pr.glitch&&((now/100)|0)%8===0){ctx.save();ctx.globalAlpha=a*.1;ctx.translate(3,-1);ctx.clip(path);const t=cover(filterCanvas.width,filterCanvas.height,canvas.width,canvas.height);ctx.drawImage(filterCanvas,t.x,t.y,t.w,t.h);ctx.restore();}
  ctx.save();ctx.globalAlpha=a*.24;ctx.strokeStyle=pr.a;ctx.lineWidth=7;ctx.shadowBlur=20;ctx.shadowColor=pr.g;ctx.stroke(path);ctx.restore();
  ctx.save();ctx.globalAlpha=a*.94;ctx.strokeStyle='rgba(230,253,255,.98)';ctx.lineWidth=1.35;ctx.stroke(path);ctx.globalAlpha=a*.50;ctx.strokeStyle=pr.a;ctx.setLineDash([10,16]);ctx.lineDashOffset=-(now*.034)%26;ctx.lineWidth=2;ctx.stroke(path);ctx.restore();
  const pulse=.5+.5*Math.sin(now*.008);ctx.save();for(let i=0;i<nodes.length;i++){if(mode==='one'&&i>1)continue;const p=nodes[i];ctx.globalAlpha=a;ctx.fillStyle=i<2&&mode==='one'?'rgba(255,245,120,.98)':pr.a;ctx.beginPath();ctx.arc(p.x,p.y,3.5+pulse*.8,0,Math.PI*2);ctx.fill();}ctx.restore();drawParticles(nodes,a,now,pr);
}
function drawDebug() {
  if(!debug||!lastResult?.landmarks?.length)return;ctx.save();ctx.lineWidth=1.15;ctx.strokeStyle='rgba(255,255,255,.66)';for(const l of lastResult.landmarks){const pts=l.map(toScreen);for(const[a,b]of LINKS){ctx.beginPath();ctx.moveTo(pts[a].x,pts[a].y);ctx.lineTo(pts[b].x,pts[b].y);ctx.stroke();}for(let i=0;i<pts.length;i++){ctx.fillStyle=i===4||i===8?'rgba(255,240,90,1)':'rgba(85,235,255,.95)';ctx.beginPath();ctx.arc(pts[i].x,pts[i].y,i===4||i===8?4:2.3,0,Math.PI*2);ctx.fill();}}ctx.restore();
}
function updateHud(now) {
  const count=lastResult?.landmarks?.length||0,fresh=now-lastSeen<260;
  if(fresh&&mode==='two'){setStatus('PORTAL · 2 TANGAN','ready');hint.classList.remove('visible');}
  else if(fresh&&mode==='one'){setStatus('PORTAL · PINCH','ready');hint.classList.remove('visible');}
  else if(count===1){setStatus('1 TANGAN','warn');hint.innerHTML='Jempol + telunjuk = <b>kontrol utama</b>';hint.classList.add('visible');}
  else if(count>=2){setStatus('2 TANGAN','warn');hint.innerHTML='Buka <b>jempol + telunjuk</b> kedua tangan';hint.classList.add('visible');}
  else{setStatus('CARI TANGAN','warn');hint.innerHTML='Tunjukkan <b>satu atau dua tangan</b>';hint.classList.add('visible');}
}
function updateFps(now){fpsFrames++;const e=now-fpsAt;if(e>=700){fpsEl.textContent=String(Math.round(fpsFrames*1000/e));fpsFrames=0;fpsAt=now;}}

function drawVideoCover(c,w,h){const vw=video.videoWidth||1,vh=video.videoHeight||1,t=cover(vw,vh,w,h);c.save();c.setTransform(1,0,0,1,0,0);c.fillStyle='#020305';c.fillRect(0,0,w,h);if(facing==='user'){c.translate(w,0);c.scale(-1,1);const mx=w-t.x-t.w;c.drawImage(video,mx,t.y,t.w,t.h);}else c.drawImage(video,t.x,t.y,t.w,t.h);c.restore();}
function pickMime(){if(!window.MediaRecorder)return'';for(const t of['video/webm;codecs=vp8','video/webm'])if(MediaRecorder.isTypeSupported(t))return t;return'';}
function startRecording(){if(recording)return;if(!window.MediaRecorder||!recordCanvas.captureStream)return pop('Browser belum dukung rekam.');try{recordCanvas.width=canvas.width;recordCanvas.height=canvas.height;recordMime=pickMime();recordStream=recordCanvas.captureStream(24);chunks=[];recorder=recordMime?new MediaRecorder(recordStream,{mimeType:recordMime,videoBitsPerSecond:4500000}):new MediaRecorder(recordStream,{videoBitsPerSecond:4500000});recorder.ondataavailable=e=>{if(e.data?.size)chunks.push(e.data);};recorder.onstop=saveRecording;recorder.start(300);recording=true;recordStarted=performance.now();recordAt=0;recBtn.classList.add('active');recBtn.textContent='■';recBadge.classList.add('visible');recTime.textContent='00:00';pop('Rekam dimulai');}catch(e){console.error(e);pop('Gagal mulai rekam.');}}
function stopRecording(){if(!recording)return;recording=false;recBtn.classList.remove('active');recBtn.textContent='●';recBadge.classList.remove('visible');if(recorder&&recorder.state!=='inactive')recorder.stop();}
function saveRecording(){try{const blob=new Blob(chunks,{type:recordMime||'video/webm'});chunks=[];if(!blob.size)return pop('Rekaman kosong.');const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`portal-${Date.now()}.webm`;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),5000);pop('Rekaman disimpan ✓');}finally{recordStream?.getTracks().forEach(t=>t.stop());recordStream=null;recorder=null;}}
function drawRecord(now){if(!recording||now-recordAt<41)return;recordAt=now;drawVideoCover(recordCtx,recordCanvas.width,recordCanvas.height);recordCtx.drawImage(canvas,0,0,recordCanvas.width,recordCanvas.height);const s=Math.floor((now-recordStarted)/1000);recTime.textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;}

function renderLoop(now) {
  if(!running)return;const dt=clamp((now-lastRender)/1000,1/240,.05);ctx.clearRect(0,0,canvas.width,canvas.height);const fresh=now-lastSeen<260;smoothDisplay(dt);const desired=fresh?1:0;portalAlpha+=(desired-portalAlpha)*(1-Math.exp(-(fresh?14:7)*dt));if(!fresh&&portalAlpha<.01)portalAlpha=0;filterFrame(now);if(display&&portalAlpha>0)drawPortal(display,portalAlpha,now);drawDebug();updateHud(now);updateFps(now);drawRecord(now);lastRender=now;rafId=requestAnimationFrame(renderLoop);
}

startBtn.addEventListener('click', openCamera);
cameraBtn.addEventListener('click', async()=>{if(recording)stopRecording();facing=facing==='user'?'environment':'user';app.dataset.facing=facing;pop(facing==='user'?'Kamera depan':'Kamera belakang');await openCamera();});
filterBtn.addEventListener('click',()=>{presetIndex=(presetIndex+1)%PRESETS.length;filterName.textContent=PRESETS[presetIndex].name;filterAt=0;pop(PRESETS[presetIndex].name);});
debugBtn.addEventListener('click',()=>{debug=!debug;debugBtn.classList.toggle('active',debug);pop(debug?'Landmark ON · kuning = jempol/telunjuk':'Landmark OFF');});
recBtn.addEventListener('click',()=>recording?stopRecording():startRecording());
window.addEventListener('resize',()=>running&&resize());
window.addEventListener('orientationchange',()=>setTimeout(()=>running&&resize(),180));
document.addEventListener('visibilitychange',()=>{if(!running)return;if(document.hidden){if(recording)stopRecording();stopLoops();}else{running=true;lastRender=performance.now();rafId=requestAnimationFrame(renderLoop);startDetectionPump();}});

initTracking();
