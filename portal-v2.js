import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const app = document.querySelector("#app");
const video = document.querySelector("#camera");
const canvas = document.querySelector("#stage");
const ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
const startBtn = document.querySelector("#startBtn");
const cameraBtn = document.querySelector("#cameraBtn");
const filterBtn = document.querySelector("#filterBtn");
const debugBtn = document.querySelector("#debugBtn");
const filterName = document.querySelector("#filterName");
const bootStatus = document.querySelector("#bootStatus");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const fpsEl = document.querySelector("#fps");
const hint = document.querySelector("#hint");
const toast = document.querySelector("#toast");

const inferenceCanvas = document.createElement("canvas");
const inferenceCtx = inferenceCanvas.getContext("2d", { alpha: false, desynchronized: true });
const filterCanvas = document.createElement("canvas");
const filterCtx = filterCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

const FILTERS = [
  { name: "THERMAL", type: "thermal" },
  { name: "HOLOGRAM", type: "css", css: "grayscale(.18) sepia(.7) saturate(5.5) hue-rotate(128deg) contrast(1.18) brightness(1.08)" },
  { name: "NEON", type: "css", css: "saturate(3.2) hue-rotate(118deg) contrast(1.28) brightness(1.05)" },
  { name: "XRAY", type: "css", css: "grayscale(1) invert(.96) contrast(1.45) brightness(1.08)" },
  { name: "VOID", type: "css", css: "invert(.82) hue-rotate(188deg) saturate(2.4) contrast(1.28)" },
  { name: "MONO", type: "css", css: "grayscale(1) contrast(1.25) brightness(1.04)" },
  { name: "RAW", type: "css", css: "none" }
];

const handConnections = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

let handLandmarker = null;
let modelReady = false;
let modelBackend = "GPU";
let stream = null;
let facingMode = "user";
let running = false;
let filterIndex = 0;
let debug = false;
let animationFrameId = null;
let inferenceTimer = null;
let filterTimerAt = 0;
let lastRenderAt = performance.now();
let lastResult = null;
let lastPortalSeenAt = 0;
let lastTwoHandSeenAt = 0;
let portalTarget = null;
let displayQuad = null;
let portalVelocity = null;
let portalAlpha = 0;
let portalMode = null;
let toastTimer = null;
let fpsFrames = 0;
let fpsStarted = performance.now();
let measuredFps = 0;
let lastParticleSpawn = 0;
let particles = [];
let inferenceCostMs = 40;

startBtn.disabled = true;
app.dataset.facing = facingMode;

const THERMAL_LUT = new Uint8Array(256 * 3);
for (let v = 0; v < 256; v++) {
  const x = v / 255;
  let r, g, b;
  if (x < 0.17) {
    r = 8; g = 4 + 25 * (x / 0.17); b = 55 + 165 * (x / 0.17);
  } else if (x < 0.36) {
    r = 0; g = 35 + 220 * ((x - 0.17) / 0.19); b = 255;
  } else if (x < 0.56) {
    r = 255 * ((x - 0.36) / 0.2); g = 255; b = 255 * (1 - (x - 0.36) / 0.2);
  } else if (x < 0.79) {
    r = 255; g = 255 * (1 - (x - 0.56) / 0.23); b = 0;
  } else {
    const t = (x - 0.79) / 0.21;
    r = 255; g = 180 + 75 * t; b = 55 + 200 * t;
  }
  const i = v * 3;
  THERMAL_LUT[i] = r | 0;
  THERMAL_LUT[i + 1] = g | 0;
  THERMAL_LUT[i + 2] = b | 0;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function midpoint(a, b) { return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5, z: ((a.z || 0) + (b.z || 0)) * 0.5 }; }

function setStatus(text, state = "warn") {
  statusText.textContent = text;
  statusDot.className = state;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
}

async function initializeModel() {
  bootStatus.textContent = "Memuat MediaPipe Hand Landmarker…";
  setStatus("Memuat model", "warn");
  const vision = await FilesetResolver.forVisionTasks(MP_ROOT);
  const common = {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.45
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" }
    });
    modelBackend = "GPU";
    bootStatus.textContent = "Hand tracking siap · GPU · mode 1–2 tangan";
  } catch (gpuError) {
    console.warn("GPU delegate gagal, fallback CPU", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" }
    });
    modelBackend = "CPU";
    bootStatus.textContent = "Hand tracking siap · CPU fallback · mode 1–2 tangan";
  }

  modelReady = true;
  setStatus("Siap", "ready");
  startBtn.disabled = false;
}

async function openCamera() {
  if (!modelReady) {
    showToast("Model masih dimuat. Tunggu sebentar.");
    return;
  }
  stopLoops();
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: 960, max: 1280 },
      height: { ideal: 540, max: 720 },
      frameRate: { ideal: 60, min: 24, max: 60 }
    }
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    if (!video.videoWidth || !video.videoHeight) {
      await new Promise(resolve => video.addEventListener("loadedmetadata", resolve, { once: true }));
    }
    await video.play();
    app.dataset.facing = facingMode;
    resizeCanvases();
    resetTrackingState();
    running = true;
    app.dataset.state = "live";
    setStatus("CARI TANGAN", "warn");
    hint.innerHTML = "Tunjukkan <b>satu atau dua tangan</b>";
    hint.classList.add("visible");
    animationFrameId = requestAnimationFrame(renderLoop);
    inferenceTimer = setTimeout(inferenceLoop, 20);
  } catch (error) {
    console.error(error);
    const insecure = !window.isSecureContext;
    bootStatus.textContent = insecure ? "Kamera butuh HTTPS. Buka dari GitHub Pages/Vercel." : "Izin kamera gagal. Izinkan kamera lalu coba lagi.";
    setStatus("Kamera gagal", "error");
    showToast(insecure ? "Kamera browser hanya aktif lewat HTTPS." : "Izin kamera belum diberikan.");
  }
}

function stopLoops() {
  running = false;
  if (animationFrameId !== null) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
  if (inferenceTimer !== null) { clearTimeout(inferenceTimer); inferenceTimer = null; }
}

function resetTrackingState() {
  lastResult = null;
  lastPortalSeenAt = 0;
  lastTwoHandSeenAt = 0;
  portalTarget = null;
  displayQuad = null;
  portalVelocity = null;
  portalAlpha = 0;
  portalMode = null;
  particles = [];
  filterTimerAt = 0;
  lastRenderAt = performance.now();
  fpsFrames = 0;
  fpsStarted = performance.now();
}

function resizeCanvases() {
  const cssW = Math.max(1, window.innerWidth);
  const cssH = Math.max(1, window.innerHeight);
  canvas.width = Math.round(cssW);
  canvas.height = Math.round(cssH);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const vw = video.videoWidth || 720;
  const vh = video.videoHeight || 1280;
  sizeProcessingCanvas(inferenceCanvas, vw, vh, 384);
  sizeProcessingCanvas(filterCanvas, vw, vh, 480);
}

function sizeProcessingCanvas(target, vw, vh, maxSide) {
  const scale = Math.min(1, maxSide / Math.max(vw, vh));
  target.width = Math.max(1, Math.round(vw * scale));
  target.height = Math.max(1, Math.round(vh * scale));
}

function coverTransform(srcW, srcH, dstW, dstH) {
  const scale = Math.max(dstW / srcW, dstH / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  return { scale, x: (dstW - drawW) * 0.5, y: (dstH - drawH) * 0.5, drawW, drawH };
}

function drawVideoMirrored(targetCtx, targetCanvas) {
  targetCtx.save();
  targetCtx.setTransform(1, 0, 0, 1, 0, 0);
  targetCtx.filter = "none";
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  if (facingMode === "user") {
    targetCtx.translate(targetCanvas.width, 0);
    targetCtx.scale(-1, 1);
  }
  targetCtx.drawImage(video, 0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.restore();
}

function landmarkToScreen(lm, transform) {
  return {
    x: transform.x + lm.x * inferenceCanvas.width * transform.scale,
    y: transform.y + lm.y * inferenceCanvas.height * transform.scale,
    z: lm.z || 0
  };
}

function handGeometry(landmarks, transform) {
  const p = index => landmarkToScreen(landmarks[index], transform);
  const thumb = p(4), indexTip = p(8), wrist = p(0), indexMcp = p(5), middleMcp = p(9), pinkyMcp = p(17);
  const palmCenter = {
    x: (wrist.x + indexMcp.x + middleMcp.x + pinkyMcp.x) * 0.25,
    y: (wrist.y + indexMcp.y + middleMcp.y + pinkyMcp.y) * 0.25
  };
  const palmScale = Math.max(24, dist(indexMcp, pinkyMcp) * 0.8 + dist(wrist, middleMcp) * 0.7);
  return { landmarks, thumb, indexTip, wrist, indexMcp, middleMcp, pinkyMcp, palmCenter, palmScale, centerX: palmCenter.x };
}

function buildTwoHandPortal(hands) {
  if (hands.length < 2) return null;
  const sorted = [...hands].sort((a, b) => a.centerX - b.centerX);
  const left = sorted[0], right = sorted[1];
  if (right.centerX - left.centerX < 56) return null;
  const leftTop = left.indexTip.y < left.thumb.y ? left.indexTip : left.thumb;
  const leftBottom = left.indexTip.y < left.thumb.y ? left.thumb : left.indexTip;
  const rightTop = right.indexTip.y < right.thumb.y ? right.indexTip : right.thumb;
  const rightBottom = right.indexTip.y < right.thumb.y ? right.thumb : right.indexTip;
  if (dist(leftTop, leftBottom) < 24 || dist(rightTop, rightBottom) < 24) return null;
  return { mode: "two", quad: [leftTop, rightTop, rightBottom, leftBottom], hands: sorted };
}

function buildOneHandPortal(hand) {
  if (!hand) return null;
  const a = hand.thumb, b = hand.indexTip;
  const edgeLen = dist(a, b);
  if (edgeLen < 24) return null;
  const mid = midpoint(a, b);
  const ex = (b.x - a.x) / edgeLen, ey = (b.y - a.y) / edgeLen;
  let nx = -ey, ny = ex;
  const outwardX = mid.x - hand.palmCenter.x, outwardY = mid.y - hand.palmCenter.y;
  if (nx * outwardX + ny * outwardY < 0) { nx *= -1; ny *= -1; }
  const desiredEdge = clamp(hand.palmScale * 1.12, edgeLen, edgeLen * 2.1);
  const half = desiredEdge * 0.5;
  const p0 = { x: mid.x - ex * half, y: mid.y - ey * half, z: 0 };
  const p1 = { x: mid.x + ex * half, y: mid.y + ey * half, z: 0 };
  const depth = clamp(hand.palmScale * 1.72, 88, Math.min(canvas.width, canvas.height) * 0.36);
  const p2 = { x: p1.x + nx * depth, y: p1.y + ny * depth, z: 0 };
  const p3 = { x: p0.x + nx * depth, y: p0.y + ny * depth, z: 0 };
  return { mode: "one", quad: [p0, p1, p2, p3], hands: [hand] };
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    area += points[i].x * points[j].y - points[j].x * points[i].y;
  }
  return Math.abs(area * 0.5);
}

function updatePortalTarget(result, now) {
  lastResult = result;
  const transform = coverTransform(inferenceCanvas.width, inferenceCanvas.height, canvas.width, canvas.height);
  const hands = (result?.landmarks || []).slice(0, 2).map(lm => handGeometry(lm, transform));
  let next = null;
  if (hands.length >= 2) {
    next = buildTwoHandPortal(hands);
    if (next) lastTwoHandSeenAt = now;
  }
  if (!next && hands.length === 1 && now - lastTwoHandSeenAt > 240) next = buildOneHandPortal(hands[0]);

  if (next && polygonArea(next.quad) >= 1250) {
    lastPortalSeenAt = now;
    portalMode = next.mode;
    if (portalTarget && portalTarget.quad?.length === 4) {
      const dt = Math.max(16, now - portalTarget.at);
      portalVelocity = next.quad.map((p, i) => ({
        x: clamp((p.x - portalTarget.quad[i].x) / dt, -1.8, 1.8),
        y: clamp((p.y - portalTarget.quad[i].y) / dt, -1.8, 1.8)
      }));
    } else portalVelocity = next.quad.map(() => ({ x: 0, y: 0 }));
    portalTarget = { ...next, at: now };
    if (!displayQuad) displayQuad = next.quad.map(p => ({ ...p }));
  }
}

function predictedTarget(now) {
  if (!portalTarget) return null;
  const lead = clamp(now - portalTarget.at, 0, 90) * 0.42;
  return portalTarget.quad.map((p, i) => ({
    x: p.x + (portalVelocity?.[i]?.x || 0) * lead,
    y: p.y + (portalVelocity?.[i]?.y || 0) * lead,
    z: p.z || 0
  }));
}

function smoothDisplayQuad(target, dt) {
  if (!target) return;
  if (!displayQuad) { displayQuad = target.map(p => ({ ...p })); return; }
  displayQuad = target.map((p, i) => {
    const prev = displayQuad[i] || p;
    const d = Math.hypot(p.x - prev.x, p.y - prev.y);
    const response = 11 + Math.min(22, d * 0.09);
    const amount = 1 - Math.exp(-response * dt);
    return { x: prev.x + (p.x - prev.x) * amount, y: prev.y + (p.y - prev.y) * amount, z: 0 };
  });
}

function inferenceLoop() {
  if (!running || !handLandmarker || video.readyState < 2) return;
  drawVideoMirrored(inferenceCtx, inferenceCanvas);
  const stamp = performance.now();
  const started = performance.now();
  try {
    const result = handLandmarker.detectForVideo(inferenceCanvas, stamp);
    inferenceCostMs = performance.now() - started;
    updatePortalTarget(result, performance.now());
  } catch (error) {
    console.warn("Hand inference gagal", error);
  }
  const baseMin = modelBackend === "GPU" ? 58 : 92;
  const targetPeriod = clamp(inferenceCostMs * 1.65, baseMin, 155);
  const wait = Math.max(8, targetPeriod - inferenceCostMs);
  inferenceTimer = setTimeout(inferenceLoop, wait);
}

function updateFilteredFrame(now) {
  if (!portalTarget || portalAlpha < 0.04 || video.readyState < 2) return;
  const current = FILTERS[filterIndex];
  const interval = current.type === "thermal" ? 88 : 48;
  if (now - filterTimerAt < interval) return;
  filterTimerAt = now;
  filterCtx.save();
  filterCtx.setTransform(1, 0, 0, 1, 0, 0);
  filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);
  filterCtx.filter = current.type === "css" ? current.css : "none";
  if (facingMode === "user") { filterCtx.translate(filterCanvas.width, 0); filterCtx.scale(-1, 1); }
  filterCtx.drawImage(video, 0, 0, filterCanvas.width, filterCanvas.height);
  filterCtx.restore();

  if (current.type === "thermal") {
    const image = filterCtx.getImageData(0, 0, filterCanvas.width, filterCanvas.height);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const lum = (data[i] * 54 + data[i + 1] * 183 + data[i + 2] * 19) >> 8;
      const lut = lum * 3;
      data[i] = THERMAL_LUT[lut]; data[i + 1] = THERMAL_LUT[lut + 1]; data[i + 2] = THERMAL_LUT[lut + 2];
    }
    filterCtx.putImageData(image, 0, 0);
  }
}

function pathQuad(quad) {
  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < quad.length; i++) ctx.lineTo(quad[i].x, quad[i].y);
  ctx.closePath();
}

function quadCenter(quad) {
  return { x: quad.reduce((s, p) => s + p.x, 0) / quad.length, y: quad.reduce((s, p) => s + p.y, 0) / quad.length };
}

function drawPortal(quad, alpha, now) {
  if (!quad || polygonArea(quad) < 1000 || alpha <= 0.01) return;
  ctx.save();
  ctx.globalAlpha = clamp(alpha, 0, 1);
  pathQuad(quad);
  ctx.clip();
  const t = coverTransform(filterCanvas.width, filterCanvas.height, canvas.width, canvas.height);
  ctx.drawImage(filterCanvas, t.x, t.y, t.drawW, t.drawH);

  const ys = quad.map(p => p.y), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scanY = minY + ((now * 0.00034) % 1) * Math.max(1, maxY - minY);
  const grad = ctx.createLinearGradient(0, scanY - 14, 0, scanY + 14);
  grad.addColorStop(0, "rgba(130,236,255,0)");
  grad.addColorStop(0.5, "rgba(220,250,255,0.18)");
  grad.addColorStop(1, "rgba(130,236,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, scanY - 18, canvas.width, 36);
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = alpha * 0.26;
  ctx.lineWidth = 7;
  ctx.strokeStyle = "rgba(82,220,255,.85)";
  ctx.shadowBlur = 18;
  ctx.shadowColor = "rgba(80,220,255,.9)";
  pathQuad(quad); ctx.stroke(); ctx.restore();

  ctx.save();
  ctx.globalAlpha = alpha * 0.34;
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = "rgba(255,80,220,.72)";
  ctx.translate(-1.5, 0.5);
  pathQuad(quad); ctx.stroke(); ctx.restore();

  ctx.save();
  ctx.globalAlpha = alpha * 0.95;
  ctx.lineWidth = 1.35;
  ctx.strokeStyle = "rgba(224,252,255,.98)";
  pathQuad(quad); ctx.stroke();
  ctx.globalAlpha = alpha * 0.46;
  ctx.setLineDash([11, 17]);
  ctx.lineDashOffset = -(now * 0.035) % 28;
  ctx.strokeStyle = "rgba(96,230,255,.95)";
  ctx.lineWidth = 2.1;
  pathQuad(quad); ctx.stroke(); ctx.restore();

  drawCornerNodes(quad, alpha, now);
  spawnAndDrawParticles(quad, alpha, now);
}

function drawCornerNodes(quad, alpha, now) {
  const pulse = 0.5 + 0.5 * Math.sin(now * 0.008);
  ctx.save();
  for (const p of quad) {
    ctx.globalAlpha = alpha * (0.68 + pulse * 0.24);
    ctx.beginPath(); ctx.arc(p.x, p.y, 3.2 + pulse * 1.2, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(230,253,255,.98)"; ctx.fill();
    ctx.globalAlpha = alpha * 0.18;
    ctx.beginPath(); ctx.arc(p.x, p.y, 9 + pulse * 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(70,220,255,.9)"; ctx.fill();
  }
  ctx.restore();
}

function spawnAndDrawParticles(quad, alpha, now) {
  if (alpha > 0.45 && now - lastParticleSpawn > 42 && particles.length < 34) {
    lastParticleSpawn = now;
    const edge = Math.floor(Math.random() * 4), a = quad[edge], b = quad[(edge + 1) % 4], t = Math.random();
    const x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t, center = quadCenter(quad);
    let nx = x - center.x, ny = y - center.y;
    const n = Math.max(1, Math.hypot(nx, ny)); nx /= n; ny /= n;
    const speed = 18 + Math.random() * 42;
    particles.push({ x, y, vx: nx * speed + (Math.random() - 0.5) * 18, vy: ny * speed + (Math.random() - 0.5) * 18, born: now, life: 330 + Math.random() * 420, size: 0.8 + Math.random() * 1.8 });
  }
  const dt = Math.min(0.05, Math.max(0.001, (now - lastRenderAt) / 1000));
  const alive = [];
  ctx.save();
  for (const p of particles) {
    const age = now - p.born;
    if (age >= p.life) continue;
    p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 10 * dt;
    const fade = 1 - age / p.life;
    ctx.globalAlpha = alpha * fade * 0.75;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(150,242,255,.95)"; ctx.fill();
    alive.push(p);
  }
  ctx.restore();
  particles = alive;
}

function drawDebug(result) {
  if (!debug || !result?.landmarks?.length) return;
  const transform = coverTransform(inferenceCanvas.width, inferenceCanvas.height, canvas.width, canvas.height);
  ctx.save();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = "rgba(255,255,255,.58)";
  ctx.fillStyle = "rgba(99,231,255,.9)";
  for (const landmarks of result.landmarks) {
    const pts = landmarks.map(lm => landmarkToScreen(lm, transform));
    for (const [a, b] of handConnections) {
      ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
    }
    for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 2.3, 0, Math.PI * 2); ctx.fill(); }
  }
  ctx.restore();
}

function updateHud(now) {
  const handCount = lastResult?.landmarks?.length || 0;
  const fresh = now - lastPortalSeenAt < 260;
  if (fresh && portalMode === "two") {
    setStatus("PORTAL · 2 TANGAN", "ready"); hint.classList.remove("visible");
  } else if (fresh && portalMode === "one") {
    setStatus("PORTAL · 1 TANGAN", "ready"); hint.classList.remove("visible");
  } else if (handCount === 1) {
    setStatus("1 TANGAN", "warn"); hint.innerHTML = "Buka <b>jempol + telunjuk</b>"; hint.classList.add("visible");
  } else if (handCount >= 2) {
    setStatus("2 TANGAN", "warn"); hint.innerHTML = "Jauhkan sedikit <b>jempol + telunjuk</b>"; hint.classList.add("visible");
  } else {
    setStatus("CARI TANGAN", "warn"); hint.innerHTML = "Tunjukkan <b>satu atau dua tangan</b>"; hint.classList.add("visible");
  }
}

function updateFps(now) {
  fpsFrames++;
  const elapsed = now - fpsStarted;
  if (elapsed >= 650) {
    measuredFps = Math.round((fpsFrames * 1000) / elapsed);
    fpsEl.textContent = String(measuredFps);
    fpsFrames = 0;
    fpsStarted = now;
  }
}

function renderLoop(now) {
  if (!running) return;
  const dt = clamp((now - lastRenderAt) / 1000, 1 / 240, 0.05);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const portalFresh = now - lastPortalSeenAt < 290;
  const target = predictedTarget(now);
  if (target) smoothDisplayQuad(target, dt);
  const alphaSpeed = portalFresh ? 10 : 4.8;
  const desiredAlpha = portalFresh ? 1 : 0;
  portalAlpha += (desiredAlpha - portalAlpha) * (1 - Math.exp(-alphaSpeed * dt));
  if (portalAlpha < 0.015 && !portalFresh) portalAlpha = 0;
  updateFilteredFrame(now);
  if (displayQuad && portalAlpha > 0) drawPortal(displayQuad, portalAlpha, now);
  drawDebug(lastResult);
  updateHud(now);
  updateFps(now);
  lastRenderAt = now;
  animationFrameId = requestAnimationFrame(renderLoop);
}

startBtn.addEventListener("click", openCamera);
cameraBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  app.dataset.facing = facingMode;
  showToast(facingMode === "user" ? "Kamera depan" : "Kamera belakang");
  await openCamera();
});
filterBtn.addEventListener("click", () => {
  filterIndex = (filterIndex + 1) % FILTERS.length;
  filterName.textContent = FILTERS[filterIndex].name;
  filterTimerAt = 0;
  showToast(`Filter ${FILTERS[filterIndex].name}`);
});
debugBtn.addEventListener("click", () => {
  debug = !debug;
  debugBtn.classList.toggle("active", debug);
  showToast(debug ? "Landmark tangan: ON" : "Landmark tangan: OFF");
});
window.addEventListener("resize", () => { if (running) resizeCanvases(); });
window.addEventListener("orientationchange", () => { setTimeout(() => running && resizeCanvases(), 180); });
document.addEventListener("visibilitychange", () => {
  if (!running) return;
  if (document.hidden) {
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    if (inferenceTimer !== null) clearTimeout(inferenceTimer);
    animationFrameId = null; inferenceTimer = null;
  } else {
    lastRenderAt = performance.now();
    animationFrameId = requestAnimationFrame(renderLoop);
    inferenceTimer = setTimeout(inferenceLoop, 30);
  }
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./sw.js").catch(error => console.warn("SW gagal", error));
  });
}

initializeModel().catch(error => {
  console.error(error);
  bootStatus.textContent = "Model gagal dimuat. Cek koneksi lalu refresh.";
  setStatus("Model gagal", "error");
  startBtn.disabled = true;
});
