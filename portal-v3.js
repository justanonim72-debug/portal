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

const filterCanvas = document.createElement("canvas");
const filterCtx = filterCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

const FILTERS = [
  { name: "THERMAL", type: "thermal" },
  { name: "HOLOGRAM", type: "css", css: "grayscale(.12) sepia(.72) saturate(5.8) hue-rotate(128deg) contrast(1.2) brightness(1.06)" },
  { name: "NEON", type: "css", css: "saturate(3.1) hue-rotate(118deg) contrast(1.28) brightness(1.04)" },
  { name: "XRAY", type: "css", css: "grayscale(1) invert(.96) contrast(1.48) brightness(1.08)" },
  { name: "VOID", type: "css", css: "invert(.84) hue-rotate(188deg) saturate(2.5) contrast(1.3)" },
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
let lastVideoTime = -1;
let lastRenderAt = performance.now();
let lastInferenceAt = 0;
let inferenceCostMs = 35;

let lastResult = null;
let lastPortalSeenAt = 0;
let lastTwoHandSeenAt = -Infinity;
let portalMode = null;
let targetQuad = null;
let displayQuad = null;
let targetHistory = [];
let portalAlpha = 0;

let filterTimerAt = 0;
let toastTimer = null;
let fpsFrames = 0;
let fpsStarted = performance.now();
let lastParticleSpawn = 0;
let particles = [];

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
function midpoint(a, b) {
  return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5, z: ((a.z || 0) + (b.z || 0)) * 0.5 };
}
function add(a, b) { return { x: a.x + b.x, y: a.y + b.y }; }
function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y }; }
function mul(a, s) { return { x: a.x * s, y: a.y * s }; }
function dot(a, b) { return a.x * b.x + a.y * b.y; }
function normalize(v) {
  const n = Math.hypot(v.x, v.y);
  return n > 1e-5 ? { x: v.x / n, y: v.y / n } : { x: 1, y: 0 };
}

function setStatus(text, state = "warn") {
  statusText.textContent = text;
  statusDot.className = state;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 1800);
}

async function initializeModel() {
  bootStatus.textContent = "Memuat MediaPipe Hand Landmarker…";
  setStatus("Memuat model", "warn");

  const vision = await FilesetResolver.forVisionTasks(MP_ROOT);
  const common = {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.52,
    minTrackingConfidence: 0.64
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" }
    });
    modelBackend = "GPU";
    bootStatus.textContent = "Tracking siap · GPU · Portal V3";
  } catch (gpuError) {
    console.warn("GPU delegate gagal, fallback CPU", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" }
    });
    modelBackend = "CPU";
    bootStatus.textContent = "Tracking siap · CPU fallback · Portal V3";
  }

  modelReady = true;
  setStatus("Siap", "ready");
  startBtn.disabled = false;
}

async function openCamera() {
  if (!modelReady) {
    showToast("Model masih dimuat.");
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
      width: { ideal: 1280 },
      height: { ideal: 720 },
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
    bootStatus.textContent = insecure
      ? "Kamera butuh HTTPS."
      : "Izin kamera gagal. Izinkan kamera lalu coba lagi.";
    setStatus("Kamera gagal", "error");
    showToast(insecure ? "Buka Portal lewat HTTPS." : "Izin kamera belum diberikan.");
  }
}

function stopLoops() {
  running = false;
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
  if (inferenceTimer !== null) {
    clearTimeout(inferenceTimer);
    inferenceTimer = null;
  }
}

function resetTrackingState() {
  lastVideoTime = -1;
  lastInferenceAt = 0;
  lastResult = null;
  lastPortalSeenAt = 0;
  lastTwoHandSeenAt = -Infinity;
  portalMode = null;
  targetQuad = null;
  displayQuad = null;
  targetHistory = [];
  portalAlpha = 0;
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

  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const maxSide = 520;
  const scale = Math.min(1, maxSide / Math.max(vw, vh));
  filterCanvas.width = Math.max(1, Math.round(vw * scale));
  filterCanvas.height = Math.max(1, Math.round(vh * scale));
}

function coverTransform(srcW, srcH, dstW, dstH) {
  const scale = Math.max(dstW / srcW, dstH / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  return {
    scale,
    x: (dstW - drawW) * 0.5,
    y: (dstH - drawH) * 0.5,
    drawW,
    drawH
  };
}

function landmarkToScreen(lm) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const t = coverTransform(vw, vh, canvas.width, canvas.height);
  const nx = facingMode === "user" ? 1 - lm.x : lm.x;
  return {
    x: t.x + nx * vw * t.scale,
    y: t.y + lm.y * vh * t.scale,
    z: lm.z || 0
  };
}

function handGeometry(landmarks) {
  const p = i => landmarkToScreen(landmarks[i]);
  const wrist = p(0);
  const thumb = p(4);
  const indexMcp = p(5);
  const indexTip = p(8);
  const middleMcp = p(9);
  const middleTip = p(12);
  const ringMcp = p(13);
  const pinkyMcp = p(17);
  const pinkyTip = p(20);

  const palmCenter = {
    x: (wrist.x + indexMcp.x + middleMcp.x + ringMcp.x + pinkyMcp.x) / 5,
    y: (wrist.y + indexMcp.y + middleMcp.y + ringMcp.y + pinkyMcp.y) / 5
  };
  const palmWidth = dist(indexMcp, pinkyMcp);
  const palmHeight = dist(wrist, middleMcp);

  return {
    landmarks,
    wrist, thumb, indexMcp, indexTip, middleMcp, middleTip,
    ringMcp, pinkyMcp, pinkyTip, palmCenter, palmWidth, palmHeight,
    centerX: palmCenter.x
  };
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length;
    area += points[i].x * points[j].y - points[j].x * points[i].y;
  }
  return Math.abs(area * 0.5);
}

function angleOrder(points) {
  const c = {
    x: points.reduce((s, p) => s + p.x, 0) / points.length,
    y: points.reduce((s, p) => s + p.y, 0) / points.length
  };
  return [...points].sort((a, b) =>
    Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x)
  );
}

function allQuadOrders(ordered) {
  const variants = [];
  for (const base of [ordered, [...ordered].reverse()]) {
    for (let k = 0; k < 4; k++) {
      variants.push([base[k], base[(k + 1) % 4], base[(k + 2) % 4], base[(k + 3) % 4]]);
    }
  }
  return variants;
}

function alignQuad(points, reference) {
  const ordered = angleOrder(points);
  if (!reference) {
    let best = ordered;
    let bestScore = Infinity;
    for (const q of allQuadOrders(ordered)) {
      const score = q[0].x + q[0].y * 0.35;
      if (score < bestScore) { bestScore = score; best = q; }
    }
    return best.map(p => ({ ...p }));
  }

  let best = null;
  let bestScore = Infinity;
  for (const q of allQuadOrders(ordered)) {
    let score = 0;
    for (let i = 0; i < 4; i++) score += dist(q[i], reference[i]);
    if (score < bestScore) { bestScore = score; best = q; }
  }
  return best.map(p => ({ ...p }));
}

function buildTwoHandPortal(hands) {
  if (hands.length < 2) return null;
  const sorted = [...hands].sort((a, b) => a.centerX - b.centerX);
  const left = sorted[0];
  const right = sorted[1];

  if (dist(left.palmCenter, right.palmCenter) < 70) return null;
  if (dist(left.thumb, left.indexTip) < 22 || dist(right.thumb, right.indexTip) < 22) return null;

  // ZERO synthetic corners: every portal corner is an actual fingertip.
  const anchors = [left.thumb, left.indexTip, right.thumb, right.indexTip];
  const quad = alignQuad(anchors, portalMode === "two" ? targetQuad : null);

  if (polygonArea(quad) < 1500) return null;
  return { mode: "two", quad, hands: sorted };
}

function buildOneHandPortal(hand) {
  if (!hand) return null;
  if (hand.palmWidth < 28 || hand.palmHeight < 34) return null;

  // Stable palm coordinate frame:
  // U = across the knuckles, V = wrist -> middle MCP.
  // Unlike V2, no direction is guessed from thumb/index movement.
  const u = normalize(sub(hand.pinkyMcp, hand.indexMcp));
  const vRaw = normalize(sub(hand.middleMcp, hand.wrist));
  let v = { x: -u.y, y: u.x };
  if (dot(v, vRaw) < 0) v = mul(v, -1);

  const width = clamp(hand.palmWidth * 2.55, 105, Math.min(canvas.width * 0.58, 330));
  const height = clamp(hand.palmHeight * 2.85, 125, Math.min(canvas.height * 0.42, 380));

  // Center stays physically tied to the palm and travels with it.
  const base = midpoint(hand.wrist, hand.middleMcp);
  const center = add(base, mul(v, height * 0.20));
  const hw = width * 0.5;
  const hh = height * 0.5;

  const quad = [
    add(add(center, mul(u, -hw)), mul(v, -hh)),
    add(add(center, mul(u,  hw)), mul(v, -hh)),
    add(add(center, mul(u,  hw)), mul(v,  hh)),
    add(add(center, mul(u, -hw)), mul(v,  hh))
  ];

  return { mode: "one", quad, hands: [hand] };
}

function median(values) {
  const a = [...values].sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

function stabilizeTarget(rawQuad, mode, now) {
  let aligned = rawQuad.map(p => ({ ...p }));
  if (targetQuad && portalMode === mode) aligned = alignQuad(aligned, targetQuad);

  if (portalMode !== mode) targetHistory = [];
  portalMode = mode;

  targetHistory.push(aligned);
  if (targetHistory.length > 3) targetHistory.shift();

  let filtered = aligned;
  if (targetHistory.length >= 3) {
    filtered = aligned.map((_, i) => ({
      x: median(targetHistory.map(q => q[i].x)),
      y: median(targetHistory.map(q => q[i].y)),
      z: 0
    }));
  }

  // Hard outlier limiter. A single bad MediaPipe frame cannot teleport a corner.
  if (targetQuad) {
    const diagonal = Math.hypot(canvas.width, canvas.height);
    const maxJump = Math.max(65, diagonal * 0.11);
    filtered = filtered.map((p, i) => {
      const prev = targetQuad[i];
      const d = dist(p, prev);
      if (d <= maxJump) return p;
      const ratio = maxJump / d;
      return {
        x: prev.x + (p.x - prev.x) * ratio,
        y: prev.y + (p.y - prev.y) * ratio,
        z: 0
      };
    });
  }

  targetQuad = filtered;
  lastPortalSeenAt = now;
  if (!displayQuad) displayQuad = filtered.map(p => ({ ...p }));
}

function updatePortalFromResult(result, now) {
  lastResult = result;
  const hands = (result?.landmarks || []).slice(0, 2).map(handGeometry);

  let portal = null;
  if (hands.length >= 2) {
    portal = buildTwoHandPortal(hands);
    if (portal) lastTwoHandSeenAt = now;
  }

  // Brief hysteresis keeps a momentary loss of hand #2 from changing geometry.
  if (!portal && hands.length === 1 && now - lastTwoHandSeenAt > 180) {
    portal = buildOneHandPortal(hands[0]);
  }

  if (portal) stabilizeTarget(portal.quad, portal.mode, now);
}

function inferenceLoop() {
  if (!running || !handLandmarker || video.readyState < 2) return;

  if (video.currentTime === lastVideoTime) {
    inferenceTimer = setTimeout(inferenceLoop, 8);
    return;
  }
  lastVideoTime = video.currentTime;

  const now = performance.now();
  const started = now;
  try {
    // V3: detect directly from the native camera frame.
    // No 384px intermediate canvas, no mirrored inference buffer.
    const result = handLandmarker.detectForVideo(video, now);
    inferenceCostMs = performance.now() - started;
    lastInferenceAt = performance.now();
    updatePortalFromResult(result, lastInferenceAt);
  } catch (error) {
    console.warn("Hand inference gagal", error);
  }

  const minPeriod = modelBackend === "GPU" ? 38 : 68;
  const maxPeriod = modelBackend === "GPU" ? 90 : 130;
  const targetPeriod = clamp(inferenceCostMs * 1.28 + 7, minPeriod, maxPeriod);
  inferenceTimer = setTimeout(inferenceLoop, Math.max(7, targetPeriod - inferenceCostMs));
}

function smoothDisplay(target, dt) {
  if (!target) return;
  if (!displayQuad) {
    displayQuad = target.map(p => ({ ...p }));
    return;
  }

  displayQuad = target.map((p, i) => {
    const prev = displayQuad[i] || p;
    const d = dist(prev, p);

    // Small motion is damped; big intentional motion catches up immediately.
    const response = 20 + Math.min(42, d * 0.20);
    const amount = 1 - Math.exp(-response * dt);
    return {
      x: prev.x + (p.x - prev.x) * amount,
      y: prev.y + (p.y - prev.y) * amount,
      z: 0
    };
  });
}

function updateFilteredFrame(now) {
  if (!targetQuad || portalAlpha < 0.03 || video.readyState < 2) return;

  const current = FILTERS[filterIndex];
  const interval = current.type === "thermal" ? 88 : 48;
  if (now - filterTimerAt < interval) return;
  filterTimerAt = now;

  filterCtx.save();
  filterCtx.setTransform(1, 0, 0, 1, 0, 0);
  filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);
  filterCtx.filter = current.type === "css" ? current.css : "none";
  if (facingMode === "user") {
    filterCtx.translate(filterCanvas.width, 0);
    filterCtx.scale(-1, 1);
  }
  filterCtx.drawImage(video, 0, 0, filterCanvas.width, filterCanvas.height);
  filterCtx.restore();

  if (current.type === "thermal") {
    const image = filterCtx.getImageData(0, 0, filterCanvas.width, filterCanvas.height);
    const data = image.data;
    for (let i = 0; i < data.length; i += 4) {
      const lum = (data[i] * 54 + data[i + 1] * 183 + data[i + 2] * 19) >> 8;
      const lut = lum * 3;
      data[i] = THERMAL_LUT[lut];
      data[i + 1] = THERMAL_LUT[lut + 1];
      data[i + 2] = THERMAL_LUT[lut + 2];
    }
    filterCtx.putImageData(image, 0, 0);
  }
}

function pathQuad(quad) {
  ctx.beginPath();
  ctx.moveTo(quad[0].x, quad[0].y);
  for (let i = 1; i < 4; i++) ctx.lineTo(quad[i].x, quad[i].y);
  ctx.closePath();
}

function quadCenter(quad) {
  return {
    x: quad.reduce((s, p) => s + p.x, 0) / 4,
    y: quad.reduce((s, p) => s + p.y, 0) / 4
  };
}

function drawPortal(quad, alpha, now) {
  if (!quad || polygonArea(quad) < 1000 || alpha <= 0.01) return;

  ctx.save();
  ctx.globalAlpha = clamp(alpha, 0, 1);
  pathQuad(quad);
  ctx.clip();

  const t = coverTransform(filterCanvas.width, filterCanvas.height, canvas.width, canvas.height);
  ctx.drawImage(filterCanvas, t.x, t.y, t.drawW, t.drawH);

  const ys = quad.map(p => p.y);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scanY = minY + ((now * 0.00035) % 1) * Math.max(1, maxY - minY);
  const grad = ctx.createLinearGradient(0, scanY - 14, 0, scanY + 14);
  grad.addColorStop(0, "rgba(110,235,255,0)");
  grad.addColorStop(0.5, "rgba(230,252,255,.18)");
  grad.addColorStop(1, "rgba(110,235,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, scanY - 18, canvas.width, 36);
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = alpha * 0.22;
  ctx.lineWidth = 8;
  ctx.strokeStyle = "rgba(70,220,255,.9)";
  ctx.shadowBlur = 22;
  ctx.shadowColor = "rgba(60,220,255,.95)";
  pathQuad(quad);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = alpha * 0.95;
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = "rgba(226,253,255,.98)";
  pathQuad(quad);
  ctx.stroke();

  ctx.globalAlpha = alpha * 0.55;
  ctx.setLineDash([10, 15]);
  ctx.lineDashOffset = -(now * 0.035) % 25;
  ctx.strokeStyle = "rgba(93,233,255,.98)";
  ctx.lineWidth = 2.2;
  pathQuad(quad);
  ctx.stroke();
  ctx.restore();

  drawCornerNodes(quad, alpha, now);
  drawParticles(quad, alpha, now);
}

function drawCornerNodes(quad, alpha, now) {
  const pulse = 0.5 + 0.5 * Math.sin(now * 0.008);
  ctx.save();
  for (const p of quad) {
    ctx.globalAlpha = alpha * (0.72 + pulse * 0.2);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.4 + pulse, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(232,254,255,.98)";
    ctx.fill();

    ctx.globalAlpha = alpha * 0.15;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 10 + pulse * 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(75,225,255,.95)";
    ctx.fill();
  }
  ctx.restore();
}

function drawParticles(quad, alpha, now) {
  if (alpha > 0.5 && now - lastParticleSpawn > 55 && particles.length < 24) {
    lastParticleSpawn = now;
    const edge = Math.floor(Math.random() * 4);
    const a = quad[edge];
    const b = quad[(edge + 1) % 4];
    const t = Math.random();
    const x = a.x + (b.x - a.x) * t;
    const y = a.y + (b.y - a.y) * t;
    const c = quadCenter(quad);

    let n = normalize({ x: x - c.x, y: y - c.y });
    const speed = 16 + Math.random() * 32;
    particles.push({
      x, y,
      vx: n.x * speed + (Math.random() - 0.5) * 12,
      vy: n.y * speed + (Math.random() - 0.5) * 12,
      born: now,
      life: 300 + Math.random() * 360,
      size: 0.8 + Math.random() * 1.4
    });
  }

  const dt = Math.min(0.05, Math.max(0.001, (now - lastRenderAt) / 1000));
  const alive = [];
  ctx.save();
  for (const p of particles) {
    const age = now - p.born;
    if (age >= p.life) continue;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    const fade = 1 - age / p.life;
    ctx.globalAlpha = alpha * fade * 0.7;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(157,244,255,.98)";
    ctx.fill();
    alive.push(p);
  }
  ctx.restore();
  particles = alive;
}

function drawDebug(result) {
  if (!debug || !result?.landmarks?.length) return;
  ctx.save();
  ctx.lineWidth = 1.3;
  ctx.strokeStyle = "rgba(255,255,255,.72)";
  ctx.fillStyle = "rgba(88,236,255,.98)";

  for (const landmarks of result.landmarks) {
    const pts = landmarks.map(landmarkToScreen);
    for (const [a, b] of handConnections) {
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    }
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function updateHud(now) {
  const handCount = lastResult?.landmarks?.length || 0;
  const fresh = now - lastPortalSeenAt < 230;

  if (fresh && portalMode === "two") {
    setStatus("PORTAL · 4 TITIK REAL", "ready");
    hint.classList.remove("visible");
  } else if (fresh && portalMode === "one") {
    setStatus("PORTAL · PALM LOCK", "ready");
    hint.classList.remove("visible");
  } else if (handCount === 1) {
    setStatus("1 TANGAN", "warn");
    hint.innerHTML = "Hadapkan <b>telapak</b> ke kamera";
    hint.classList.add("visible");
  } else if (handCount >= 2) {
    setStatus("2 TANGAN", "warn");
    hint.innerHTML = "Buka <b>jempol + telunjuk</b> di kedua tangan";
    hint.classList.add("visible");
  } else {
    setStatus("CARI TANGAN", "warn");
    hint.innerHTML = "Tunjukkan <b>satu atau dua tangan</b>";
    hint.classList.add("visible");
  }
}

function updateFps(now) {
  fpsFrames++;
  const elapsed = now - fpsStarted;
  if (elapsed >= 650) {
    fpsEl.textContent = String(Math.round(fpsFrames * 1000 / elapsed));
    fpsFrames = 0;
    fpsStarted = now;
  }
}

function renderLoop(now) {
  if (!running) return;

  const dt = clamp((now - lastRenderAt) / 1000, 1 / 240, 0.05);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const fresh = now - lastPortalSeenAt < 230;
  if (targetQuad) smoothDisplay(targetQuad, dt);

  const desiredAlpha = fresh ? 1 : 0;
  const response = fresh ? 12 : 7;
  portalAlpha += (desiredAlpha - portalAlpha) * (1 - Math.exp(-response * dt));
  if (!fresh && portalAlpha < 0.01) portalAlpha = 0;

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

window.addEventListener("resize", () => {
  if (running) resizeCanvases();
});

window.addEventListener("orientationchange", () => {
  setTimeout(() => running && resizeCanvases(), 180);
});

document.addEventListener("visibilitychange", () => {
  if (!running) return;
  if (document.hidden) {
    if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
    if (inferenceTimer !== null) clearTimeout(inferenceTimer);
    animationFrameId = null;
    inferenceTimer = null;
  } else {
    lastRenderAt = performance.now();
    animationFrameId = requestAnimationFrame(renderLoop);
    inferenceTimer = setTimeout(inferenceLoop, 20);
  }
});

initializeModel().catch(error => {
  console.error(error);
  bootStatus.textContent = "Model gagal dimuat. Cek koneksi internet.";
  setStatus("Model gagal", "error");
});
