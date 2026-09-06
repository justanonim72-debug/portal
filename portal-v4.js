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
const recBtn = document.querySelector("#recBtn");
const filterName = document.querySelector("#filterName");
const bootStatus = document.querySelector("#bootStatus");
const statusText = document.querySelector("#statusText");
const statusDot = document.querySelector("#statusDot");
const fpsEl = document.querySelector("#fps");
const hint = document.querySelector("#hint");
const toast = document.querySelector("#toast");
const recBadge = document.querySelector("#recBadge");
const recTime = document.querySelector("#recTime");

const filterCanvas = document.createElement("canvas");
const filterCtx = filterCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
const recordCanvas = document.createElement("canvas");
const recordCtx = recordCanvas.getContext("2d", { alpha: false, desynchronized: true });

const PRESETS = [
  { name: "THERMAL HOLO", type: "thermal", edgeA: "rgba(78,226,255,.96)", edgeB: "rgba(255,103,231,.74)", glow: "rgba(73,219,255,.96)", dash: "rgba(121,238,255,.96)", grid: true, scan: true, glitch: false, chroma: true, particles: true },
  { name: "XRAY GLITCH", type: "css", css: "grayscale(1) invert(.96) contrast(1.52) brightness(1.08)", edgeA: "rgba(214,244,255,.98)", edgeB: "rgba(95,180,255,.72)", glow: "rgba(170,218,255,.86)", dash: "rgba(208,244,255,.96)", grid: false, scan: true, glitch: true, chroma: true, particles: true },
  { name: "VOID NEON", type: "css", css: "invert(.84) hue-rotate(188deg) saturate(2.7) contrast(1.34)", edgeA: "rgba(163,94,255,.97)", edgeB: "rgba(56,234,255,.82)", glow: "rgba(104,96,255,.96)", dash: "rgba(113,227,255,.94)", grid: false, scan: true, glitch: false, chroma: true, particles: true },
  { name: "HOLOGRAM+", type: "css", css: "grayscale(.08) sepia(.78) saturate(6.2) hue-rotate(126deg) contrast(1.22) brightness(1.07)", edgeA: "rgba(76,239,255,.98)", edgeB: "rgba(255,118,223,.7)", glow: "rgba(67,225,255,.96)", dash: "rgba(138,247,255,.96)", grid: true, scan: true, glitch: true, chroma: false, particles: true },
  { name: "RAW SHIELD", type: "css", css: "none", edgeA: "rgba(118,239,255,.98)", edgeB: "rgba(255,255,255,.7)", glow: "rgba(92,235,255,.9)", dash: "rgba(144,246,255,.96)", grid: false, scan: false, glitch: false, chroma: false, particles: true }
];

const handConnections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

let handLandmarker = null;
let modelReady = false;
let modelBackend = "GPU";
let stream = null;
let facingMode = "user";
let running = false;
let presetIndex = 0;
let debug = false;
let animationFrameId = null;
let inferenceTimer = null;
let lastVideoTime = -1;
let lastRenderAt = performance.now();
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
let particles = [];
let lastParticleSpawn = 0;
let mediaRecorder = null;
let recordStream = null;
let recordedChunks = [];
let recording = false;
let recordingStartedAt = 0;
let recordMimeType = "";

startBtn.disabled = true;
app.dataset.facing = facingMode;

const THERMAL_LUT = new Uint8Array(256 * 3);
for (let v = 0; v < 256; v++) {
  const x = v / 255;
  let r, g, b;
  if (x < .17) { r = 7; g = 5 + 26 * (x / .17); b = 55 + 170 * (x / .17); }
  else if (x < .36) { r = 0; g = 35 + 220 * ((x - .17) / .19); b = 255; }
  else if (x < .56) { r = 255 * ((x - .36) / .2); g = 255; b = 255 * (1 - (x - .36) / .2); }
  else if (x < .79) { r = 255; g = 255 * (1 - (x - .56) / .23); b = 0; }
  else { const t = (x - .79) / .21; r = 255; g = 176 + 79 * t; b = 50 + 205 * t; }
  const i = v * 3;
  THERMAL_LUT[i] = r | 0; THERMAL_LUT[i + 1] = g | 0; THERMAL_LUT[i + 2] = b | 0;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function midpoint(a, b) { return { x: (a.x + b.x) * .5, y: (a.y + b.y) * .5, z: 0 }; }
function add(a, b) { return { x: a.x + b.x, y: a.y + b.y, z: 0 }; }
function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: 0 }; }
function mul(a, s) { return { x: a.x * s, y: a.y * s, z: 0 }; }
function dot(a, b) { return a.x * b.x + a.y * b.y; }
function normalize(v, fallback = { x: 1, y: 0 }) { const n = Math.hypot(v.x, v.y); return n > 1e-5 ? { x: v.x / n, y: v.y / n, z: 0 } : { ...fallback, z: 0 }; }
function avg(points) { return { x: points.reduce((s, p) => s + p.x, 0) / points.length, y: points.reduce((s, p) => s + p.y, 0) / points.length, z: 0 }; }
function lerpPoint(a, b, t) { return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: 0 }; }

function setStatus(text, state = "warn") { statusText.textContent = text; statusDot.className = state; }
function showToast(message) { toast.textContent = message; toast.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove("show"), 1900); }

async function initializeModel() {
  bootStatus.textContent = "Memuat MediaPipe Hand Landmarker…";
  setStatus("Memuat model", "warn");
  const vision = await FilesetResolver.forVisionTasks(MP_ROOT);
  const common = { runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: .52, minHandPresenceConfidence: .52, minTrackingConfidence: .58 };
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, { ...common, baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" } });
    modelBackend = "GPU"; bootStatus.textContent = "Tracking siap · GPU · Portal V4";
  } catch (error) {
    console.warn("GPU delegate gagal, fallback CPU", error);
    handLandmarker = await HandLandmarker.createFromOptions(vision, { ...common, baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" } });
    modelBackend = "CPU"; bootStatus.textContent = "Tracking siap · CPU fallback · Portal V4";
  }
  modelReady = true; setStatus("Siap", "ready"); startBtn.disabled = false;
}

async function openCamera() {
  if (!modelReady) return showToast("Model masih dimuat.");
  if (recording) stopRecording();
  stopLoops();
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 24, max: 60 } } });
    video.srcObject = stream;
    if (!video.videoWidth || !video.videoHeight) await new Promise(resolve => video.addEventListener("loadedmetadata", resolve, { once: true }));
    await video.play();
    app.dataset.facing = facingMode; resizeCanvases(); resetTrackingState(); running = true; app.dataset.state = "live";
    setStatus("CARI TANGAN", "warn"); hint.innerHTML = "Buka <b>jempol + telunjuk</b>"; hint.classList.add("visible");
    animationFrameId = requestAnimationFrame(renderLoop); inferenceTimer = setTimeout(inferenceLoop, 18);
  } catch (error) {
    console.error(error); const insecure = !window.isSecureContext; bootStatus.textContent = insecure ? "Kamera butuh HTTPS." : "Izin kamera gagal."; setStatus("Kamera gagal", "error"); showToast(insecure ? "Buka Portal lewat HTTPS." : "Izinkan kamera lalu coba lagi.");
  }
}

function stopLoops() { running = false; if (animationFrameId !== null) cancelAnimationFrame(animationFrameId); if (inferenceTimer !== null) clearTimeout(inferenceTimer); animationFrameId = null; inferenceTimer = null; }
function resetTrackingState() { lastVideoTime = -1; lastResult = null; lastPortalSeenAt = 0; lastTwoHandSeenAt = -Infinity; portalMode = null; targetQuad = null; displayQuad = null; targetHistory = []; portalAlpha = 0; particles = []; filterTimerAt = 0; lastRenderAt = performance.now(); fpsFrames = 0; fpsStarted = performance.now(); }

function resizeCanvases() {
  const w = Math.max(1, window.innerWidth), h = Math.max(1, window.innerHeight);
  canvas.width = Math.round(w); canvas.height = Math.round(h); canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  recordCanvas.width = canvas.width; recordCanvas.height = canvas.height;
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720, maxSide = 520, s = Math.min(1, maxSide / Math.max(vw, vh));
  filterCanvas.width = Math.max(1, Math.round(vw * s)); filterCanvas.height = Math.max(1, Math.round(vh * s));
}

function coverTransform(srcW, srcH, dstW, dstH) { const scale = Math.max(dstW / srcW, dstH / srcH), drawW = srcW * scale, drawH = srcH * scale; return { scale, x: (dstW - drawW) * .5, y: (dstH - drawH) * .5, drawW, drawH }; }
function landmarkToScreen(lm) { const vw = video.videoWidth || 1, vh = video.videoHeight || 1, t = coverTransform(vw, vh, canvas.width, canvas.height), nx = facingMode === "user" ? 1 - lm.x : lm.x; return { x: t.x + nx * vw * t.scale, y: t.y + lm.y * vh * t.scale, z: lm.z || 0 }; }

function handGeometry(landmarks) {
  const p = i => landmarkToScreen(landmarks[i]);
  const wrist = p(0), thumbMcp = p(2), thumbIp = p(3), thumbTip = p(4), indexMcp = p(5), indexPip = p(6), indexDip = p(7), indexTip = p(8), middleMcp = p(9), middleTip = p(12), ringMcp = p(13), ringTip = p(16), pinkyMcp = p(17), pinkyTip = p(20);
  const center = avg([wrist, indexMcp, middleMcp, ringMcp, pinkyMcp]);
  return { landmarks, wrist, thumbMcp, thumbIp, thumbTip, indexMcp, indexPip, indexDip, indexTip, middleMcp, middleTip, ringMcp, ringTip, pinkyMcp, pinkyTip, center, centerX: center.x, handScale: Math.max(24, dist(indexMcp, pinkyMcp) + dist(wrist, middleMcp) * .55) };
}

function polygonArea(points) { let area = 0; for (let i = 0; i < points.length; i++) { const j = (i + 1) % points.length; area += points[i].x * points[j].y - points[j].x * points[i].y; } return Math.abs(area * .5); }
function angleOrder(points) { const c = avg(points); return [...points].sort((a, b) => Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x)); }
function allQuadOrders(ordered) { const variants = []; for (const base of [ordered, [...ordered].reverse()]) for (let k = 0; k < 4; k++) variants.push([base[k], base[(k + 1) % 4], base[(k + 2) % 4], base[(k + 3) % 4]]); return variants; }
function alignQuad(points, reference) { const ordered = angleOrder(points); if (!reference) return ordered.map(p => ({ ...p })); let best = null, scoreBest = Infinity; for (const q of allQuadOrders(ordered)) { let score = 0; for (let i = 0; i < 4; i++) score += dist(q[i], reference[i]); if (score < scoreBest) { scoreBest = score; best = q; } } return best.map(p => ({ ...p })); }

function buildTwoHandPortal(hands) {
  if (hands.length < 2) return null;
  const sorted = [...hands].sort((a, b) => a.centerX - b.centerX), left = sorted[0], right = sorted[1];
  if (dist(left.center, right.center) < 70 || dist(left.thumbTip, left.indexTip) < 22 || dist(right.thumbTip, right.indexTip) < 22) return null;
  const quad = alignQuad([left.thumbTip, left.indexTip, right.thumbTip, right.indexTip], portalMode === "two" ? targetQuad : null);
  if (polygonArea(quad) < 1400) return null;
  return { mode: "two", quad };
}

function buildOneHandPortal(hand) {
  if (!hand) return null;
  const p0 = { ...hand.thumbTip }, p1 = { ...hand.indexTip }, baseLen = dist(p0, p1);
  if (baseLen < 28) return null;
  const baseCenter = midpoint(p0, p1), baseAxis = normalize(sub(p1, p0));
  const primaryRoot = midpoint(hand.thumbIp, hand.indexPip);
  let primaryForward = normalize(sub(baseCenter, primaryRoot), { x: -baseAxis.y, y: baseAxis.x });
  const secondaryTips = avg([hand.middleTip, hand.ringTip, hand.pinkyTip]), secondaryRoots = avg([hand.middleMcp, hand.ringMcp, hand.pinkyMcp]);
  let secondaryForward = normalize(sub(secondaryTips, secondaryRoots), primaryForward);
  if (dot(primaryForward, secondaryForward) < 0) secondaryForward = mul(secondaryForward, -1);
  let forward = normalize(add(mul(primaryForward, .78), mul(secondaryForward, .22)), primaryForward);
  const parallel = dot(forward, baseAxis);
  forward = normalize(sub(forward, mul(baseAxis, parallel * .72)), { x: -baseAxis.y, y: baseAxis.x });
  if (dot(forward, primaryForward) < 0) forward = mul(forward, -1);
  const fingerReach = (dist(hand.middleMcp, hand.middleTip) + dist(hand.ringMcp, hand.ringTip) + dist(hand.pinkyMcp, hand.pinkyTip)) / 3;
  const depth = clamp(baseLen * 1.18 + fingerReach * .42, 76, Math.min(canvas.width, canvas.height) * .38);
  const naturalFar = add(baseCenter, mul(forward, depth)), supportBiasRaw = sub(secondaryTips, naturalFar), supportBiasLen = Math.hypot(supportBiasRaw.x, supportBiasRaw.y);
  const supportBias = supportBiasLen > 1e-5 ? mul(normalize(supportBiasRaw), Math.min(depth * .20, supportBiasLen * .20)) : { x: 0, y: 0, z: 0 };
  const farCenter = add(naturalFar, supportBias);
  let secondaryAxis = normalize(sub(hand.pinkyTip, hand.middleTip), baseAxis);
  if (dot(secondaryAxis, baseAxis) < 0) secondaryAxis = mul(secondaryAxis, -1);
  const farAxis = normalize(add(mul(baseAxis, .84), mul(secondaryAxis, .16)), baseAxis), spread = dist(hand.middleTip, hand.pinkyTip), farWidthRatio = clamp(.86 + (spread / Math.max(1, baseLen)) * .12, .82, 1.18), farHalf = baseLen * .5 * farWidthRatio;
  const p2 = add(farCenter, mul(farAxis, farHalf)), p3 = add(farCenter, mul(farAxis, -farHalf)), quad = [p0, p1, p2, p3];
  if (polygonArea(quad) < 1200) return null;
  return { mode: "one", quad };
}

function median(values) { const arr = [...values].sort((a, b) => a - b); return arr[Math.floor(arr.length / 2)]; }
function stabilizeTarget(rawQuad, mode, now) {
  let aligned = rawQuad.map(p => ({ ...p }));
  if (targetQuad && portalMode === mode && mode === "two") aligned = alignQuad(aligned, targetQuad);
  if (portalMode !== mode) targetHistory = [];
  portalMode = mode; targetHistory.push(aligned); if (targetHistory.length > 3) targetHistory.shift();
  let filtered = aligned;
  if (targetHistory.length >= 3) filtered = aligned.map((_, i) => ({ x: median(targetHistory.map(q => q[i].x)), y: median(targetHistory.map(q => q[i].y)), z: 0 }));
  if (targetQuad) {
    const maxBase = mode === "one" ? 54 : 76;
    filtered = filtered.map((p, i) => { const prev = targetQuad[i], d = dist(prev, p), maxJump = i < 2 && mode === "one" ? maxBase * 1.3 : maxBase; return d <= maxJump ? p : lerpPoint(prev, p, maxJump / d); });
  }
  targetQuad = filtered; lastPortalSeenAt = now; if (!displayQuad) displayQuad = filtered.map(p => ({ ...p }));
}

function updatePortalFromResult(result, now) {
  lastResult = result; const hands = (result?.landmarks || []).slice(0, 2).map(handGeometry); let portal = null;
  if (hands.length >= 2) { portal = buildTwoHandPortal(hands); if (portal) lastTwoHandSeenAt = now; }
  if (!portal && hands.length === 1 && now - lastTwoHandSeenAt > 160) portal = buildOneHandPortal(hands[0]);
  if (portal) stabilizeTarget(portal.quad, portal.mode, now);
}

function inferenceLoop() {
  if (!running || !handLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) { inferenceTimer = setTimeout(inferenceLoop, 7); return; }
  lastVideoTime = video.currentTime;
  const now = performance.now();
  try { const started = performance.now(); const result = handLandmarker.detectForVideo(video, now); inferenceCostMs = performance.now() - started; updatePortalFromResult(result, performance.now()); }
  catch (error) { console.warn("Hand inference gagal", error); }
  const minPeriod = modelBackend === "GPU" ? 34 : 60, maxPeriod = modelBackend === "GPU" ? 86 : 126, targetPeriod = clamp(inferenceCostMs * 1.20 + 6, minPeriod, maxPeriod);
  inferenceTimer = setTimeout(inferenceLoop, Math.max(6, targetPeriod - inferenceCostMs));
}

function smoothDisplay(target, dt) {
  if (!target) return;
  if (!displayQuad) { displayQuad = target.map(p => ({ ...p })); return; }
  displayQuad = target.map((p, i) => { const prev = displayQuad[i] || p, d = dist(prev, p), response = d < 12 ? 11 : d < 34 ? 18 : 31, amount = 1 - Math.exp(-response * dt); return lerpPoint(prev, p, amount); });
}

function updateFilteredFrame(now) {
  if (!targetQuad || portalAlpha < .025 || video.readyState < 2) return;
  const preset = PRESETS[presetIndex], interval = preset.type === "thermal" ? 78 : 44;
  if (now - filterTimerAt < interval) return;
  filterTimerAt = now; filterCtx.save(); filterCtx.setTransform(1, 0, 0, 1, 0, 0); filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height); filterCtx.filter = preset.type === "css" ? preset.css : "none";
  if (facingMode === "user") { filterCtx.translate(filterCanvas.width, 0); filterCtx.scale(-1, 1); }
  filterCtx.drawImage(video, 0, 0, filterCanvas.width, filterCanvas.height); filterCtx.restore();
  if (preset.type === "thermal") {
    const image = filterCtx.getImageData(0, 0, filterCanvas.width, filterCanvas.height), data = image.data;
    for (let i = 0; i < data.length; i += 4) { const lum = (data[i] * 54 + data[i + 1] * 183 + data[i + 2] * 19) >> 8, lut = lum * 3; data[i] = THERMAL_LUT[lut]; data[i + 1] = THERMAL_LUT[lut + 1]; data[i + 2] = THERMAL_LUT[lut + 2]; }
    filterCtx.putImageData(image, 0, 0);
  }
}

function pathQuad(quad) { ctx.beginPath(); ctx.moveTo(quad[0].x, quad[0].y); for (let i = 1; i < 4; i++) ctx.lineTo(quad[i].x, quad[i].y); ctx.closePath(); }
function quadCenter(quad) { return avg(quad); }
function drawGrid(quad, alpha, preset) {
  if (!preset.grid) return; const [a,b,c,d] = quad; ctx.save(); ctx.globalAlpha = alpha * .14; ctx.strokeStyle = preset.edgeA; ctx.lineWidth = .8;
  for (let i = 1; i < 7; i++) { const t = i / 7, p = lerpPoint(a, d, t), q = lerpPoint(b, c, t); ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
  for (let i = 1; i < 7; i++) { const t = i / 7, p = lerpPoint(a, b, t), q = lerpPoint(d, c, t); ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
  ctx.restore();
}
function drawGlitch(quad, alpha, now, preset) { if (!preset.glitch || ((now / 95) | 0) % 7 > 1) return; ctx.save(); ctx.globalAlpha = alpha * .13; ctx.translate(3.4, -1.2); pathQuad(quad); ctx.clip(); const t = coverTransform(filterCanvas.width, filterCanvas.height, canvas.width, canvas.height); ctx.drawImage(filterCanvas, t.x, t.y, t.drawW, t.drawH); ctx.restore(); }
function drawParticles(quad, alpha, now, preset) {
  if (preset.particles && alpha > .5 && now - lastParticleSpawn > 48 && particles.length < 30) {
    lastParticleSpawn = now; const edge = Math.floor(Math.random() * 4), a = quad[edge], b = quad[(edge + 1) % 4], t = Math.random(), x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t, c = quadCenter(quad), n = normalize({ x: x - c.x, y: y - c.y }), speed = 18 + Math.random() * 34;
    particles.push({ x, y, vx: n.x * speed + (Math.random() - .5) * 14, vy: n.y * speed + (Math.random() - .5) * 14, born: now, life: 320 + Math.random() * 380, size: .8 + Math.random() * 1.6 });
  }
  const dt = Math.min(.05, Math.max(.001, (now - lastRenderAt) / 1000)), alive = []; ctx.save();
  for (const p of particles) { const age = now - p.born; if (age >= p.life) continue; p.x += p.vx * dt; p.y += p.vy * dt; const fade = 1 - age / p.life; ctx.globalAlpha = alpha * fade * .72; ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fillStyle = preset.edgeA; ctx.fill(); alive.push(p); }
  ctx.restore(); particles = alive;
}

function drawPortal(quad, alpha, now) {
  if (!quad || polygonArea(quad) < 1000 || alpha <= .01) return;
  const preset = PRESETS[presetIndex];
  ctx.save(); ctx.globalAlpha = clamp(alpha, 0, 1); pathQuad(quad); ctx.clip(); const t = coverTransform(filterCanvas.width, filterCanvas.height, canvas.width, canvas.height); ctx.drawImage(filterCanvas, t.x, t.y, t.drawW, t.drawH);
  if (preset.scan) { const ys = quad.map(p => p.y), minY = Math.min(...ys), maxY = Math.max(...ys), scanY = minY + ((now * .00036) % 1) * Math.max(1, maxY - minY), grad = ctx.createLinearGradient(0, scanY - 16, 0, scanY + 16); grad.addColorStop(0, "rgba(120,240,255,0)"); grad.addColorStop(.5, "rgba(235,253,255,.2)"); grad.addColorStop(1, "rgba(120,240,255,0)"); ctx.fillStyle = grad; ctx.fillRect(0, scanY - 20, canvas.width, 40); }
  ctx.restore();
  drawGrid(quad, alpha, preset); drawGlitch(quad, alpha, now, preset);
  ctx.save(); ctx.globalAlpha = alpha * .25; ctx.lineWidth = 8; ctx.strokeStyle = preset.edgeA; ctx.shadowBlur = 24; ctx.shadowColor = preset.glow; pathQuad(quad); ctx.stroke(); ctx.restore();
  if (preset.chroma) { ctx.save(); ctx.globalAlpha = alpha * .28; ctx.lineWidth = 2; ctx.strokeStyle = preset.edgeB; ctx.translate(-1.5, .7); pathQuad(quad); ctx.stroke(); ctx.restore(); }
  ctx.save(); ctx.globalAlpha = alpha * .96; ctx.lineWidth = 1.4; ctx.strokeStyle = "rgba(229,253,255,.98)"; pathQuad(quad); ctx.stroke(); ctx.globalAlpha = alpha * .58; ctx.setLineDash([10,15]); ctx.lineDashOffset = -(now * .036) % 25; ctx.strokeStyle = preset.dash; ctx.lineWidth = 2.2; pathQuad(quad); ctx.stroke(); ctx.restore();
  const pulse = .5 + .5 * Math.sin(now * .008); ctx.save();
  for (let i = 0; i < quad.length; i++) { const p = quad[i]; ctx.globalAlpha = alpha * (.76 + pulse * .18); ctx.beginPath(); ctx.arc(p.x, p.y, 3.5 + pulse, 0, Math.PI * 2); ctx.fillStyle = i < 2 && portalMode === "one" ? "rgba(255,255,255,1)" : preset.edgeA; ctx.fill(); ctx.globalAlpha = alpha * .15; ctx.beginPath(); ctx.arc(p.x, p.y, 10 + pulse * 3, 0, Math.PI * 2); ctx.fillStyle = preset.glow; ctx.fill(); }
  ctx.restore(); drawParticles(quad, alpha, now, preset);
}

function drawDebug(result) {
  if (!debug || !result?.landmarks?.length) return; ctx.save(); ctx.lineWidth = 1.25; ctx.strokeStyle = "rgba(255,255,255,.7)";
  for (const landmarks of result.landmarks) { const pts = landmarks.map(landmarkToScreen); for (const [a,b] of handConnections) { ctx.beginPath(); ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke(); } for (let i = 0; i < pts.length; i++) { const p = pts[i], primary = i === 4 || i === 8; ctx.fillStyle = primary ? "rgba(255,245,125,1)" : "rgba(86,237,255,.98)"; ctx.beginPath(); ctx.arc(p.x, p.y, primary ? 4.2 : 2.5, 0, Math.PI * 2); ctx.fill(); } }
  ctx.restore();
}

function updateHud(now) {
  const handCount = lastResult?.landmarks?.length || 0, fresh = now - lastPortalSeenAt < 245;
  if (fresh && portalMode === "two") { setStatus("PORTAL · 2 TANGAN", "ready"); hint.classList.remove("visible"); }
  else if (fresh && portalMode === "one") { setStatus("PORTAL · PINCH LOCK", "ready"); hint.classList.remove("visible"); }
  else if (handCount === 1) { setStatus("1 TANGAN", "warn"); hint.innerHTML = "Buka <b>jempol + telunjuk</b> — jari lain bantu arah"; hint.classList.add("visible"); }
  else if (handCount >= 2) { setStatus("2 TANGAN", "warn"); hint.innerHTML = "Buka <b>jempol + telunjuk</b> di kedua tangan"; hint.classList.add("visible"); }
  else { setStatus("CARI TANGAN", "warn"); hint.innerHTML = "Tunjukkan <b>satu atau dua tangan</b>"; hint.classList.add("visible"); }
}
function updateFps(now) { fpsFrames++; const elapsed = now - fpsStarted; if (elapsed >= 650) { fpsEl.textContent = String(Math.round(fpsFrames * 1000 / elapsed)); fpsFrames = 0; fpsStarted = now; } }

function drawVideoCover(targetCtx, width, height) {
  const vw = video.videoWidth || 1, vh = video.videoHeight || 1, t = coverTransform(vw, vh, width, height); targetCtx.save(); targetCtx.setTransform(1, 0, 0, 1, 0, 0); targetCtx.fillStyle = "#020305"; targetCtx.fillRect(0, 0, width, height);
  if (facingMode === "user") { targetCtx.translate(width, 0); targetCtx.scale(-1, 1); const mirroredX = width - t.x - t.drawW; targetCtx.drawImage(video, mirroredX, t.y, t.drawW, t.drawH); }
  else targetCtx.drawImage(video, t.x, t.y, t.drawW, t.drawH);
  targetCtx.restore();
}
function drawRecordFrame(now) { if (!recording) return; drawVideoCover(recordCtx, recordCanvas.width, recordCanvas.height); recordCtx.drawImage(canvas, 0, 0, recordCanvas.width, recordCanvas.height); const seconds = Math.floor((now - recordingStartedAt) / 1000); recTime.textContent = `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`; }
function pickRecordingMime() { if (!window.MediaRecorder) return ""; for (const type of ["video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"]) if (MediaRecorder.isTypeSupported(type)) return type; return ""; }
function startRecording() {
  if (recording) return; if (!window.MediaRecorder || !recordCanvas.captureStream) return showToast("Browser ini belum dukung rekam.");
  try {
    recordCanvas.width = canvas.width; recordCanvas.height = canvas.height; drawVideoCover(recordCtx, recordCanvas.width, recordCanvas.height); recordCtx.drawImage(canvas, 0, 0); recordMimeType = pickRecordingMime(); recordStream = recordCanvas.captureStream(30); recordedChunks = [];
    mediaRecorder = recordMimeType ? new MediaRecorder(recordStream, { mimeType: recordMimeType, videoBitsPerSecond: 6_000_000 }) : new MediaRecorder(recordStream, { videoBitsPerSecond: 6_000_000 });
    mediaRecorder.ondataavailable = e => { if (e.data?.size) recordedChunks.push(e.data); }; mediaRecorder.onstop = saveRecording; mediaRecorder.start(250); recording = true; recordingStartedAt = performance.now(); recBtn.classList.add("active"); recBtn.textContent = "■"; recBadge.classList.add("visible"); recTime.textContent = "00:00"; showToast("Rekam dimulai");
  } catch (error) { console.error(error); showToast("Gagal mulai rekam."); }
}
function stopRecording() { if (!recording) return; recording = false; recBtn.classList.remove("active"); recBtn.textContent = "●"; recBadge.classList.remove("visible"); if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop(); }
function saveRecording() {
  try { const blob = new Blob(recordedChunks, { type: recordMimeType || "video/webm" }); recordedChunks = []; if (!blob.size) return showToast("Rekaman kosong."); const url = URL.createObjectURL(blob), a = document.createElement("a"); a.href = url; a.download = `portal-${new Date().toISOString().replace(/[:.]/g,"-")}.webm`; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000); showToast("Rekaman disimpan ✓"); }
  finally { if (recordStream) recordStream.getTracks().forEach(track => track.stop()); recordStream = null; mediaRecorder = null; }
}

function renderLoop(now) {
  if (!running) return; const dt = clamp((now - lastRenderAt) / 1000, 1 / 240, .05); ctx.clearRect(0, 0, canvas.width, canvas.height); const fresh = now - lastPortalSeenAt < 245; if (targetQuad) smoothDisplay(targetQuad, dt); const desiredAlpha = fresh ? 1 : 0, response = fresh ? 12 : 7; portalAlpha += (desiredAlpha - portalAlpha) * (1 - Math.exp(-response * dt)); if (!fresh && portalAlpha < .01) portalAlpha = 0;
  updateFilteredFrame(now); if (displayQuad && portalAlpha > 0) drawPortal(displayQuad, portalAlpha, now); drawDebug(lastResult); updateHud(now); updateFps(now); drawRecordFrame(now); lastRenderAt = now; animationFrameId = requestAnimationFrame(renderLoop);
}

startBtn.addEventListener("click", openCamera);
cameraBtn.addEventListener("click", async () => { if (recording) stopRecording(); facingMode = facingMode === "user" ? "environment" : "user"; app.dataset.facing = facingMode; showToast(facingMode === "user" ? "Kamera depan" : "Kamera belakang"); await openCamera(); });
filterBtn.addEventListener("click", () => { presetIndex = (presetIndex + 1) % PRESETS.length; filterName.textContent = PRESETS[presetIndex].name; filterTimerAt = 0; showToast(PRESETS[presetIndex].name); });
debugBtn.addEventListener("click", () => { debug = !debug; debugBtn.classList.toggle("active", debug); showToast(debug ? "Landmark ON · kuning = jempol/telunjuk" : "Landmark OFF"); });
recBtn.addEventListener("click", () => recording ? stopRecording() : startRecording());
window.addEventListener("resize", () => running && resizeCanvases());
window.addEventListener("orientationchange", () => setTimeout(() => running && resizeCanvases(), 180));
document.addEventListener("visibilitychange", () => { if (!running) return; if (document.hidden) { if (recording) stopRecording(); if (animationFrameId !== null) cancelAnimationFrame(animationFrameId); if (inferenceTimer !== null) clearTimeout(inferenceTimer); animationFrameId = null; inferenceTimer = null; } else { lastRenderAt = performance.now(); animationFrameId = requestAnimationFrame(renderLoop); inferenceTimer = setTimeout(inferenceLoop, 18); } });

initializeModel().catch(error => { console.error(error); bootStatus.textContent = "Model gagal dimuat. Cek koneksi internet."; setStatus("Model gagal", "error"); });