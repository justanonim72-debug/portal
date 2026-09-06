import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const app = document.querySelector("#app");
const video = document.querySelector("#camera");
const canvas = document.querySelector("#stage");
const ctx = canvas.getContext("2d", { alpha: false, desynchronized: true });
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

const sourceCanvas = document.createElement("canvas");
const sourceCtx = sourceCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
const filterCanvas = document.createElement("canvas");
const filterCtx = filterCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

const FILTERS = [
  { name: "THERMAL", type: "thermal" },
  { name: "MONO", type: "css", css: "grayscale(1) contrast(1.2) brightness(1.05)" },
  { name: "NEON", type: "css", css: "saturate(2.4) hue-rotate(125deg) contrast(1.18)" },
  { name: "INVERT", type: "css", css: "invert(1) contrast(1.1)" },
  { name: "RAW", type: "css", css: "none" }
];

let handLandmarker = null;
let stream = null;
let facingMode = "user";
let filterIndex = 0;
let debug = false;
let running = false;
let modelReady = false;
let lastVideoTime = -1;
let lastInferenceAt = 0;
let portalAlpha = 0;
let smoothed = null;
let toastTimer = null;
let fpsFrames = 0;
let fpsStarted = performance.now();
let measuredFps = 0;
let thermalTick = 0;
let inferenceBusy = false;

const THERMAL_LUT = new Uint8Array(256 * 3);
for (let v = 0; v < 256; v++) {
  const x = v / 255;
  let r, g, b;
  if (x < 0.2) {
    r = 0; g = 0; b = 80 + 175 * (x / 0.2);
  } else if (x < 0.4) {
    r = 0; g = 255 * ((x - 0.2) / 0.2); b = 255;
  } else if (x < 0.6) {
    r = 255 * ((x - 0.4) / 0.2); g = 255; b = 255 * (1 - (x - 0.4) / 0.2);
  } else if (x < 0.82) {
    r = 255; g = 255 * (1 - (x - 0.6) / 0.22); b = 0;
  } else {
    r = 255; g = 255 * ((x - 0.82) / 0.18); b = 255 * ((x - 0.82) / 0.18);
  }
  const i = v * 3;
  THERMAL_LUT[i] = r | 0;
  THERMAL_LUT[i + 1] = g | 0;
  THERMAL_LUT[i + 2] = b | 0;
}

const handConnections = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

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

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    bootStatus.textContent = "Hand tracking siap · GPU";
  } catch (gpuError) {
    console.warn("GPU delegate gagal, fallback CPU", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    bootStatus.textContent = "Hand tracking siap · CPU fallback";
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
      frameRate: { ideal: 30, max: 60 }
    }
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    resizeCanvases();
    running = true;
    lastVideoTime = -1;
    smoothed = null;
    portalAlpha = 0;
    app.dataset.state = "live";
    setStatus("Tracking", "ready");
    requestAnimationFrame(renderLoop);
  } catch (error) {
    console.error(error);
    const insecure = !window.isSecureContext;
    bootStatus.textContent = insecure
      ? "Kamera butuh HTTPS. Buka dari GitHub Pages/Vercel."
      : "Izin kamera gagal. Izinkan kamera lalu coba lagi.";
    setStatus("Kamera gagal", "error");
    showToast(insecure ? "Kamera browser hanya aktif lewat HTTPS." : "Izin kamera belum diberikan.");
  }
}

function resizeCanvases() {
  // Render the final stage at CSS-pixel resolution. On a phone this is much
  // cheaper than DPR=2 while still looking sharp for a camera effect.
  const cssW = window.innerWidth;
  const cssH = window.innerHeight;
  canvas.width = Math.round(cssW);
  canvas.height = Math.round(cssH);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const processingWidth = Math.min(vw, 640);
  const processingHeight = Math.max(1, Math.round(processingWidth * vh / vw));
  sourceCanvas.width = processingWidth;
  sourceCanvas.height = processingHeight;
  filterCanvas.width = processingWidth;
  filterCanvas.height = processingHeight;
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

function landmarkToScreen(lm, transform) {
  // Front camera is rendered mirrored, so mirror x for landmarks too.
  const normalizedX = facingMode === "user" ? 1 - lm.x : lm.x;
  return {
    x: transform.x + normalizedX * video.videoWidth * transform.scale,
    y: transform.y + lm.y * video.videoHeight * transform.scale,
    z: lm.z || 0
  };
}

function drawCameraFrame(width, height, transform) {
  ctx.save();
  ctx.fillStyle = "#020305";
  ctx.fillRect(0, 0, width, height);
  if (facingMode === "user") {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    const mirroredX = width - transform.x - transform.drawW;
    ctx.drawImage(video, mirroredX, transform.y, transform.drawW, transform.drawH);
  } else {
    ctx.drawImage(video, transform.x, transform.y, transform.drawW, transform.drawH);
  }
  ctx.restore();
}

function updateSourceFrame() {
  sourceCtx.save();
  sourceCtx.setTransform(1,0,0,1,0,0);
  sourceCtx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  if (facingMode === "user") {
    sourceCtx.translate(sourceCanvas.width, 0);
    sourceCtx.scale(-1, 1);
  }
  sourceCtx.drawImage(video, 0, 0, sourceCanvas.width, sourceCanvas.height);
  sourceCtx.restore();
}

function applyFilter() {
  const filter = FILTERS[filterIndex];
  filterCtx.save();
  filterCtx.setTransform(1,0,0,1,0,0);
  filterCtx.clearRect(0, 0, filterCanvas.width, filterCanvas.height);
  filterCtx.filter = filter.type === "css" ? filter.css : "none";
  filterCtx.drawImage(sourceCanvas, 0, 0);
  filterCtx.restore();

  if (filter.type === "thermal") {
    // Thermal conversion is intentionally done at a reduced cadence on mobile.
    // Between conversions the previous filtered buffer is reused.
    thermalTick++;
    if (thermalTick % 2 === 0 || thermalTick < 3) {
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
}

function smoothPoint(prev, next, amount = 0.38) {
  if (!prev) return { ...next };
  return {
    x: prev.x + (next.x - prev.x) * amount,
    y: prev.y + (next.y - prev.y) * amount,
    z: prev.z + (next.z - prev.z) * amount
  };
}

function buildPortal(result, transform) {
  if (!result?.landmarks || result.landmarks.length < 2) return null;

  const candidates = result.landmarks.slice(0, 2).map((landmarks, index) => {
    const thumb = landmarkToScreen(landmarks[4], transform);
    const indexTip = landmarkToScreen(landmarks[8], transform);
    const wrist = landmarkToScreen(landmarks[0], transform);
    const middleMcp = landmarkToScreen(landmarks[9], transform);
    return {
      landmarks,
      index,
      thumb,
      indexTip,
      centerX: (wrist.x + middleMcp.x) * 0.5
    };
  }).sort((a, b) => a.centerX - b.centerX);

  const left = candidates[0];
  const right = candidates[1];
  const handGap = right.centerX - left.centerX;
  if (handGap < 70) return null;

  // On each hand, use thumb + index tip as the vertical edge of the portal.
  // Sort by y so hand handedness/orientation cannot flip the edge.
  const leftTopRaw = left.indexTip.y < left.thumb.y ? left.indexTip : left.thumb;
  const leftBottomRaw = left.indexTip.y < left.thumb.y ? left.thumb : left.indexTip;
  const rightTopRaw = right.indexTip.y < right.thumb.y ? right.indexTip : right.thumb;
  const rightBottomRaw = right.indexTip.y < right.thumb.y ? right.thumb : right.indexTip;

  const leftEdge = Math.hypot(leftBottomRaw.x - leftTopRaw.x, leftBottomRaw.y - leftTopRaw.y);
  const rightEdge = Math.hypot(rightBottomRaw.x - rightTopRaw.x, rightBottomRaw.y - rightTopRaw.y);
  if (leftEdge < 30 || rightEdge < 30) return null;

  const raw = [leftTopRaw, rightTopRaw, rightBottomRaw, leftBottomRaw];
  if (!smoothed) smoothed = raw.map(p => ({ ...p }));
  else smoothed = raw.map((p, i) => smoothPoint(smoothed[i], p));

  return {
    quad: smoothed,
    hands: candidates
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

function drawPortal(quad, alpha, transform) {
  if (!quad || polygonArea(quad) < 1800) return;

  const [tl, tr, br, bl] = quad;
  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha));

  // The reel is a filter window, not a separate camera screen: clip the
  // filtered copy of the SAME frame to the hand-defined quadrilateral.
  // This keeps eyes/nose/background perfectly registered outside vs inside.
  ctx.beginPath();
  ctx.moveTo(tl.x, tl.y);
  ctx.lineTo(tr.x, tr.y);
  ctx.lineTo(br.x, br.y);
  ctx.lineTo(bl.x, bl.y);
  ctx.closePath();
  ctx.clip();
  ctx.drawImage(filterCanvas, transform.x, transform.y, transform.drawW, transform.drawH);

  ctx.restore();
  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
  ctx.beginPath();
  ctx.moveTo(tl.x, tl.y);
  ctx.lineTo(tr.x, tr.y);
  ctx.lineTo(br.x, br.y);
  ctx.lineTo(bl.x, bl.y);
  ctx.closePath();
  ctx.fillStyle = "rgba(214, 243, 255, 0.025)";
  ctx.fill();

  ctx.shadowColor = "rgba(165, 232, 255, .8)";
  ctx.shadowBlur = 14;
  ctx.strokeStyle = "rgba(224, 247, 255, .92)";
  ctx.lineWidth = 1.35;
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(77, 181, 255, .35)";
  ctx.lineWidth = 4;
  ctx.stroke();

  drawCorner(tl, tr, bl);
  drawCorner(tr, br, tl);
  drawCorner(br, bl, tr);
  drawCorner(bl, tl, br);
  ctx.restore();
}

function drawCorner(p, a, b) {
  const len = 18;
  const va = unit(a.x - p.x, a.y - p.y);
  const vb = unit(b.x - p.x, b.y - p.y);
  ctx.beginPath();
  ctx.moveTo(p.x + va.x * len, p.y + va.y * len);
  ctx.lineTo(p.x, p.y);
  ctx.lineTo(p.x + vb.x * len, p.y + vb.y * len);
  ctx.strokeStyle = "rgba(255,255,255,.96)";
  ctx.lineWidth = 2.2;
  ctx.stroke();
}

function unit(x, y) {
  const length = Math.hypot(x, y) || 1;
  return { x: x / length, y: y / length };
}

function drawDebugHands(result, transform) {
  if (!debug || !result?.landmarks) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const landmarks of result.landmarks) {
    const points = landmarks.map(lm => landmarkToScreen(lm, transform));
    ctx.strokeStyle = "rgba(110, 234, 255, .72)";
    ctx.lineWidth = 1.5;
    for (const [a, b] of handConnections) {
      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    }
    for (let i = 0; i < points.length; i++) {
      ctx.beginPath();
      ctx.arc(points[i].x, points[i].y, i === 4 || i === 8 ? 4.6 : 2.2, 0, Math.PI * 2);
      ctx.fillStyle = i === 4 || i === 8 ? "rgba(255,235,120,.95)" : "rgba(229,252,255,.8)";
      ctx.fill();
    }
  }
  ctx.restore();
}

function updateFps(now) {
  fpsFrames++;
  const elapsed = now - fpsStarted;
  if (elapsed >= 700) {
    measuredFps = fpsFrames * 1000 / elapsed;
    fpsEl.textContent = Math.round(measuredFps);
    fpsFrames = 0;
    fpsStarted = now;
  }
}

function renderLoop(now) {
  if (!running) return;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const transform = coverTransform(video.videoWidth, video.videoHeight, width, height);

  drawCameraFrame(width, height, transform);
  updateSourceFrame();

  let result = null;
  if (handLandmarker && video.currentTime !== lastVideoTime && !inferenceBusy) {
    inferenceBusy = true;
    try {
      result = handLandmarker.detectForVideo(video, performance.now());
      lastVideoTime = video.currentTime;
      lastInferenceAt = now;
      window.__lastHandResult = result;
    } catch (error) {
      console.error("Hand detection error", error);
    } finally {
      inferenceBusy = false;
    }
  } else {
    result = window.__lastHandResult || null;
  }

  const portal = buildPortal(result, transform);
  const hasTwoHands = Boolean(result?.landmarks?.length >= 2);
  if (portal) {
    applyFilter();
    portalAlpha += (1 - portalAlpha) * 0.22;
    drawPortal(portal.quad, portalAlpha, transform);
    hint.classList.remove("visible");
    setStatus("Portal aktif", "ready");
  } else {
    portalAlpha += (0 - portalAlpha) * 0.18;
    if (!hasTwoHands) {
      hint.innerHTML = "Tunjukkan <b>dua tangan</b> ke kamera";
      setStatus(hasTwoHands ? "Tracking" : "Cari 2 tangan", "warn");
    } else {
      hint.innerHTML = "Jauhkan tangan & buka <b>jempol + telunjuk</b>";
      setStatus("Buka portal", "warn");
    }
    hint.classList.add("visible");
  }

  drawDebugHands(result, transform);
  updateFps(now);
  requestAnimationFrame(renderLoop);
}

startBtn.addEventListener("click", openCamera);

filterBtn.addEventListener("click", () => {
  filterIndex = (filterIndex + 1) % FILTERS.length;
  filterName.textContent = FILTERS[filterIndex].name;
  thermalTick = 0;
  showToast(`Filter: ${FILTERS[filterIndex].name}`);
});

debugBtn.addEventListener("click", () => {
  debug = !debug;
  debugBtn.classList.toggle("active", debug);
  showToast(debug ? "Landmark tangan: ON" : "Landmark tangan: OFF");
});

cameraBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  showToast(facingMode === "user" ? "Kamera depan" : "Kamera belakang");
  await openCamera();
});

window.addEventListener("resize", () => {
  if (video.videoWidth) resizeCanvases();
});

window.addEventListener("orientationchange", () => {
  setTimeout(() => video.videoWidth && resizeCanvases(), 250);
});

window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
  handLandmarker?.close?.();
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js").catch(console.warn));
}

startBtn.disabled = true;
initializeModel().catch(error => {
  console.error(error);
  bootStatus.textContent = "Model gagal dimuat. Cek koneksi internet lalu refresh.";
  setStatus("Model gagal", "error");
});
