// Portal V5.1 — classic MediaPipe worker.
// MediaPipe's Emscripten/WASM loader still depends on importScripts() in a
// worker context. A module worker can fail with "ModuleFactory not set".
// Keep this file import/export-free so it executes as a classic worker.

const MP_VERSION = "0.10.35";
const BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.cjs`;
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let landmarker = null;
let backend = "CPU";
let ready = false;
let mediaPipe = null;

async function loadMediaPipeClassic() {
  if (mediaPipe) return mediaPipe;

  // jsDelivr serves the CJS file as application/node on some paths. Browsers
  // reject that MIME in importScripts(), so fetch it and execute a same-origin
  // Blob with an explicit JavaScript MIME type.
  const response = await fetch(BUNDLE_URL, { cache: "force-cache" });
  if (!response.ok) throw new Error(`MediaPipe bundle HTTP ${response.status}`);
  const code = await response.text();

  // CommonJS shim expected by vision_bundle.cjs.
  self.module = { exports: {} };
  self.exports = self.module.exports;

  const blobUrl = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
  try {
    importScripts(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }

  mediaPipe = self.module && self.module.exports;
  if (!mediaPipe?.FilesetResolver || !mediaPipe?.HandLandmarker) {
    throw new Error("MediaPipe classic bundle tidak mengekspor API vision");
  }
  return mediaPipe;
}

function hasWorkerWebGL2() {
  try {
    return typeof OffscreenCanvas !== "undefined" && !!new OffscreenCanvas(2, 2).getContext("webgl2");
  } catch (_) {
    return false;
  }
}

async function createLandmarker(mp, delegate) {
  const vision = await mp.FilesetResolver.forVisionTasks(WASM_ROOT);
  return mp.HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.48,
    minHandPresenceConfidence: 0.50,
    minTrackingConfidence: 0.58
  });
}

function warmUp() {
  if (!landmarker || typeof OffscreenCanvas === "undefined") return;
  let bitmap = null;
  try {
    const c = new OffscreenCanvas(2, 2);
    const c2d = c.getContext("2d");
    if (!c2d) return;
    c2d.fillStyle = "#000";
    c2d.fillRect(0, 0, 2, 2);
    bitmap = c.transferToImageBitmap();
    landmarker.detectForVideo(bitmap, 1);
  } catch (error) {
    // Warm-up is only an optimization. A real frame will still verify runtime.
    console.warn("Portal worker warmup skipped", error);
  } finally {
    bitmap?.close?.();
  }
}

async function init() {
  const mp = await loadMediaPipeClassic();

  // GPU in a worker requires a usable worker-side WebGL2 context. If creation
  // fails, rebuild cleanly with CPU rather than failing the whole app.
  if (hasWorkerWebGL2()) {
    try {
      landmarker = await createLandmarker(mp, "GPU");
      backend = "GPU";
    } catch (gpuError) {
      console.warn("Portal worker GPU gagal, fallback CPU", gpuError);
      landmarker = await createLandmarker(mp, "CPU");
      backend = "CPU";
    }
  } else {
    landmarker = await createLandmarker(mp, "CPU");
    backend = "CPU";
  }

  warmUp();
  ready = true;
  postMessage({ type: "ready", backend });
}

self.onmessage = async event => {
  const msg = event.data || {};

  if (msg.type === "init") {
    try {
      await init();
    } catch (error) {
      postMessage({ type: "fatal", message: error?.message || String(error) });
    }
    return;
  }

  if (msg.type !== "frame") return;
  const bitmap = msg.bitmap;
  if (!bitmap) return;

  if (!ready || !landmarker) {
    bitmap.close?.();
    postMessage({ type: "result", session: msg.session, landmarks: [], cost: 0, skipped: true, backend });
    return;
  }

  const started = performance.now();
  try {
    const result = landmarker.detectForVideo(bitmap, msg.timestamp || performance.now());
    const landmarks = (result.landmarks || []).map(hand =>
      hand.map(p => ({ x: p.x, y: p.y, z: p.z || 0 }))
    );
    postMessage({
      type: "result",
      session: msg.session,
      landmarks,
      cost: performance.now() - started,
      backend
    });
  } catch (error) {
    postMessage({
      type: "result",
      session: msg.session,
      landmarks: [],
      cost: performance.now() - started,
      backend,
      error: error?.message || String(error)
    });
  } finally {
    bitmap.close?.();
  }
};
