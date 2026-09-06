import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const MP_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let landmarker = null;
let backend = "GPU";
let ready = false;

async function init() {
  const vision = await FilesetResolver.forVisionTasks(MP_ROOT);
  const common = {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.48,
    minHandPresenceConfidence: 0.50,
    minTrackingConfidence: 0.58
  };

  try {
    landmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" }
    });
    backend = "GPU";
  } catch (gpuError) {
    landmarker = await HandLandmarker.createFromOptions(vision, {
      ...common,
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "CPU" }
    });
    backend = "CPU";
  }

  ready = true;
  postMessage({ type: "ready", backend });
}

self.onmessage = async event => {
  const msg = event.data || {};
  if (msg.type === "init") {
    try { await init(); }
    catch (error) { postMessage({ type: "fatal", message: error?.message || String(error) }); }
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
    const landmarks = (result.landmarks || []).map(hand => hand.map(p => ({ x: p.x, y: p.y, z: p.z || 0 })));
    postMessage({ type: "result", session: msg.session, landmarks, cost: performance.now() - started, backend });
  } catch (error) {
    postMessage({ type: "result", session: msg.session, landmarks: [], cost: performance.now() - started, backend, error: error?.message || String(error) });
  } finally {
    bitmap.close?.();
  }
};
