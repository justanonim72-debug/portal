// Portal V6 — classic CPU-first MediaPipe worker for Android/Chromium.
// Intentionally no import/export: MediaPipe CJS + Emscripten expects importScripts().
const MP_VERSION = '0.10.35';
const BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.cjs`;
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker = null;
let ready = false;
let mediaPipe = null;

const post = (type, extra = {}) => self.postMessage({ type, ...extra });
const timeout = (promise, ms, label) => Promise.race([
  promise,
  new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timeout ${ms}ms`)), ms))
]);

async function loadClassicBundle() {
  if (mediaPipe) return mediaPipe;
  post('progress', { stage: 'bundle', text: 'Mengunduh engine tracking…' });
  const response = await timeout(fetch(BUNDLE_URL, { cache: 'force-cache' }), 10000, 'bundle fetch');
  if (!response.ok) throw new Error(`MediaPipe bundle HTTP ${response.status}`);
  const code = await timeout(response.text(), 7000, 'bundle read');

  self.module = { exports: {} };
  self.exports = self.module.exports;
  const blobUrl = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
  try {
    importScripts(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
  mediaPipe = self.module?.exports;
  if (!mediaPipe?.FilesetResolver || !mediaPipe?.HandLandmarker) {
    throw new Error('MediaPipe classic API tidak tersedia');
  }
  return mediaPipe;
}

async function init() {
  const mp = await loadClassicBundle();
  post('progress', { stage: 'wasm', text: 'Menyiapkan WASM…' });
  const vision = await timeout(mp.FilesetResolver.forVisionTasks(WASM_ROOT), 12000, 'WASM init');

  // CPU first on purpose. Worker GPU init can stall for tens of seconds on Android.
  post('progress', { stage: 'model', text: 'Memuat model tangan…' });
  landmarker = await timeout(mp.HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'CPU' },
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.46,
    minHandPresenceConfidence: 0.48,
    minTrackingConfidence: 0.56
  }), 15000, 'model init');

  ready = true;
  post('ready', { backend: 'CPU' });
}

self.onmessage = async event => {
  const msg = event.data || {};
  if (msg.type === 'init') {
    try { await timeout(init(), 30000, 'worker init'); }
    catch (error) { post('fatal', { message: error?.message || String(error) }); }
    return;
  }
  if (msg.type !== 'frame') return;

  const bitmap = msg.bitmap;
  if (!bitmap) return;
  if (!ready || !landmarker) {
    bitmap.close?.();
    post('result', { session: msg.session, landmarks: [], cost: 0, skipped: true, backend: 'CPU' });
    return;
  }

  const started = performance.now();
  try {
    const detection = landmarker.detectForVideo(bitmap, msg.timestamp || performance.now());
    const landmarks = (detection.landmarks || []).map(hand => hand.map(p => ({ x: p.x, y: p.y, z: p.z || 0 })));
    post('result', { session: msg.session, landmarks, cost: performance.now() - started, backend: 'CPU' });
  } catch (error) {
    post('result', { session: msg.session, landmarks: [], cost: performance.now() - started, backend: 'CPU', error: error?.message || String(error) });
  } finally {
    bitmap.close?.();
  }
};
