# Portal Android Native

Native Android version of Portal. This is now the primary runtime path; the GitHub Pages PWA remains a web demo.

## Runtime design

- CameraX 1.6.2 for preview, image analysis and video capture.
- MediaPipe Hand Landmarker 0.10.35.
- GPU delegate is attempted first on a dedicated MediaPipe thread; CPU is fallback only.
- `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST` prevents inference backlog.
- Thumb tip (landmark 4) + index tip (landmark 8) are the hard anchors for one-hand control.
- Other fingers can only bend the far membrane slightly; they do not move the anchor edge.
- CameraX `OverlayEffect` renders the membrane through the camera OpenGL pipeline and targets both Preview + VideoCapture, so recorded MP4 includes the portal overlay.

## Build from GitHub Actions

Run **Build Portal Android APK** or push a change under `android/`.

Artifact:

- `Portal-Android-GPU`
- `Portal-native-GPU-debug.apk`

The workflow downloads the official MediaPipe `hand_landmarker.task` model during CI, so the public source repo does not need to store the binary model.

## Device gate

The UI shows the actual MediaPipe backend after initialization:

- `TRACKING · GPU` = desired path.
- `TRACKING · CPU fallback` = GPU delegate failed on that runtime and must be investigated before performance tuning.

The top-right HUD reports average inference latency and detector updates/second separately from camera rendering.
