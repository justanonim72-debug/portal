# Portal native Android

CameraX camera + MediaPipe Hand Landmarker (GPU preferred, CPU fallback) with a persistent four-corner **live camera filter plane**. Both preview and MP4 recording use the same GPU effect.

1. Pinch your thumb to index, middle, ring or pinky for a moment to create a small panel.
2. Release the pinch. Touch a corner or any edge with an extended fingertip, pause briefly, then drag. Multiple fingertips/hands can grab distinct corners or opposite edges.
3. Fold/pinch the controlling finger to release. The panel remains until `×` reset or camera switch.
4. Cycle Violet / Invert / Mono / Thermal / Clear, toggle the debug hand skeleton, switch front/back camera, or record a silent MP4 to `Movies/Portal`.

Use [Build Portal Android APK](../.github/workflows/build-portal-android.yml) for a tested installable APK. It runs JVM tests, lint, Android emulator shader/recording checks, signature verification and APK alignment checks. The official MediaPipe model is fetched during CI.

For local builds use Java 17, Gradle 8.11.1, Android platform 36, and place the [official model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) at `app/src/main/assets/hand_landmarker.task`:

```sh
gradle :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
gradle :app:connectedDebugAndroidTest
```

See [interaction architecture and verification](docs/interaction.md) for root causes, the exact state machine, coordinate/identity/filter contracts, performance measurements and remaining physical-device validation.
