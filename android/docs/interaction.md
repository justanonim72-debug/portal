# Portal interaction and verification

Issue: [#1](https://github.com/justanonim72-debug/portal/issues/1).
Visual reference: [Agustina Waigel's original reel](https://www.instagram.com/reel/DctUY49xsjj/).
The 27-second reel was downloaded and inspected at 0–24 seconds in 3-second intervals. It shows straight four-sided filter windows: violet, inverted, monochrome and thermal camera content, small borders, and deliberate independent corner movement. It also shows crossed shapes; this implementation deliberately prevents those, as issue #1 requires. The reel is a visual reference, not a source of hand identity or gesture code. The explicit pinch/persist/grab contract comes from the issue.

## Root causes found in history

- `940c5cb` introduced a quadrilateral but its `Grab` retained only a finger slot. Matching selected the nearest finger with that slot from **either** hand. There was no persistent hand identity, and only one grab could exist.
- Grabs targeted eight isolated points (corners and edge midpoints), not entire edges. Corner grabs snapped the vertex to the fingertip, losing the initial contact offset.
- `e4d99c7` made geometry isotropic using the *rotated* image dimensions. However, the input buffer itself was not rotated: `ImageProcessingOptions` describes the inference ROI. MediaPipe projects its output landmarks back into the original input image. The renderer then incorrectly undid an additional rotation. This affected skeleton alignment as well as the meaning of horizontal distances.
- `a1dafc7` drew translucent gradients/grid lines on `OverlayEffect`. An overlay canvas cannot sample/filter the live camera image. The renderer also added a second normalized-coordinate motion limiter/filter after geometry validation.
- A frame counter controlled debounce/dropout, portal lifetime depended on seeing hands, and a minimum seed clamp could exceed a small measured finger segment.

The reusable direct RGBA buffer and synchronous MediaPipe `VIDEO` ownership fix from `14106e7` is preserved. CameraX cross-use-case sensor transforms are preserved, with the incorrect extra rotation/mirroring removed. Raw hand count remains independent of valid geometry/tracks.

## Pipeline

```
CameraX RGBA ImageAnalysis (KEEP_ONLY_LATEST)
  -> dedicated MediaPipe thread, GPU preferred / CPU initialization fallback
  -> original analysis-pixel landmarks + raw detected count
  -> HandTracks (global two-hand assignment, per-joint independent filters)
  -> PortalInteraction (explicit reducer + constrained four-corner geometry)
  -> immutable PortalState published atomically

CameraX SurfaceTexture (independent GL thread)
  -> current camera texture, CameraX output texture transform
  -> PortalRenderer (live color filter inside the quad, thin border/handles)
  -> shared CameraEffect PREVIEW | VIDEO_CAPTURE -> preview / Recorder MP4
```

`SurfaceOutput.getSensorToBufferTransform()` composed with the captured `analysisToSensor` matrix maps **both** the panel and skeleton into each output buffer. CameraX's `updateTransformMatrix()` independently maps the live camera texture, including crop/rotation/mirroring. There is no assumed preview aspect ratio, selfie X flip, landmark ROI rotation, or additional geometry smoothing in the renderer. GL's bottom-left fragment coordinates are converted to top-left output pixels once.

Every interaction distance, dot/cross product, velocity, angle and seed basis uses **original analysis pixels** (`x * inputWidth`, `y * inputHeight`), so X and Y have equal physical image scale. Rotation/reflection only occurs at the output boundary. Changing camera, analysis dimensions or rotation resets the session.

MediaPipe projection source: [hand_landmarks_detector_graph.cc](https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc), `LandmarkProjectionCalculator` projects cropped hand landmarks back to the full input image. CameraX contracts: [SurfaceOutput](https://developer.android.com/reference/androidx/camera/core/SurfaceOutput), [SurfaceProcessor](https://developer.android.com/reference/androidx/camera/core/SurfaceProcessor).

## State machine and controls

| State | Entry / exit |
|---|---|
| `IDLE` | No panel. Thumb within 0.22 hand scales of index/middle/ring/pinky enters candidate. |
| `TRIGGER_CANDIDATE` | Retain the same `(hand ID, fingertip landmark)` for 90 ms. Distance above 0.38 scales, a missing hand or a different candidate cancels/restarts debounce. |
| `SEEDED` | One tiny rectangle centered at the actual thumb/finger midpoint. Its basis is the distal finger segment, frozen at creation. Even its diagonal is smaller than that measured segment; there is no screen-size minimum. Wait for trigger release (0.38 scales) for 70 ms. |
| `ACTIVE` | Panel persists indefinitely, including with no hands. Any eligible fingertip can touch a corner/full edge for 70 ms to grab it. Trigger tips must leave the seed first. |
| `GRABBING` | One latched `(hand ID, fingertip landmark, corner/edge)`. Move one corner or both edge endpoints by fingertip delta, preserving contact offset. |
| `RESIZING` | Two or more latched fingertips; simultaneous deltas applied together. Handles cannot own overlapping vertices. Opposite edges or distinct corners can move concurrently, including on the same hand. |

Pinch the grabbing finger to the thumb or fold that finger for 70 ms to release it. For a thumb grab, close it toward a fingertip. Missing owners freeze their handle for up to 250 ms, then release it; the panel persists. Returning hands rebase contact and reset their filters so travel during the miss is not applied later. Extreme one-frame jumps are rebased rather than applied. A released tip must exit the border capture zone before relatching while continuously visible. A missing tip must reestablish contact debounce.

Normal UI shows tiny corner markers and highlights owned handles. Debug adds a separately filtered 21-joint skeleton, colored by stable ID. `×` explicitly resets the panel. Camera switch resets the session and is disabled during a recording; stop recording first. Existing camera selection, backend status and silent MediaStore MP4 recording remain available.

## Tracking, constraints and cost

- At most two identities, assigned globally over both observations using palm proximity, bounded constant-velocity prediction and handedness confidence. Missing identities are retained for 500 ms; IDs are monotonically allocated and never reassigned to another finger by nearest-distance selection.
- One Euro filtering uses a shared XY speed/cutoff per landmark for rotational symmetry. Interaction cutoff 4 Hz / beta 0.035; independent skeleton cutoff 3 Hz / beta 0.025; derivative cutoff 1 Hz. Faster deliberate movements reduce lag. Trigger measurements use raw landmarks so the initial size/location is tied to the current physical segment.
- Simultaneous proposals are constrained along their movement path to preserve positive winding, convex corners, seed-relative minimum edge length and area. Handles/vertices are never reordered. Invalid/crossing motion stops at the boundary, and reverse motion can immediately recover.
- Analysis runs synchronously on its worker so the direct RGBA buffer remains owned until `detectForVideo` returns and the `MPImage` is closed. CameraX drops stale analysis frames. Generation checks prevent in-flight results from republishing a reset/camera-change session.
- Rendering samples the live external camera texture on GL; it does not copy camera bitmaps, wait for inference, or perform CPU pixel readback. Buffers/programs are reused. Immutable tracking snapshots allocate small arrays at inference rate.
- The readout measures average analysis/inference processing milliseconds, tracking updates/second, raw hand count, and analysis-start-to-first-render milliseconds. The latter excludes sensor exposure/ISP delay, compositor presentation and display scanout; it is not a photon-to-display measurement. Device measurements with/without recording are still required.

## Automated and device verification

`gradle :app:testDebugUnitTest :app:lintDebug :app:assembleDebug` builds the APK and runs the pure Java regression suite. `python3 tools/test-core.py` runs the same core tests with cached JUnit jars on hosts without a supported Android build environment.

CI also runs `:app:connectedDebugAndroidTest` on an API 35 emulator:

- Real GLES camera-texture pixel checks verify a horizontal quadrilateral in a portrait buffer, inside/outside color filtering, sensor transform composition and reset.
- A real CameraX preview/Recorder test records front and back cameras around a lifecycle unbind/rebind, decodes the resulting MP4 frames and checks that the GPU violet filter is present. It supplies deterministic panel geometry; it does not claim to validate real hand detection from emulator camera imagery.
- APK signature/alignment checks and install via instrumentation verify an installable artifact. Reports and decoded recording PNGs are uploaded with CI.

Physical-device checks still needed: skeleton alignment on a real portrait hand, all four trigger fingers at near/far distances, deliberate same-hand/two-hand grabs (including horizontal holds), low light/occlusion and crossing hands, front/back preview-to-gallery alignment, and GPU/CPU tracking/latency while recording. Two visually indistinguishable overlapping hands with ambiguous handedness remain a tracking ambiguity; no temporal matcher can guarantee identity through a complete long occlusion. Device-only acceptance must not be inferred from the synthetic tests.
