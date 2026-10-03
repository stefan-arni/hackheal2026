# NoseThumb tracking (iOS)

Live nose-bridge-to-thumb distance for the Near Point Convergence (NPC) test, measured on an iPhone with MediaPipe and the phone's depth cameras.

- **Front camera:** TrueDepth (the Face ID camera) for depth.
- **Back camera:** LiDAR for depth (Pro iPhones only).
- **MediaPipe Pose Landmarker** (full model) finds the eyes, nose and shoulders.
- **MediaPipe Hand Landmarker** finds the thumb tip.
- **Output:** distance from the bridge of the nose to the thumb tip, in cm, updated live.

This repo is only the tracking part: no networking or video calling.

## Requirements

- Xcode 26 (the project uses the Xcode 16+ "synchronized folder" format)
- A real iPhone running iOS 17 or later. The simulator has no TrueDepth or LiDAR camera.
  - Front camera: any iPhone with Face ID.
  - Back camera: an iPhone Pro (12 Pro or later) with LiDAR.
- [CocoaPods](https://cocoapods.org) for MediaPipe

## Setup

```bash
pod install
open NoseThumb.xcworkspace
```

Always open **NoseThumb.xcworkspace**, not the `.xcodeproj`. Select your iPhone, set your signing team under *Signing & Capabilities*, and run.

The MediaPipe model files are included in `NoseThumb/`:

| File | Used for |
|---|---|
| `pose_landmarker_full.task` | Pose: nose, eye corners, shoulders |
| `hand_landmarker.task` | Hand: thumb tip |

Both come from Google's official MediaPipe model storage.

## How the measurement works

1. **Camera + depth.** `CameraManager` runs the front TrueDepth or back LiDAR camera. Each video frame arrives together with a depth map (distance from the phone per pixel), paired by `AVCaptureDataOutputSynchronizer`. Both streams are rotated to portrait. The front camera is mirrored like a selfie; the back camera is not.
2. **Tracking.** `PoseDetector` sends every frame to both MediaPipe models (live-stream mode).
3. **Bridge of the nose.**
   - **Facing the camera:** the midpoint of the two inner eye corners (pose landmarks 1 and 4).
   - **Side view (profile):** that midpoint lies inside the face, so the code walks along that row of the depth map toward the nose tip until depth jumps to the background. That edge is the front of the nose bridge. If one frame fails, the last position is held for up to 1 s.
4. **Thumb tip.** Hand landmark 4.
5. **Distance in cm.** The NPC test is done side-on, so the bridge and the thumb are about the same distance from the phone. The code measures in that plane:

   ```
   focal length (px) = (image long side / 2) / tan(field of view / 2)
   distance (cm)     = pixel distance between the two points × bridge depth (cm) / focal length
   ```

   The thumb's own depth is **not** used for the distance: a thin, moving fingertip often reads the wall behind it.

## Reading the values (for integration)

All values are on `camera.poseDetector` (`PoseDetector`) and `camera` (`CameraManager`). Both are `@Observable`, so SwiftUI views update automatically. Values change on the main thread.

| Value | Type | Meaning |
|---|---|---|
| `bridgeThumbCM` | `Double?` | **The measurement:** bridge-to-thumb distance in cm. `nil` when it can't be measured reliably. |
| `bridgeFront` | `TrackedPoint?` | Bridge of the nose (front edge in profile). |
| `bridgeDepthCM` | `Double?` | Phone-to-bridge distance in cm. |
| `thumb` | `TrackedPoint?` | Thumb tip, `nil` if no hand is found. |
| `thumbDepthCM` | `Double?` | Phone-to-thumb distance (informational only, not used for the measurement). |
| `isProfile` | `Bool` | The face is side-on to the camera, as the test requires. |
| `personDetected` | `Bool` | A person is in view. |
| `nose`, `leftShoulder`, `rightShoulder` | `TrackedPoint?` | Other pose points. |
| `imageSize` | `CGSize` | Camera frame size in pixels (portrait). |
| `fieldOfView` | `Float` | Camera field of view in degrees. |
| `camera.position` | `.front` / `.back` | Active camera. Switch with `camera.switchCamera()`. |

`TrackedPoint` has `x` and `y` as **0…1 fractions of the camera frame** (0,0 = top-left), plus a `visibility` score (0…1). Points with visibility below `PoseDetector.minVisibility` (0.5) should be treated as "not visible".

`ContentView.trackingSnapshot(_:)` shows how to collect all values into one JSON-ready dictionary, for example to send to a dashboard.

## Screens

- **Clinician view:** camera with dots (green = bridge, pink = thumb, cyan = shoulder), the live distance, and debug info.
- **Patient view:** camera and one large instruction (turn side-on, raise the thumb, bring it toward the nose…), no numbers.

Switch between them with the button at the bottom. The debug lines (FPS, depth rate, bridge status) are temporary.

## Files

| File | What it does |
|---|---|
| `CameraManager.swift` | Camera session: TrueDepth/LiDAR selection, depth format, rotation/mirroring, frame + depth delivery |
| `PoseDetector.swift` | MediaPipe pose + hand, bridge detection, depth reading, distance calculation |
| `LandmarkOverlay.swift` | Draws the dots on the camera image |
| `CameraPreview.swift` | Full-screen camera preview |
| `ContentView.swift` | Patient / clinician screens |

## Known limits

- **Not yet validated against a ruler.** Treat distances as unverified.
- No depth closer than about 16 cm to the front camera.
- Depth can arrive at a lower rate than video (about 14/s on LiDAR). The last valid depth is reused for up to 1 s.
- Front camera: about 24–30 FPS with both models and depth on an A15 iPhone.
- The distance assumes the thumb stays on the patient's midline (side view).
