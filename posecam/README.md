# posecam — pose and eye tracking over WebSocket

Two **independent** programs. The iPhone app streams camera frames to either one (or
both) over a WebSocket and gets JSON results back. Each has a laptop test client that
uses your webcam in place of the iPhone.

| | Server | Laptop client | Port | What it does |
|---|---|---|---|---|
| **Pose** | `server.py` | `test_client.py` | 8765 | Full-body landmarks, spine angle, single-leg balance, BESS balance test, duck → quack |
| **Eyes** | `eye_server.py` | `eye_client.py` | 8766 | Iris tracking + alert when one eye drifts |

They share only `ws_server.py` / `ws_client.py` (frame decoding and connection
handling). Pose code lives in `pose_analyzer.py`, `balance.py`, `bess.py`, `duck.py` and
`pose_pipeline.py` (which runs them together); eye code in `eye_tracker.py`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

MediaPipe models download into `models/` the first time each server starts.

## Run

```powershell
# Pose
python server.py            # terminal 1
python test_client.py       # terminal 2

# Eyes
python eye_server.py        # terminal 1
python eye_client.py        # terminal 2
```

Run all four at once if you want both. Keys: **q** quits either window, **r** resets the
balance stats and quack count in the pose window, **c** recalibrates in the eye window.
The pose window has a BESS panel on the right with a button per test (see below).

The iPhone screen with the three test buttons is `ios/BessTestView.swift`.

| Command | What it does |
|---|---|
| `python test_client.py --local` / `python eye_client.py --local` | Run MediaPipe directly, no server |
| `--source clip.mp4` | Video file instead of the webcam (`--source 1` for a second camera) |
| `--source me.jpg` | One image; prints the JSON result |
| `--json` | Send base64 JSON instead of binary (test both iOS formats) |
| `--log out.csv` | Save per-frame values |
| `python server.py --model lite` | Faster pose model; `heavy` is more accurate |
| `python server.py --no-balance` / `--no-duck` / `--no-bess` | Turn off balance, duck or BESS |
| `python server.py --bess-eyes auto` | BESS: turn eye tracking back on (off by default) |
| `python test_client.py --bess tandem` | Start a BESS test right away (`double`, `tandem`, `single`) |
| `python test_client.py --nondominant right` | BESS non-dominant leg (default left; **n** toggles) |
| `python test_client.py --mute` | No quack sound |
| `python quack.py` | Just play the quack (`sounds/quack.wav`) |
| `python eye_server.py --threshold 0.08` | More sensitive eye alerts (see tuning) |
| `pytest -q` | All tests (no models needed) |

Windows: allow camera access under Settings → Privacy & security → Camera → "Let
desktop apps access your camera", and allow the server on **Private networks** when
the firewall asks (needed for the iPhone).

## Pose: spine angle

MediaPipe has no spine landmarks, so the spine is the **trunk line from the hip
midpoint to the shoulder midpoint**.

| Field | Meaning |
|---|---|
| `trunk_angle_deg` | 2D angle from vertical in the image. Positive = shoulders to the image-right of the hips. **From a side view this is the forward/back lean, the most reliable measurement.** From the front it's the sideways lean. |
| `inclination_deg` | 3D total tilt from vertical (0 = upright), from MediaPipe world landmarks |
| `flexion_deg` | 3D forward lean, positive = toward the camera. Noisy from a front view (depth from one camera). |
| `lateral_deg` | 3D sideways lean, positive = toward image-right |
| `trunk_visibility`, `reliable` | Lowest visibility of the 4 hip/shoulder points; `reliable` is false below 0.5 |

## Pose: single-leg balance

While someone stands on one leg, the pose server times it and sends an alert the
moment the raised foot touches the ground.

1. **Both feet down:** idle.
2. **Lift one foot** (above 8 cm) for 0.3 s: the timer starts. You get a
   `balance_started` status saying which foot is up.
3. **Raised foot comes back down** (below 3 cm): a `foot_touchdown` alert with how
   long they held it. Then it's ready for the next attempt, on either leg. It keeps a
   touch count and best time until you press **r** (or send `{"type":"recalibrate"}`).

How it works: each foot's lowest point (ankle, heel or toe) is found and its height
above the floor is measured. The "foot down" level is lower than the "foot up" level,
so a foot hovering just above the floor doesn't flicker on and off, and a single
noisy frame can't trigger a touchdown.

**Floor calibration:** whenever both feet are flat for about a second (including at
the start), the floor is re-learned. In 3D, each foot's resting offset is removed
(the model often reads one foot a centimeter or two "higher" even when it's down).
In 2D, a floor line through both resting feet is stored and the raised foot is
measured from it, so wobble in the standing foot no longer looks like the raised
foot moving. Until the first calibration, the raised foot is measured against the
other foot. The window shows "floor: calibrated" once it's learned. Turn it off with
`--no-floor-calibration`.

Height is measured in 3D by default (`--balance-mode world`, meters), which isn't
fooled by perspective: a foot placed further back looks higher in the image even
when it's on the floor. `--balance-mode image` measures in 2D instead, as a fraction
of leg length; try it if the 3D estimate is jumpy for your camera setup. Tune with
`--lift-threshold` and `--touch-threshold`, using the foot heights logged by
`test_client.py --log balance.csv`.

**Setup:** the whole body, including both feet, must be in frame, ideally with the
camera at hip height a few meters away. If the feet can't be seen (`skip_reason:
"feet_not_visible"`), the current state is held rather than guessed.

## Pose: BESS balance test

The Balance Error Scoring System on a firm surface: three 20-second stances, hands on
hips. The score is the number of errors, per stance and in total. **Eye tracking is
off for now**: eyes aren't checked or scored (see "Eyes" below).

| Button | Stance |
|---|---|
| **Feet together** (`double`) | feet side by side, touching |
| **Tandem** (`tandem`) | one foot in front of the other, **non-dominant foot in back** |
| **Single leg** (`single`) | standing on the **non-dominant** leg |

**Running a test:** pick the non-dominant leg, press a stance button, and get into
position during the 5-second countdown. The last second of the countdown records the
person's own start position, so scoring is relative to how they actually stand. Then
20 seconds of scoring, and the result. Re-running a stance replaces its score;
**Reset scores** clears all three.

**One point per error, one rule per error type**, each counted once per distinct
event (it has to clear before it counts again):

| Error | Rule |
|---|---|
| Hands off hips | a wrist moves away from its hip (beyond 0.45 torso-lengths, or 0.2 further than at the start) |
| Step, stumble or fall | a stance foot's ankle moves more than 0.15 leg-lengths, the raised foot touches down (single leg), or the hips drop 0.25 leg-lengths |
| Hip > 30° | either hip goes past 30° of flexion or abduction, i.e. the thigh rotates more than 30° relative to the trunk (see below) |
| Heel or forefoot lifted | a stance foot's heel or toe rises more than 0.05 leg-lengths |
| Out of position > 5 s | any of the above held continuously for more than 5 seconds (one extra point per episode) |

As in the standard BESS, errors that start within 0.5 s of each other count as one
(e.g. a step that also lifts the heel), and a stance scores at most 10. A condition
has to last 0.2 s to count, so a single bad frame doesn't. Frames where a body part
can't be seen don't start or end an error.

**How the hip angle is measured:** mostly from the 2D image, because a single
camera's depth estimate is too noisy (it could read 15° on someone standing
perfectly straight, with spikes past 30°). Sideways and in-image angles come
straight from the image. Bending toward or away from the camera is measured from how
much shorter the torso and thigh look than at the start (30° of tilt makes a segment
look 13% shorter); the 3D model is only used to tell toward from away. For standing
legs it's the change since the start position, so each person's natural posture
doesn't count, and whole-body sway from the ankles (trunk and thighs moving together)
isn't hip flexion. The raised leg in the single-leg stance starts at about 20° on
purpose, so it's measured as its absolute angle from the trunk. The panel shows the
live angles ("hip L 12 R 8 deg", red above 30), and `--log` records them. It works
from the front or the side; avoid steep camera angles from above or below.

**Single-leg tracking pauses for feet together and tandem.** During those two tests
both feet are supposed to stay down, so the single-leg "raised foot touched the
ground" tracker (and its FOOT DOWN alert) is paused; the window says so. Heel or
forefoot lifts are still scored as BESS errors. Tracking resumes when the test ends,
and stays on during the single-leg test.

At the start of each trial it warns about setup problems, for example standing on the
wrong leg in the single-leg test, the wrong foot in back in tandem, or hands not on
the hips. Warnings don't change the score.

**Camera:** fixed (tripod), whole body in frame including feet and hands, a couple of
meters away. A side or 45° view makes the hip angle and steps easier to see. If the
start position can't be seen for 10 s after the countdown, the test is called off
with a message.

**Eyes (off for now):** by default eyes aren't tracked, the face model isn't loaded,
and "Eyes opened" isn't an error type. The code is still there: `--bess-eyes auto`
checks eyes with the face model on an enlarged crop of the head, and `--bess-eyes
manual` lets the app send `{"type":"bess_mark","error":"eyes_open"}`.

Not automated: the standard BESS rule that a person who can't hold the stance for at
least 5 s scores 10 for that stance. The examiner should apply that by hand.

This is a demo aid, not a validated clinical scoring tool. Compare it against a
trained examiner before relying on it.

## Pose: duck → quack

Duck down quickly and the test client quacks (and flashes "QUACK!"). The sound is
`sounds/quack.wav`, a real duck quack trimmed to 0.2 s so it plays the instant the
duck is detected. Swap in any short WAV to change it; if the file is missing,
`quack.py` falls back to a synthesized quack. On the iPhone, add the same file to
the app bundle and play it when the `duck` event arrives.

- **Full body in frame:** a duck is the head dropping 20% below the person's height
  from the last 2 seconds (measured in 3D, head to feet). Tune with `--duck-drop`.
- **Feet out of frame** (e.g. webcam on a desk): a duck is the nose dropping 0.8
  shoulder-widths. Tune with `--duck-drop-image`.

Because the reference is "the last 2 seconds", slowly sinking into a chair doesn't
quack, only a quick drop does. One quack per duck: you have to come most of the way
back up before the next one counts.

## Eyes: drift alerts

Tracks both irises with MediaPipe's face model and alerts when **one eye moves out of
line with the other**, e.g. one eye turns outward while the other keeps looking ahead.

1. **Calibration (first ~2 s):** look straight at the camera. The server learns this
   person's normal alignment, so natural small asymmetries don't trigger alerts.
   Press **c** (or send `{"type":"recalibrate"}`) to redo it.
2. **Monitoring:** for each eye it measures where the iris sits between the eye
   corners. Looking somewhere with both eyes moves them together and is fine. When
   the gap between the eyes changes past the threshold for 0.5 s, it alerts with which
   eye moved and whether the drift is horizontal or vertical.
3. Frames are ignored during **blinks**, when the **head is turned** too far, or when the
   face is **too small** to measure (`skip_reason` says which).

**Try it:** after calibration, cross your eyes for a second. Both eyes swing in
opposite directions, which looks like misalignment, so the window flashes red and
beeps. Look normally again and the alert clears.

The face should fill a good part of the frame, with decent light. In a full-body shot
the eyes are only a few pixels wide and you'll get `face_too_small`.

**Tuning:** `score` = drift ÷ threshold, shown live and logged with `--log eyes.csv`.
Watch the score while looking around normally, then set `--threshold` (horizontal,
default 0.10 eye-widths), `--vthreshold` (vertical, default 0.08) and `--hold`
(default 0.5 s) on `eye_server.py`.

"right"/"left" are the **subject's** eyes assuming an unmirrored camera; with a
mirrored front-camera feed they're swapped.

This is a demo/screening aid, not a validated medical device.

## WebSocket protocol (for the iPhone app)

Same input format for both servers. Connect to `ws://<laptop-LAN-IP>:8765` (pose) or
`:8766` (eyes) on the same Wi-Fi, then send either:

- **Binary:** raw JPEG bytes of one frame. From Swift:
  `task.send(.data(jpegData))` with `URLSessionWebSocketTask`.
- **Text JSON:** `{"type":"frame","image":"<base64 JPEG>","frame_id":42,"timestamp_ms":...,"rotate":0}`.
  `rotate` (0/90/180/270, clockwise) fixes sideways frames; OpenCV ignores EXIF
  orientation, so send upright frames or set `rotate`.
- `{"type":"ping"}` → `{"type":"pong"}`
- `{"type":"recalibrate"}` → pose server: reset balance stats, floor calibration and quack count; eye server: re-learn the baseline
- Pose server, BESS: `{"type":"bess_start","stance":"double"|"tandem"|"single","nondominant":"left"|"right"}`,
  `{"type":"bess_cancel"}`, `{"type":"bess_reset"}`, `{"type":"bess_status"}`,
  `{"type":"bess_mark","error":"hands_off_hips"}` (manual error). Each gets an immediate
  `{"type":"ack", "scores": {...}, "total": n, "complete": bool}` or `{"type":"error","command":...}`.

**Pose server** replies once per processed frame:

```json
{"type": "pose", "frame_id": 42, "client_timestamp_ms": 1696291200000,
 "image_size": [640, 480], "detected": true,
 "landmarks": [{"name": "nose", "x": 0.51, "y": 0.22, "z": -0.3, "visibility": 0.99, "presence": 0.99}, "... 33 total"],
 "world_landmarks": ["... 33, in meters, origin at hip centre"],
 "spine": {"trunk_angle_deg": 4.2, "inclination_deg": 6.1, "flexion_deg": 3.0,
           "lateral_deg": 4.5, "trunk_length_m": 0.48, "trunk_visibility": 0.97, "reliable": true},
 "balance": {"state": "balancing", "event": null, "lifted_foot": "right", "standing_foot": "left",
             "balance_time_s": 4.1, "foot_heights": {"left": 0.0, "right": 0.12}, "mode": "world",
             "touch_count": 1, "last_hold_s": 6.3, "best_hold_s": 6.3, "skip_reason": null},
 "inference_ms": 18.2, "dropped_frames": 0}
```

`balance.state` is `"idle"`, `"lifting"` or `"balancing"`. Extra messages follow the
frame where something happens:

```json
{"type": "status", "kind": "balance_started", "lifted_foot": "right", "standing_foot": "left"}
{"type": "alert", "kind": "foot_touchdown", "foot": "right", "held_s": 6.3,
 "touch_count": 1, "best_hold_s": 6.3}
{"type": "event", "kind": "duck", "count": 3, "drop": 0.27, "mode": "world"}
```

Each pose result also has `"duck": {"ducking": false, "count": 3, "drop": 0.04,
"threshold": 0.2, "mode": "world"}`, and `"bess"`:

```json
"bess": {"phase": "running", "stance": "tandem", "nondominant": "left", "time_left": 12.4,
         "errors": 2, "by_type": {"hands_off_hips": 1, "step_stumble_fall": 1, "...": 0},
         "active": ["hands_off_hips"], "warnings": [],
         "log": [{"error": "hands_off_hips", "label": "Hands off hips", "t": 4.9, "counted": true}],
         "session": {"scores": {"double": 1, "tandem": null, "single": null}, "total": 1, "complete": false},
         "events": []}
```

`phase` is `idle`, `countdown` (with `countdown_left`, `waiting_for_view`) or `running`.
BESS messages, sent as they happen:

```json
{"type": "status", "kind": "bess_started", "stance": "tandem", "nondominant": "left"}
{"type": "status", "kind": "bess_running", "stance": "tandem", "warnings": []}
{"type": "event", "kind": "bess_error", "error": "hands_off_hips", "label": "Hands off hips",
 "t": 4.9, "counted": true, "errors": 1}
{"type": "result", "kind": "bess_done", "stance": "tandem", "label": "Tandem", "errors": 2,
 "by_type": {...}, "log": [...], "coverage": 0.98,
 "session": {"scores": {"double": 1, "tandem": 2, "single": null}, "total": 3, "complete": false}}
{"type": "status", "kind": "bess_failed", "reason": "Couldn't see the full body ..."}
```

`coverage` is the share of frames where the body could be measured. A low value means
the score may be missing errors.

**Eye server** replies once per processed frame:

```json
{"type": "eyes", "frame_id": 42, "image_size": [960, 720], "face_detected": true,
 "state": "monitoring", "alerting": false,
 "deviation_h": 0.012, "deviation_v": -0.004, "score": 0.12,
 "drifting_eye": "left", "direction": "horizontal", "skip_reason": null, "head_yaw": 0.03,
 "eyes": {"right": {"h": 0.51, "v": 0.02, "iris_center_px": [412.3, 301.8], "...": "..."},
          "left": {"...": "..."}},
 "inference_ms": 9.1, "dropped_frames": 0}
```

`state` is `"calibrating"` (with `calibration_progress` 0–1) until the baseline is
learned, then `"monitoring"`. When something changes, an extra message follows that
frame's result, so the app can react without checking every frame:

```json
{"type": "status", "kind": "eye_calibrated"}
{"type": "alert", "kind": "eye_misalignment", "event": "start",
 "drifting_eye": "left", "direction": "horizontal", "score": 1.8, "deviation_h": -0.18}
{"type": "alert", "kind": "eye_misalignment", "event": "end", "alert_duration_s": 3.2}
```

Landmark `x`/`y` are normalized 0–1 to the received image. If a server is busy, older
frames are dropped and only the newest is processed, so the feed stays live. For pose,
~640 px wide is plenty; for eyes, send ~960 px or more.

**iPhone screen:** `ios/BessTestView.swift` is a SwiftUI view (iOS 16+) with the
three test buttons, a non-dominant-leg picker, the live countdown, timer and errors,
and the score table. It includes `PosecamClient`, a
small WebSocket client; call `posecam.sendFrame(jpegData)` from your camera code to
stream frames to the same connection.

iOS notes: plain `ws://` to a LAN IP needs an App Transport Security exception
(`NSAllowsLocalNetworking`) plus `NSLocalNetworkUsageDescription` in Info.plist.
Use `wss://` for anything outside your LAN, especially with patient video.
