# posecam — pose and eye tracking over WebSocket

Two **independent** programs. The iPhone app streams camera frames to either one (or
both) over a WebSocket and gets JSON results back. Each has a laptop test client that
uses your webcam in place of the iPhone.

| | Server | Laptop client | Port | What it does |
|---|---|---|---|---|
| **Pose** | `server.py` | `test_client.py` | 8765 | Full-body landmarks, spine angle, single-leg balance, BESS balance test, sway tests (quiet / tandem / Romberg, with iPhone LiDAR depth), duck → quack |
| **Eyes** | `eye_server.py` | `eye_client.py` | 8766 | Iris tracking: drift alerts, saccade speed, smoothness (close-up camera) |

They share only `ws_server.py` / `ws_client.py` (frame decoding, connection handling,
session recording). Pose code lives in `pose_analyzer.py`, `balance.py`, `bess.py`,
`sway.py`, `depth.py`, `duck.py` and `pose_pipeline.py` (which runs them together); eye
code in `eye_tracker.py`. `playback.py` replays recorded sessions and video files.

## Quick start: BESS with the doctor's dashboard

```bash
cd posecam && source .venv/bin/activate
python start.py
```

`start.py` starts the pose server (`server.py`, port 8765: MediaPipe pose, LiDAR depth,
BESS scoring) and the call server (`doctor_call_server.py`, port 8088: video call and
dashboard), prints the IP to type into the TeleVision app, and opens the doctor page
(http://localhost:8088). Ctrl+C stops both.

The doctor page joins the room by itself and reconnects if the connection drops. Once
the patient joins from the TeleVision app (same room), it calls them and starts the
phone's balance stream automatically. The strip at the top shows each link (server,
video call, phone → pose server with frames per second, MediaPipe pose, LiDAR depth),
and the MediaPipe skeleton is drawn over the patient's video. Click a stance (Feet
together, Tandem, Single leg) to run it; the dashboard shows the live timer, errors and
the per-stance results.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

MediaPipe models download into `models/` the first time each server starts.

**macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
SYSTEM_VERSION_COMPAT=0 pip install -r requirements.txt
```

`SYSTEM_VERSION_COMPAT=0` matters on Intel Macs with an older Python build (e.g.
Anaconda's): that Python reports macOS as "10.16", so pip rejects mediapipe's
macOS 11+ wheels and says "No matching distribution found for mediapipe". The newest
mediapipe with Intel-Mac wheels is 0.10.21, which pip then picks.

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
| `--source clip.mp4` | Video file instead of the webcam (`--source 1` for a second camera); iPhone portrait videos are rotated upright automatically |
| `--source me.jpg` | One image; prints the JSON result |
| `--json` | Send base64 JSON instead of binary (test both iOS formats) |
| `--log out.csv` | Save per-frame values |
| `python server.py --model lite` | Faster pose model; `heavy` is more accurate |
| `python server.py --no-balance` / `--no-duck` / `--no-bess` | Turn off balance, duck or BESS |
| `python server.py --bess-eyes auto` | BESS: check eyes with the face model (default `manual`: the examiner marks "Eyes opened" in the app; `off`: not scored) |
| `python test_client.py --bess tandem` | Start a BESS test right away (`double`, `tandem`, `single`) |
| `python test_client.py --nondominant right` | BESS non-dominant leg (default left; **n** toggles) |
| `python test_client.py --mute` | No quack sound |
| `python quack.py` | Just play the quack (`sounds/quack.wav`) |
| `python eye_server.py --threshold 0.08` | More sensitive eye alerts (see tuning) |
| `python server.py --record` | Save every session to `recordings/` (the app's Record button does it on demand) |
| `python server.py --no-sway` / `--sway-duration 20` | Turn off sway tests / change their default length |
| `python playback.py recordings/x.jsonl` | Replay a recorded iPhone session (incl. depth) through the same pipeline |
| `python playback.py clip.mov --test quiet@0:20` | Run a sway test on a video file (test at 0 s, 20 s long) |
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
above the floor is measured. Feet count as visible down to a MediaPipe visibility of
0.3: dark trousers and shoes give clearly visible feet scores of only 0.35-0.5. A lift
only counts while the foot stays above the lift level; dipping back below it restarts
the attempt, so the timer is never backdated. The "foot down" level is lower than the "foot up" level,
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

The Balance Error Scoring System on a firm surface: three stances, hands on hips, eyes
closed. Each stance lasts `BessConfig.duration_s` (bess.py), **10 seconds** by default;
the standard BESS uses 20 s, so scores aren't directly comparable to published 20-s
norms. The length is sent in every BESS status as `duration_s`, and the app, the laptop
window and the doctor page use that for their timers. The score is the number of errors, per stance and in total.
"Eyes opened" is marked by the examiner in the app (see "Eyes" below).

| Button | Stance |
|---|---|
| **Feet together** (`double`) | feet side by side, touching |
| **Tandem** (`tandem`) | one foot in front of the other, **non-dominant foot in back** |
| **Single leg** (`single`) | standing on the **non-dominant** leg |

**Running a test:** pick the non-dominant leg, get the person into position, and press
a stance button. The first second records the person's own start position (no get-ready
countdown; `--bess-countdown` sets it), so scoring is relative to how they actually
stand. Then
the stance's scoring time (`duration_s`), and the result. Re-running a stance replaces its score;
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

**Eyes:** at full-body distance the camera can't reliably see whether the eyes are
open, so by default (`--bess-eyes manual`) "Eyes opened" is an error type that the
examiner marks with the app's **Eyes opened +1** button
(`{"type":"bess_mark","error":"eyes_open"}`). `--bess-eyes auto` checks eyes with the
face model on an enlarged crop of the head instead; `--bess-eyes off` leaves eyes out.

Sway (centre-of-mass velocity and area, as in the sway tests) is also measured during
each stance: live as `bess.sway_cm`, and per stance as `sway` in `bess_done` and in the
session summary. The session summary also has `by_stance` (errors by type for each
stance) and `coverage`.

Not automated: the standard BESS rule that a person who can't hold the stance for at
least 5 s scores 10 for that stance. The examiner should apply that by hand.

This is a demo aid, not a validated clinical scoring tool. Compare it against a
trained examiner before relying on it.

## Pose: sway tests (quiet stance, tandem, Romberg)

Postural sway, measured from the camera, for four 30-second tests (length set per
test from the app, or `--sway-duration`):

| Test (`test`) | Stance |
|---|---|
| **Quiet stance** (`quiet`) | feet hip-width apart, arms at sides, eyes open |
| **Tandem stance** (`tandem`) | one foot directly in front of the other, heel to toe |
| **Romberg, eyes open** (`romberg_eo`) | feet together, arms at sides |
| **Romberg, eyes closed** (`romberg_ec`) | same, eyes closed when the countdown ends |

Like BESS: a 5-second countdown to get into position (its last second records the
start position), then the recording. Single-leg tracking is paused during a test, and
BESS and sway tests can't run at the same time.

**What's measured.** Body sway is the movement of an approximate centre of mass, a
point between the hip midpoint and the shoulder midpoint (65% of the way to the hips),
with the person facing the camera:

| Result | Meaning |
|---|---|
| `mean_velocity_cm_s` | sway velocity: path length ÷ time (also `ml_` side-to-side and `ap_` front-back) |
| `path_length_cm` | total distance the centre of mass travelled |
| `area_95_cm2` | sway area: the 95% confidence ellipse |
| `rms_ml_cm`, `rms_ap_cm`, `range_*` | size of the sway in each direction |
| `directional` | directional instability: biggest drift from the start position forward / back / left / right (the person's own left and right), main sway axis (`main_axis_deg`, 0 = side-to-side, 90 = front-back), and `dominant` direction (one axis's RMS 1.5× the other's) |
| `trunk_lean` | sideways trunk lean vs the start posture: max to each side, and how often and how long it went past `lean_limit_deg` (default 10°, set per test). Crossing it sends a live `sway_lean` alert. |
| `errors`, `by_type`, `log` | BESS-style error points (see below) |
| `trail` | the centre-of-mass path at 10 Hz (`[t, ml_cm, ap_cm]`), for plotting |

**Errors (points, lower is better).** The trial runs its full length; every balance
error is one point, scored like BESS: each counts once per event (it has to clear
before it can count again), errors starting within 0.5 s of each other count once,
and a trial scores at most 10.

| Error | Rule |
|---|---|
| Step / stumble | an ankle moves 0.15 leg-lengths from where the feet last settled (left/right label swaps by the model don't count). Putting a lifted foot back into the starting stance isn't another step. |
| Trunk lean past limit | sideways lean past `lean_limit` (default 10°) for 0.3 s |
| Swayed too far | centre of mass more than `drift_limit` (default 10 cm) from the start for 0.2 s |
| Out of position > 5 s | any of the above held for more than 5 s (including feet staying out of the stance) |

The session keeps a score table (`scores`, `total_errors`), like BESS. Live, each pose
result's `sway` has `errors`, `active` and `log`, and each error is also sent as
`{"type": "event", "kind": "sway_error", "error": "step", "label": "Step / stumble", "t": 4.2, "counted": true, "errors": 1}`.
The app's Balance tab works like the BESS screen: a button per test, the live countdown,
timer and "Errors: N" with a "+1" log (the phone vibrates on each point), and the score
table. The detailed sway measurements are in the server log and the `sway_done` result.

After both Romberg tests the session reports eyes-closed ÷ eyes-open ratios for
velocity, path and area (`romberg`), and `positive` if balance was lost with eyes
closed only.

**Depth vs 2D.** The front-back direction is movement toward the camera, which a
plain video can't measure. With the iPhone app streaming **LiDAR** depth (back camera,
iPhone 12 Pro and later Pro models), the torso's distance is read from the depth map
(median over a grid inside the shoulder-hip area), so sway is in real centimetres in
both directions (`"mode": "depth"`). Without depth (`"mode": "2d"`, e.g. a video file),
only side-to-side sway is measured, converted to cm with MediaPipe's metric skeleton;
front-back values, sway area and the direction ratio are `null`. **TrueDepth** (front
camera) works the same way but is only accurate within about 1 m, too close for a
full-body view, so the back camera is the one to use.

Before velocities, the track is resampled to a fixed rate and lightly smoothed (0.2 s
zero-phase moving average) so landmark jitter doesn't count as sway. Sway velocity is
still sensitive to camera noise: compare results recorded the same way (same camera,
distance and light), not against force-plate norms.

**Setup:** camera fixed (tripod or propped up) at about hip height, 2-3 m away, the
whole body including the feet in frame, the person facing it.

This is a demo aid, not a validated clinical measurement.

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

## Eyes: saccades, speed and smoothness

Two timed tests (10 s by default, `--test-duration`), for a close-up camera with the
head still. In `eye_client.py` press **s** or **p** (or `--test saccades|pursuit`):

| Test | What the person does | Speed | Smoothness |
|---|---|---|---|
| **Saccades** (`s`) | looks back and forth between two points as fast as possible | per saccade: peak and mean speed (deg/s), size (deg), duration (ms); median and max over the test | % of target-to-target jumps that landed in **one** saccade (an undershoot followed by a small corrective jump in the same direction counts against it) |
| **Smooth pursuit** (`p`) | follows a slowly moving target, e.g. a finger moving side to side | pursuit speed (deg/s); peak speed of any catch-up saccades | % of the eye's path covered by smooth tracking rather than catch-up saccades, plus catch-up saccades per second |

The window shows a live trace of left/right gaze over the last 4 seconds (saccades in
red), the current eye speed, and the results when the test ends. Each saccade is also
printed as it happens. `--log eyes.csv` records gaze, speed and saccades per frame.

**How it's measured:** both irises' positions between the eye corners (so head
movement isn't eye movement) are averaged and converted to degrees with a standard
eyeball model (12 mm radius, 30 mm eye opening). There's no per-person calibration,
so absolute degrees are approximate (roughly ±15%); comparisons between tests done
the same way are more reliable. Speed is measured over a ~30 ms span so landmark
jitter doesn't look like movement. A saccade is a movement over 80 deg/s
(`--saccade-velocity`), at least 2 deg and 15-250 ms long. Each saccade's peak speed
is then re-measured over the shortest span the frame rate allows. Blinks break the
trace and are never counted as saccades.

**Frame rate matters for speed.** Saccades peak at 300-500 deg/s and last 20-80 ms.
On simulated saccades, measured peak speed was about 55-80% of the true value at
30 fps, about 90% at 60 fps, and 95-100% at 120-240 fps. Saccade size and duration,
and the smoothness scores, are fine at 30 fps. The results warn when the frame rate
is below 50 fps or the head turned.

**For accurate speed, record slow motion.** Record the test as a 120 or 240 fps
slow-motion video on the iPhone and run `python eye_client.py --source clip.mov
--test saccades`. The video's own timestamps are used, so it doesn't matter how fast
the laptop processes it. When streaming live, send frames as JSON with
`"timestamp_ms"` (capture time); `eye_client.py` now does this by default
(`--binary` turns it off). Capture timestamps keep network and processing delays out
of the speed numbers.

This is a demo/screening aid, not a validated clinical measurement.

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
- Eye server tests: `{"type":"eye_test_start","mode":"saccades"|"pursuit","duration":10}`,
  `{"type":"eye_test_cancel"}`, `{"type":"eye_test_status"}` → immediate `ack` or `error`.
  For eye speeds, send frames as JSON with `"timestamp_ms"` = capture time.
- **Depth** (optional, iPhone app): add to the JSON frame `"depth"` (base64), `"depth_format"`
  (`"uint16_mm"` little-endian millimetres, or `"float16_m"`), `"depth_size": [w, h]`,
  `"intrinsics": [fx, fy, cx, cy]` (in depth-map pixels) and `"camera": "lidar"|"truedepth"`.
  The depth map must cover the same view as the image; it's rotated with `rotate`. Each
  pose result then has `"depth": {"camera": "lidar", "torso_m": [x, y, z], "valid_points": 81, ...}`
  (`torso_m` null if the body couldn't be found in it). See `depth.py`.
- Recording: `{"type":"record_start","name":"optional"}` / `{"type":"record_stop"}` save every
  incoming message to `recordings/<name>.jsonl` on the laptop (ack has `"file"`). Replay
  with `python playback.py recordings/<name>.jsonl`.
- Pose server, sway tests: `{"type":"sway_start","test":"quiet"|"tandem"|"romberg_eo"|"romberg_ec","duration":30,"lean_limit":10,"drift_limit":10}`,
  `{"type":"sway_cancel"}`, `{"type":"sway_reset"}`, `{"type":"sway_status"}` → `ack` with
  `"results"` (latest per test), `"scores"`, `"total_errors"`, `"complete"` and `"romberg"`, or `error`.
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
"bess": {"phase": "running", "stance": "tandem", "nondominant": "left", "duration_s": 10.0, "time_left": 6.4,
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

Each pose result also has `"sway"`: `{"phase": "running", "test": "quiet", "mode": "depth",
"time_left": 12.4, "ml_cm": 0.8, "ap_cm": -1.2, "lean_deg": 3.1, "lean_over_limit": false, ...}`.
Sway messages:

```json
{"type": "status", "kind": "sway_started", "test": "quiet", "label": "Quiet stance", "instructions": "...", "countdown_s": 5, "duration_s": 30}
{"type": "status", "kind": "sway_running", "test": "quiet", "mode": "depth", "warnings": []}
{"type": "alert", "kind": "sway_lean", "event": "start", "side": "left", "lean_deg": 12.4, "limit_deg": 10, "t": 7.2}
{"type": "result", "kind": "sway_done", "test": "quiet", "mode": "depth", "lost_balance": false,
 "coverage": 0.99, "metrics": {"mean_velocity_cm_s": 1.2, "path_length_cm": 36.1, "area_95_cm2": 4.3,
 "rms_ml_cm": 0.4, "rms_ap_cm": 0.6, "directional": {"dominant": "front-back", "largest_drift": "forward", ...}, ...},
 "trunk_lean": {"left_deg": 4.1, "right_deg": 3.2, "times_over_limit": 0, ...},
 "session": {"results": {...}, "romberg": {...}}}
{"type": "status", "kind": "sway_failed", "reason": "..."}
```

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
learned, then `"monitoring"`. Each eye result also has
`"movement": {"gaze_deg": [x, y], "velocity_dps": 12.5, "in_saccade": false, "saccade": null}`
and `"eye_test": {"running": true, "mode": "saccades", "time_left": 6.2, "saccades": 7}`.
During a test:

```json
{"type": "event", "kind": "saccade", "amplitude_deg": 20.3, "peak_velocity_dps": 561,
 "mean_velocity_dps": 304, "duration_ms": 67, "direction": "right", "t_start": ..., "t_end": ...}
{"type": "result", "kind": "eye_test_done", "mode": "saccades", "label": "Saccades", "duration_s": 10.0,
 "speed": {"count": 12, "per_second": 1.2,
           "peak_velocity_dps": {"median": 560, "max": 610, "min": 470},
           "mean_velocity_dps": {...}, "amplitude_deg": {...}, "duration_ms": {...}},
 "smoothness": {"score": 86, "primary_saccades": 11, "corrective_saccades": 1, "single_step_pct": 86,
                "peak_velocity_cv": 0.08, "explanation": "..."},
 "quality": {"samples": 1200, "effective_fps": 120.0, "tracked_pct": 99, "warnings": []},
 "saccades": [...]}
```

For `"mode": "pursuit"`, `smoothness` has `score`, `smooth_path_pct`,
`catch_up_saccades`, `catch_up_per_second` and `pursuit_speed_dps`. When something changes, an extra message follows that
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

## iPhone app

`ios/` is a complete SwiftUI app (iOS 17+) for the Modified BESS test: live video
with the skeleton, the three test steps, live errors with an "Eyes opened +1" button,
a summary with reference-range bars, and the error-by-stance parameter table. It
streams the back camera with LiDAR depth to the pose server.

| File | What it is |
|---|---|
| `PosecamApp.swift` | app entry point (wires camera frames to the server connection) |
| `DepthCamera.swift` | video + depth capture (LiDAR), the camera preview and skeleton overlay |
| `BessTestView.swift` | `PosecamClient` (WebSocket) and the BESS screen |

The sway tests (quiet, tandem, Romberg) are still on the server, without an app screen.

**Build it** (Xcode 26, a real iPhone; the simulator has no depth camera). The project
is ready: `ios/Posecam.xcodeproj` (camera / local-network permissions, portrait only,
iOS 17+ already set).

1. Open `ios/Posecam.xcodeproj` in Xcode.
2. Click **Posecam** (top of the file list) → target **Posecam** → **Signing &
   Capabilities** → Team: your Apple ID (Add an Account... if none; a free one works).
   If it says the bundle identifier is taken, change `edu.northeastern.posecam` to
   something unique.
3. Plug in the iPhone (unlock it, tap Trust). On the phone turn on Settings → Privacy &
   Security → **Developer Mode** (it restarts).
4. Choose the iPhone at the top of the Xcode window, press **Run** (▶).
5. First launch only: on the phone, Settings → General → VPN & Device Management → trust
   your Apple ID. Allow camera and local network when the app asks.

**Use it:** on the laptop `python server.py` (it prints the address to type, e.g.
`ws://10.0.5.250:8765`); phone and laptop on the same Wi-Fi. In the app: type the
laptop IP, Connect, Start Camera, prop the phone up 2-3 m away (back camera facing
you), check the green skeleton covers you and the badge says "depth ✓", then tap a
test. **Record** saves the session on the laptop for `playback.py`.

iOS notes: plain `ws://` to a LAN IP needs an App Transport Security exception
(`NSAllowsLocalNetworking`) plus `NSLocalNetworkUsageDescription` in Info.plist.
Use `wss://` for anything outside your LAN, especially with patient video.
