# Live demo integration (branch `integration`)

iPhone → posecam → replay service → fal (SAM 3D Body) → doctor dashboard, for the demo
protocol: **3 back-to-back BESS trials of 10 s each (feet together, tandem, single leg), ~15 s
apart, one session.** Research demo, not a diagnostic device.

## Components and ports

| Component | Where | Port | What it does |
|---|---|---|---|
| iOS test screen | `posecam/ios/BessTestView.swift` (Liyan) | – | Three BESS buttons; sends camera JPEGs + `bess_start` over WebSocket |
| posecam pose server | `posecam/server.py` | **8765** (ws) | MediaPipe 2D pose, balance, BESS scoring (countdown 5 s + `--bess-duration` s) |
| └ replay forwarder | `posecam/replay_forward.py` | → 8017 | During each trial, from 1 s before scoring (baseline): uniform frames at 1 fps (double, tandem) or 1.5 fps (single leg), plus 10 fps bursts (±0.5 s, max 10 burst frames per trial) for foot events only (touchdown, step/stumble/fall, heel/forefoot lift); `/end` with all events and the BESS result |
| └ live publisher | `posecam/replay_publish.py` | → 8017 | ~10 Hz summaries (2D landmarks, balance, foot heights, trunk angle, BESS timer/errors/scores) + every event → dashboard |
| posecam eye server (optional) | `posecam/eye_server.py` | 8766 (ws) | Eye tracking; not part of the balance protocol |
| replay service | `replay/service/app.py` (FastAPI) | **8017** (http) | Sessions, fal queue, staged replay publishing, dashboard, reports |
| └ scheduler | `replay/service/scheduler.py` | – | One queue for all trials in a session, 3 frames in flight (fal's measured parallelism): error bursts (+ each trial's first 4 coarse "anchor" frames for the floor fit) first, then coarse-to-fine uniform frames by trial deadline; frames after their trial's deadline go last |
| └ live hub | `replay/service/live.py` | – | Session store + server-sent-events feed for the dashboard |
| fal | `fal-ai/sam-3/3d-body` | https | ~4.4 s per frame, $0.018 per call, cached by content hash in `replay/data/fal_cache` |
| doctor dashboard | `replay/dashboard/index.html` | 8017 `/dashboard/{session}` | Live skeleton, stance / timer / errors / trunk angle cards, event feed, one replay card per trial |
| 3D viewer | `replay/viewer/` | 8017 `/viewer/?src=…` | Instant replay, coverage badge (`done/received frames · stage`) |
| trial report | `replay/service/report.py` | 8017 `/replay/{trial}/report` | Printable one-trial report |
| session report | `replay/service/session_report.py` | 8017 `/session/{session}/report` | One-page BESS-style summary per stance |
| public tunnel (optional) | `cloudflared` | → 8017 | Public dashboard URL for a remote doctor |

## Run

```bash
./run_demo.sh --mock --warmup          # rehearsal: fal mocked (free), 4.4 s/frame, 3 parallel, one 150 s straggler
./run_demo.sh --live --warmup          # real fal: asks before starting and before the warm-up call
./run_demo.sh --live --warmup --max-usd 1.20   # + hard stop: no fal call beyond $1.20 (incl. warm-up)
./run_demo.sh --live --tunnel --eyes   # + public dashboard URL, + eye server
```

`run_demo.sh` starts the replay service (8017) and posecam (8765, with `--replay-url`,
`--publish-url`, `--session-id`, `--bess-duration 10`, `--patient-height-cm 185`), creates a
session, prints the dashboard URL, the session report URL and the laptop IP for the phone.
Logs go to `replay/data/logs/`. Ctrl-C stops everything.

Without the phone, rehearse with a recorded video (real-time JSON frames with `timestamp_ms`):

```bash
cd replay && uv run tools/rehearse_protocol.py --session <id> --shots data/reports/rehearsal
```

It uses `replay/data/protocol.mov` if it exists. Otherwise it picks the three 10 s segments of
`data/IMG_9691.mov` that best match each stance (from the scout landmarks), streams them with
15 s gaps, waits for the replays, prints the measured timeline and saves dashboard screenshots.

Tests: `cd replay && uv run pytest` (113) · `cd posecam && python -m pytest` (128).
Dashboard screenshots of a live page: `node replay/tools/screenshot.mjs <url> out.png [wait_ms]`.

## Data flow for one trial

1. Phone presses **Double** → `{"type":"bess_start","stance":"double"}` on ws 8765. posecam
   starts the 5 s countdown; the forwarder opens trial `bess-double-<ms>` (one clock per trial:
   the phone's `timestamp_ms` if frames carry it, else the server's receive time).
2. Scoring starts (`bess_running`): the forwarder sends the last countdown second (start
   position), then 1.5 fps frames (`POST /replay/{trial}/frame`, `kind=uniform`, `session=…`).
   Each foot event (touchdown, step/stumble/fall, heel/forefoot lift) adds a 10 fps burst
   (`kind=burst`), up to 10 burst frames per trial. Hands off hips, hip angle, eyes and out of
   position are reported in `/end` but get no burst frames.
3. In parallel the publisher pushes summaries and events (`POST /live/{session}/push`); the
   dashboard gets them over `GET /live/{session}/stream` (SSE).
4. The replay service queues each frame for fal (scheduler order above) and stores results per
   trial in `replay/data/trials/<trial>/frames/`.
5. `bess_done` → `POST /replay/{trial}/end` (events, BESS errors, patient height, session).
   The trial's deadline is set to end + 30 s (`REPLAY_DEADLINE_S`).
6. Publishing, each an atomic bundle swap in `replay/data/trials/<trial>/`:
   - **error** replay as soon as the burst and anchor frames are done (≥ 6 frames; if one of
     them straggles, at end + 15 s once 75 % are done),
   - **full** replay when every frame is done, or **deadline** replay at end + 30 s with whatever
     is done (the viewer interpolates the gaps; the badge shows the coverage),
   - **update** when late frames finish afterwards (at most every 10 s; logged as late frames).
   The dashboard card shows "processing x/y frames · ETA", then loads the viewer in
   instant-replay mode, plus the report link. Updates appear as a "Load update" button so the
   doctor's view is not reset.
7. `/session/{session}/report`: per stance: BESS errors (posecam), side-to-side sway, minimum
   margin, time outside the base of support, quality and frame coverage, links to each replay
   and report, and the note "shortened 10 s demo protocol (standard BESS uses 20 s)". Mock or
   synthetic data gets a yellow banner saying the numbers are not meaningful.

Floor fit: frames within 1 s of a foot event (step, lift, touchdown, fall) are not used.
Posecam errors that say nothing about the feet (hands off hips, hip angle, out of position,
eyes) no longer remove floor-fit frames. Before this fix, a trial with many of them had no
floor fit and fell back to an unaligned camera-frame replay.

## Frame budget (why the caps)

fal reconstructs ~0.7 frames/s (3 in parallel × ~4.4 s). One trial every ~30 s means only
~20 frames per trial can finish before its deadline. Per trial: 11 uniform frames for double
and tandem (1 fps × 11 s), 17 for single leg (1.5 fps), plus ≤ 10 burst frames on foot events
only. That is at most ~70 frames per session, fewer when a stance has no foot events.
The first mock rehearsal without the caps sent 87 frames for one trial and missed every deadline.

The fal client's own concurrency limit now follows `REPLAY_FAL_WORKERS` (3). Before this it
defaulted to 2, so earlier rehearsals ran 2 frames at once, not 3.

Spend cap: `--max-usd X` (env `REPLAY_MAX_USD`) is a hard stop in the replay service. Once
X / $0.018 calls (warm-up and retries included) are billed or in flight, further frames are
refused, not sent. Replays then publish with the frames already done.

## Measured timeline (mock rehearsal, 2026-10-03, previous budget: 1.5 fps, bursts on every error, 45 s deadline, 2 fal slots)

`./run_demo.sh --mock --warmup` + `tools/rehearse_protocol.py`: IMG_9691 segments streamed
in real time at 720 px, fal mocked at 4.4 s ± 15 % per frame, 3 in parallel, the 12th call
(a burst frame of trial 1) held 150 s. Seconds after each trial's end:

| Trial | Frames (burst) | Error replay | Deadline replay | Updates (late frames) |
|---|---|---|---|---|
| Feet together | 26 (10) | 16.0 s (straggler fallback) | 45.8 s, 17/26 frames | 117–160 s → 26/26 |
| Tandem | 26 (10) | 20.7 s | 45.9 s, 14/26 frames | 122–144 s → 26/26 |
| Single leg | 26 (10) | 18.3 s | 46.1 s, 23/26 frames | 57–116 s → 26/26 |

Total 78 frames per session → **≈ $1.40 live** at $0.018/call (+ $0.018 warm-up).
Deadline coverage is lower for trials 1–2 because the straggler holds one of the three workers
for 150 s. Screenshots of every stage: `replay/data/reports/rehearsal/` (local, not committed).

## Plan B: iPhone as Continuity Camera (no iOS code)

If the iOS app can't send frames yet, use the iPhone as a Mac webcam. The rehearsal tool streams
it into posecam exactly like the phone would (JSON frames with `timestamp_ms`).

Setup (once):
1. iPhone and Mac on the same Apple ID, Wi-Fi and Bluetooth on (macOS 13+, iOS 16+).
   iPhone: Settings › General › AirPlay & Continuity › Continuity Camera on.
2. Mac: System Settings › Privacy & Security › Camera: allow the app you run the tool from
   (Terminal, iTerm or VS Code). Without this, every camera fails with "Cannot use … Camera".
3. Mount the iPhone on a tripod, **locked**, rear camera facing the patient, still, ~2.5–3 m
   away, whole body including feet in frame. Portrait mounting is fine (use `--rotate`).

Run:
```bash
./run_demo.sh --live --warmup                      # terminal 1: prints the session id
cd replay
uv run tools/rehearse_protocol.py --list-cameras   # e.g. [1] Stefan's iPhone 17 Camera
uv run tools/rehearse_protocol.py --session <id> --camera auto --rotate 90 --trigger key
```
- `--camera auto` picks the iPhone (never its Desk View), else the first camera; or pass an index or name.
- `--rotate 0|90|180|270` rotates clockwise before sending (portrait mount: usually 90 or 270;
  check the dashboard skeleton is upright).
- `--trigger key`: press Enter to start each trial (feet together → tandem → single leg).
  `--trigger timer`: first trial after `--first-delay` s, then `--gap` s (15) after each trial ends.
- After the last trial it waits for the replays and prints the timeline, like the clip rehearsal.

Checked on this laptop: the device list shows "Stefan's iPhone 17 Camera" and `auto` picks it.
Capture itself could not be tested from this session: macOS refused camera access to the app
running it (step 2).

## A. Real-iPhone run (Liyan's iOS test screen)

1. Laptop and iPhone on the same Wi-Fi (no client isolation; a phone hotspot works).
2. `./run_demo.sh --live --warmup` (or `--mock` first). Note the printed **Laptop IP** and port **8765**.
3. In the app (BessTestView): enter the laptop IP, Connect. The camera pipeline must call
   `posecam.sendFrame(jpeg)` for every frame (portrait, full frame, never cropped; ≥ 720 px wide).
4. Phone on a tripod ~2.5–3 m away at hip height, whole body including feet in frame, still camera.
5. Open the dashboard URL on the laptop. Check the live skeleton moves and the phase card says
   "Between trials".
6. Press **Feet Together**; patient gets into position during the 5 s countdown, hands on hips,
   stays 10 s. Wait ~15 s. **Tandem**. Wait ~15 s. **Single Leg** (non-dominant foot up).
7. Watch each replay card: processing → error replay → full/deadline replay (~30 s after each trial).
8. Open the session report; print or save as PDF.

## B. Zoom dry run (doctor screen-shares the dashboard)

1. Laptop: `./run_demo.sh --live --tunnel` (`brew install cloudflared` once). Send the doctor
   the printed **public dashboard** URL (it is unauthenticated; share only for the demo and stop
   the tunnel afterwards).
2. Doctor opens it in Chrome, full screen (⌃⌘F), and shares that **window** in Zoom
   (not the whole screen; tick "Optimize for video clip" off, the page is mostly static).
3. The page is readable at 1280×720: big cards, high contrast, one 3D view at a time.
4. Run the protocol as in A. The doctor narrates live cards during each trial and the replay
   cards afterwards ("▶ Instant replay" in the viewer, "Report ↗" per trial).
5. Backup if fal or the network is slow: "◀ Previous trial (cached)" opens the stored demo replay.
6. End with the session report link.

## What's still missing

- The iOS camera pipeline calling `sendFrame` (BessTestView only has the WebSocket client);
  frames from the phone carry no `timestamp_ms` yet (server receive time is used).
  Plan B (Continuity Camera) avoids this; its capture still needs one test with camera permission.
- No real-iPhone end-to-end run yet; the mock rehearsal uses the recorded clip at 720 px.
- Mock replays use the nearest stored pose per frame, so their numbers are not meaningful
  (the dashboard and viewer say "fal MOCK").
- Phone gravity per frame is not sent (`gravity` field), so floor tilt comes from SAM's feet only.
- The tunnel URL has no authentication.
- BESS on the clip produces "hands off hips" errors because the recording was not hands-on-hips.
- Session data lives in memory plus `replay/data/trials`; a service restart keeps sessions but
  loses in-progress trial state.
