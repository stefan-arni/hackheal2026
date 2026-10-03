# Demo runbook (one page)

Protocol: 3 BESS trials of 10 s each, in this order: feet together → tandem → single leg.
Each has a 5 s countdown and they run ~15 s apart, all in one session. Research demo, not a diagnostic device.

## Roles
- **Operator** (laptop): starts the services and presses Enter to start each trial.
- **Patient**: stands in front of the iPhone.
- **Doctor**: watches the dashboard, live or over Zoom, and narrates.

## Start order (laptop, from the repo root)
1. Terminal 1, replay service + posecam:
   `./run_demo.sh --live --warmup --max-usd 1.20` (answer `y` twice).
   It prints the **session id**, the **dashboard** and **session report** URLs, and the laptop IP.
2. iPhone: locked, on a tripod 2.5–3 m away at hip height, rear camera facing the patient.
   Whole body including feet in frame; portrait is fine.
3. Terminal 2, camera into posecam (Continuity Camera, see INTEGRATION.md "Plan B"):
   `cd replay && uv run tools/rehearse_protocol.py --session <id> --camera auto --rotate 90 --trigger key --shots data/reports/live`
   Check the dashboard skeleton is upright. If not, restart with `--rotate 270` (or 0).
4. Open the dashboard: `http://localhost:8017/dashboard/<id>`.
   On the same Wi-Fi: `http://<laptop IP>:8017/dashboard/<id>`.
   Remote doctor: add `--tunnel` in step 1 and send the printed public URL.

## Patient instructions (operator reads them before each Enter)
1. **Feet together:** "Feet side by side, touching. Hands on hips, eyes open, look straight ahead. Stay still."
2. **Tandem:** "Left foot directly in front of the right, heel touching toe. Hands on hips. Stay still."
3. **Single leg:** "Stand on your right leg and lift the left foot. Hands on hips. Stay still."
- Each trial: get in position during the 5 s countdown, hold for 10 s, relax.
  Wait ~15 s between trials, until the operator says "next".

## Doctor clicks
- During each trial: watch the big cards (stance, timer, BESS errors, trunk angle) and the event feed.
- After each trial, the replay card goes "processing x/y frames · ETA", then the **error replay**
  (~20 s), then the **full/deadline replay** (≤ 30 s after the trial ends).
  - The 3D viewer opens in instant-replay mode. **▶ Instant replay** replays the biggest event.
  - **Report ↗** opens the printable trial report.
  - **⟳ Load update** appears when late frames arrive.
- At the end: **Session report ↗** (top right) shows one page per stance; print it or save it as PDF.

## Fallbacks
| Trigger | Do this |
|---|---|
| Skeleton freezes or the badge says "reconnecting…" for > 5 s (stream died) | Ctrl-C terminal 2 and restart step 3. If the iPhone camera won't open ("Cannot use … Camera"): unlock and re-lock the phone, keep it still, or use `--camera 0` (MacBook camera). Restart the trial (Enter). |
| No phone camera at all | Terminal 2: `uv run tools/rehearse_protocol.py --session <id>` streams the recorded clip as if live. |
| A replay is not ready when the doctor needs it | Click **◀ Previous trial (cached)** for the stored replay of an earlier run. |
| fal budget reached ("session cap" in replay.log) | Replays publish with the frames already done. Continue; say "partial reconstruction". |
| Everything dies (laptop, network, services) | Play the MP4: `replay/data/reports/IMG_9691/instant_replay_stepdown.mp4` (QuickTime, full screen). |

Logs: `replay/data/logs/{replay,posecam}.log`. Ctrl-C in terminal 1 stops everything.
