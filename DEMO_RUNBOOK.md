# Demo runbook (one page)

Protocol: 3 BESS trials of 10 s (feet together → tandem → single leg), 8 s countdown each,
~15 s apart, one session. Research demo, not a diagnostic device.

## Roles
- **Patient**: stands ~2.5 m in front of the phone.
- **Phone person**: holds or mounts the iPhone running the posecam app and taps the stance buttons.
  Stands next to the patient during single leg (safety).
- **Operator** (laptop): starts the services and watches the logs.
- **Doctor**: shares the dashboard in Zoom (Share › the Chrome **window** › tick
  **"Optimize for video clip"**) and narrates.

## Start order
1. Laptop, repo root: `./run_demo.sh --live --warmup --max-usd 1.20` (8 s countdown is the default; answer `y` twice).
   Note the printed **session id** and **laptop IP**. Phone and laptop must be on the same Wi-Fi.
2. iPhone (**Pro model with LiDAR**), posecam app: Laptop IP = `<laptop IP>` (the app adds `:8765`) › **Connect** (dot turns green)
   › **Start Camera** › Non-dominant leg = **right**.
3. Phone in **portrait**, ~2.5 m away at hip height, whole body including feet in frame, light in
   front of the patient (no window behind them).
4. Open these before the demo:
   - dashboard `http://localhost:8017/dashboard/<id>` (Zoom: the `--tunnel` public URL, if remote)
   - session report `http://localhost:8017/session/<id>/report`
   - backup replay `http://localhost:8017/viewer/?src=/cache/demo_live/&jump=step`
   - backup MP4 in QuickTime: `replay/data/reports/IMG_9691/instant_replay_stepdown.mp4`

## Each trial
The phone person taps the stance and steps out of frame. The patient gets into position during
the 8 s countdown, holds 10 s, then relaxes. Wait ~15 s, then the next stance.
1. **Feet together**: feet side by side, touching. Hands on hips, eyes closed.
2. **Tandem**: left foot in front, right foot behind, heel touching toe. Hands on hips, eyes closed.
3. **Single leg**: stand on the right leg, left foot up. Hands on hips, eyes closed. Make **one
   deliberate foot-down** around second 5, then lift again.
- If the patient opens their eyes, the phone person taps **Eyes opened +1**.

## What the doctor sees
- During a trial: the live skeleton, stance, timer counting down, BESS errors and trunk angle cards, the event feed.
- After a trial: its card shows "processing x/y frames · ETA", then the **error replay** (~15–20 s after the
  trial ends, if there was a foot event), then the **full or deadline replay** (≤ 30 s). The 3D viewer opens
  in instant replay; **Report ↗** opens the trial report.
- At the end: **Session report ↗**, one printable page with the three stances.

## Fallbacks
| Trigger | Do this |
|---|---|
| App won't connect, no skeleton on the dashboard, or skeleton frozen > 5 s (stream died) | **Plan B**, iPhone as Continuity Camera (INTEGRATION.md): `cd replay && uv run tools/rehearse_protocol.py --session <id> --camera auto --rotate 90 --trigger key`, then Enter starts each stance |
| Plan B fails ("Cannot use … Camera": allow Camera for the terminal in System Settings; lock the phone) | The recorded clip: `uv run tools/rehearse_protocol.py --session <id>` |
| A replay isn't ready when the doctor needs it | **◀ Previous trial (cached)** on the dashboard |
| Budget cap reached (`session cap` in `replay/data/logs/replay.log`) | Replays publish with the frames already done. Carry on and say "partial reconstruction" |
| Everything dies | Play the MP4 full screen |

Logs: `replay/data/logs/{replay,posecam}.log`. Ctrl-C in the run_demo terminal stops everything.
