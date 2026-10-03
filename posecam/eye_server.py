"""Eye server: receives video frames from the iPhone app, tracks both irises
with MediaPipe, alerts when one eye drifts out of line with the other, and
measures eye movements: saccades (speed) and smoothness.

Separate from the pose server; run either or both.

Run:
    python eye_server.py                       # ws://0.0.0.0:8766
    python eye_server.py --threshold 0.08      # more sensitive

Frame/ping protocol: see ws_server.py. Send frames as JSON with
"timestamp_ms" (capture time) so eye speeds are measured on the camera's clock.
Also:
  {"type": "recalibrate"}  ->  re-learn this person's normal eye alignment
  {"type": "eye_test_start", "mode": "saccades" | "pursuit", "duration": 10}
  {"type": "eye_test_cancel"} / {"type": "eye_test_status"}

Reply, one per processed frame:
  {"type": "eyes", "frame_id": 42, "face_detected": true,
   "state": "calibrating" | "monitoring", "calibration_progress": 0.6,
   "alerting": false, "deviation_h": 0.01, "deviation_v": 0.0, "score": 0.1,
   "drifting_eye": "left", "direction": "horizontal", "skip_reason": null,
   "head_yaw": 0.03, "eyes": {"right": {...}, "left": {...}}, "inference_ms": 9.1,
   "movement": {"gaze_deg": [3.1, -0.4], "velocity_dps": 12.5, "in_saccade": false, "saccade": null},
   "eye_test": {"running": true, "mode": "saccades", "time_left": 6.2, "saccades": 7, "events": []}}

Plus, when something changes, an extra message right after that frame:
  {"type": "status", "kind": "eye_calibrated"}
  {"type": "alert", "kind": "eye_misalignment", "event": "start",
   "drifting_eye": "left", "direction": "horizontal", "score": 1.8, ...}
  {"type": "alert", "kind": "eye_misalignment", "event": "end", "alert_duration_s": 3.2}
  {"type": "event", "kind": "saccade", "amplitude_deg": 18.2, "peak_velocity_dps": 455,
   "mean_velocity_dps": 290, "duration_ms": 63, "direction": "left", ...}   (during a test)
  {"type": "result", "kind": "eye_test_done", "mode": "saccades",
   "speed": {...}, "smoothness": {"score": 80, ...}, "quality": {...}, "saccades": [...]}
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ws_server import serve

log = logging.getLogger("eye-server")

ALERT_FIELDS = ("drifting_eye", "direction", "score", "deviation_h", "deviation_v",
                "alert_duration_s")


def eye_events(result: dict, frame_id=None) -> list[dict]:
    """Extra messages to send after a frame: calibration done, alert start/end,
    and eye-movement test messages (saccades, results)."""
    return drift_events(result, frame_id) + movement_events(result, frame_id)


def movement_events(result: dict, frame_id=None) -> list[dict]:
    msgs = []
    for ev in (result.get("eye_test") or {}).get("events", []):
        kind = ev["kind"]
        if kind == "saccade":
            msgs.append({"type": "event", "frame_id": frame_id, **ev})
        elif kind == "eye_test_done":
            msgs.append({"type": "result", "frame_id": frame_id, **ev})
            sp, sm = ev["speed"], ev["smoothness"]
            pv = (sp["peak_velocity_dps"] or {}).get("median")
            log.warning("EYE TEST %s done: %d saccades, peak speed %s deg/s, smoothness %s%s",
                        ev["mode"], sp["count"], pv, sm["score"],
                        "".join(f"\n    warning: {w}" for w in ev["quality"]["warnings"]))
        else:
            msgs.append({"type": "status", "frame_id": frame_id, **ev})
            log.info("eye test %s", kind.replace("eye_test_", ""))
    return msgs


def drift_events(result: dict, frame_id=None) -> list[dict]:
    event = result.get("event")
    if not event:
        return []
    if event == "calibrated":
        log.info("eye baseline calibrated")
        return [{"type": "status", "kind": "eye_calibrated", "frame_id": frame_id}]
    msg = {"type": "alert", "kind": "eye_misalignment", "event": event, "frame_id": frame_id,
           **{k: result[k] for k in ALERT_FIELDS if result.get(k) is not None}}
    if event == "start":
        log.warning("ALERT: %s eye drifting (%s, score %.2f)",
                    msg.get("drifting_eye"), msg.get("direction"), msg.get("score", 0))
    else:
        log.info("alert cleared after %.1fs", msg.get("alert_duration_s", 0))
    return [msg]


def main():
    p = argparse.ArgumentParser(description="MediaPipe eye-alignment WebSocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--threshold", type=float, default=0.10,
                   help="horizontal drift that triggers an alert, in eye-widths")
    p.add_argument("--vthreshold", type=float, default=0.08,
                   help="vertical drift that triggers an alert, in eye-widths")
    p.add_argument("--hold", type=float, default=0.5,
                   help="seconds the drift must last before alerting")
    p.add_argument("--calibration", type=float, default=2.0,
                   help="seconds of steady looking used to learn the baseline")
    p.add_argument("--saccade-velocity", type=float, default=80.0,
                   help="eye speed (deg/s) above which movement counts as a saccade")
    p.add_argument("--test-duration", type=float, default=10.0,
                   help="default length of a saccade / pursuit test, in seconds")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from eye_tracker import EyeAnalyzer, MonitorConfig, ensure_face_model
    ensure_face_model()  # download once up front, not per connection

    from eye_movement import MovementConfig

    def factory():
        return EyeAnalyzer(MonitorConfig(h_threshold=args.threshold, v_threshold=args.vthreshold,
                                         hold_s=args.hold, calibration_s=args.calibration),
                           MovementConfig(saccade_velocity=args.saccade_velocity,
                                          test_duration_s=args.test_duration))

    try:
        asyncio.run(serve(args.host, args.port, factory, result_type="eyes",
                          events_fn=eye_events, name="eye server"))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
