"""Eye test client: stands in for the iPhone app for eye tracking.

Grabs frames from your webcam (or a video file), sends them to the eye server,
draws the tracked irises, and:
  - flashes + beeps when one eye drifts out of line with the other
  - runs eye-movement tests and shows a live gaze trace:
      s  saccade test: look back and forth between two points as fast as you can
      p  pursuit test: follow a slowly moving target (e.g. a finger) with your eyes
    Each test runs 10 s (--test-duration) and reports speed and smoothness.

Usage:
    python eye_server.py                        # terminal 1
    python eye_client.py                        # terminal 2: webcam -> eye server
    python eye_client.py --local                # skip the server, run MediaPipe directly
    python eye_client.py --source clip.mov      # video file (uses the file's own timestamps)
    python eye_client.py --test saccades        # start a test right away
    python eye_client.py --log eyes.csv         # log per-frame values

For accurate saccade SPEED, record a slow-motion video (120/240 fps) on the iPhone
and run it with --source: the file's timestamps are used, so processing speed
doesn't matter. A live webcam is limited to its frame rate (often 30 fps).

Keys: q / Esc quit, c recalibrate drift baseline, s saccade test, p pursuit test,
x cancel test. Camera close to the face, head still.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from eye_movement import TEST_MODES
from eye_tracker import draw_eyes
from ws_client import IMAGE_EXTS, RemoteClient, open_source, resize_to_width


def beep():
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND)   # non-blocking
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class RemoteEyes:
    def __init__(self, url: str, use_json: bool):
        self.client = RemoteClient(url, "eyes", use_json)

    def __call__(self, frame, quality: int = 80, timestamp_ms: float | None = None) -> dict:
        return self.client(frame, quality, timestamp_ms)

    def recalibrate(self):
        self.client.send_json({"type": "recalibrate"})

    def command(self, msg: dict):
        self.client.send_json(msg)

    def take_replies(self) -> list[dict]:
        return self.client.take_replies()

    def close(self):
        self.client.close()


class LocalEyes:
    """Runs MediaPipe in-process (no server)."""

    def __init__(self, threshold: float, vthreshold: float, hold: float, test_duration: float,
                 saccade_velocity: float):
        from eye_movement import MovementConfig
        from eye_tracker import EyeAnalyzer, MonitorConfig
        self.analyzer = EyeAnalyzer(MonitorConfig(h_threshold=threshold, v_threshold=vthreshold,
                                                  hold_s=hold),
                                    MovementConfig(test_duration_s=test_duration,
                                                   saccade_velocity=saccade_velocity))
        self._replies: list[dict] = []

    def __call__(self, frame, quality: int = 80, timestamp_ms: float | None = None) -> dict:
        t = timestamp_ms / 1000.0 if timestamp_ms is not None else None
        return self.analyzer.process_bgr(frame, t=t)

    def recalibrate(self):
        self.analyzer.recalibrate()

    def command(self, msg: dict):
        self._replies += self.analyzer.handle_command(msg)

    def take_replies(self) -> list[dict]:
        r, self._replies = self._replies, []
        return r

    def close(self):
        self.analyzer.close()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def describe(r: dict) -> str:
    if not r.get("face_detected"):
        text = "no face"
    elif r["state"] == "calibrating":
        text = f"calibrating {int(r.get('calibration_progress', 0) * 100)}%"
    elif "score" in r:
        text = (f"drift h {r['deviation_h']:+.3f} v {r['deviation_v']:+.3f} score {r['score']:.2f}"
                + ("  ALERT" if r["alerting"] else ""))
    else:
        text = "calibrated"
    if r.get("skip_reason") and r["skip_reason"] != "no_face":
        text += f"  [{r['skip_reason']}]"
    mv = r.get("movement") or {}
    if mv.get("gaze_deg"):
        text += f"  gaze {mv['gaze_deg'][0]:+5.1f},{mv['gaze_deg'][1]:+5.1f} deg"
        if mv.get("velocity_dps") is not None:
            text += f"  {mv['velocity_dps']:5.0f} deg/s"
    test = r.get("eye_test") or {}
    if test.get("running"):
        text += f"  [{test['mode']} {test['time_left']:.1f}s, {test['saccades']} saccades]"
    return text


def fmt_stat(s, unit=""):
    if not s:
        return "-"
    return f"{s['median']:.0f}{unit} (max {s['max']:.0f})"


def result_lines(res: dict) -> list[str]:
    sp, sm, q = res["speed"], res["smoothness"], res["quality"]
    lines = [f"{res['label']} test ({res['duration_s']:.1f} s, {q['effective_fps']:.0f} fps)"]
    if res["mode"] == "saccades":
        lines += [
            f"Saccades: {sp['count']}",
            f"Peak speed: {fmt_stat(sp['peak_velocity_dps'], ' deg/s')}",
            f"Mean speed: {fmt_stat(sp['mean_velocity_dps'], ' deg/s')}",
            f"Size: {fmt_stat(sp['amplitude_deg'], ' deg')}   Duration: {fmt_stat(sp['duration_ms'], ' ms')}",
            f"Smoothness: {sm['score'] if sm['score'] is not None else '-'}% landed in one jump "
            f"({sm['corrective_saccades']} corrective)",
        ]
    else:
        ps = sm["pursuit_speed_dps"]
        lines += [
            f"Smoothness: {sm['score'] if sm['score'] is not None else '-'}% of path smooth",
            # f"Catch-up saccades: {sm['catch_up_saccades']} "
            # f"({sm['catch_up_per_second'] if sm['catch_up_per_second'] is not None else '-'}/s)",
            # f"Pursuit speed: {fmt_stat(ps, ' deg/s')}",
            # f"Saccade peak speed: {fmt_stat(sp['peak_velocity_dps'], ' deg/s')}",
        ]
    lines += [f"! {w}" for w in q["warnings"]]
    return lines


def announce(r: dict):
    ev = r.get("event")
    if ev == "calibrated":
        print(">> eye baseline calibrated")
    elif ev == "start":
        print(f">> ALERT: {r.get('drifting_eye')} eye drifting "
              f"({r.get('direction')}, score {r.get('score')})")
        beep()
    elif ev == "end":
        print(f">> alert cleared after {r.get('alert_duration_s')}s")
    for e in (r.get("eye_test") or {}).get("events", []):
        k = e["kind"]
        if k == "eye_test_started":
            print(f">> {TEST_MODES[e['mode']]} test: {e['duration_s']:.0f} s, head still")
        elif k == "saccade":
            print(f"   saccade {e['direction']:>5}: {e['amplitude_deg']:5.1f} deg, peak "
                  f"{e['peak_velocity_dps']:4.0f} deg/s, mean {e['mean_velocity_dps']:4.0f} deg/s, "
                  f"{e['duration_ms']:3.0f} ms")
        elif k == "eye_test_done":
            print(">> " + "\n   ".join(result_lines(e)))
        elif k == "eye_test_cancelled":
            print(">> test cancelled")


# --------------------------------------------------------------------------- #
# Drawing: gaze trace + test status / results
# --------------------------------------------------------------------------- #

class TracePlot:
    """Horizontal gaze over the last few seconds; saccades in red."""

    def __init__(self, seconds: float = 4.0, range_deg: float = 25.0):
        self.seconds, self.range = seconds, range_deg
        self.pts: deque = deque()
        self.result: dict | None = None
        self.result_until = 0.0

    def add(self, t: float, r: dict):
        mv = r.get("movement") or {}
        g = mv.get("gaze_deg")
        self.pts.append((t, g[0] if g else None, bool(mv.get("in_saccade")) or bool(mv.get("saccade"))))
        while self.pts and t - self.pts[0][0] > self.seconds:
            self.pts.popleft()
        for e in (r.get("eye_test") or {}).get("events", []):
            if e["kind"] == "eye_test_done":
                self.result, self.result_until = e, time.time() + 12
            elif e["kind"] == "eye_test_started":
                self.result = None

    def draw(self, frame, r: dict):
        h, w = frame.shape[:2]
        ph, x0, x1 = 110, 15, w - 15
        y0 = h - 70 - ph
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y0 + ph), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        mid = y0 + ph // 2
        cv2.line(frame, (x0, mid), (x1, mid), (90, 90, 90), 1)
        cv2.putText(frame, "gaze left/right (deg), last 4 s", (x0 + 6, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        if self.pts:
            t_end = self.pts[-1][0]
            xs = [x for _, x, _ in self.pts if x is not None]
            # center the plot on what's visible (display only; measurements use differences)
            c = (max(xs) + min(xs)) / 2 if xs else 0.0
            prev = None
            for t, x, sac in self.pts:
                if x is None:
                    prev = None
                    continue
                px = int(x1 - (t_end - t) / self.seconds * (x1 - x0))
                py = int(mid - max(-1, min(1, (x - c) / self.range)) * (ph // 2 - 6))
                if prev:
                    cv2.line(frame, prev, (px, py), (60, 60, 255) if sac else (120, 230, 120), 2)
                prev = (px, py)
        mv = r.get("movement") or {}
        if mv.get("velocity_dps") is not None:
            cv2.putText(frame, f"{mv['velocity_dps']:.0f} deg/s", (x1 - 110, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255) if mv.get("in_saccade")
                        else (200, 200, 200), 1, cv2.LINE_AA)

        test = r.get("eye_test") or {}
        if test.get("running"):
            label = TEST_MODES[test["mode"]]
            hint = ("look back and forth fast" if test["mode"] == "saccades"
                    else "follow the moving target")
            txt = f"{label}: {test['time_left']:.1f}s  ({hint})  saccades {test['saccades']}"
            _label(frame, txt, 15, 64, (0, 220, 255))
        elif self.result and time.time() < self.result_until:
            lines = result_lines(self.result)
            bw = min(w - 30, 560)
            bh = 26 + 22 * len(lines)
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 50), (15 + bw, 50 + bh), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            for i, ln in enumerate(lines):
                color = (0, 200, 255) if ln.startswith("!") else ((255, 255, 255) if i else (0, 220, 255))
                cv2.putText(frame, ln, (25, 72 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
        return frame


def _label(frame, s, x, y, color):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + 8), (20, 20, 20), -1)
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Test client for the eye server")
    p.add_argument("--url", default="ws://localhost:8766")
    p.add_argument("--source", default="0", help="webcam index, video path, or image path")
    p.add_argument("--local", action="store_true", help="run MediaPipe locally, no server")
    p.add_argument("--threshold", type=float, default=0.10, help="--local only: horizontal threshold")
    p.add_argument("--vthreshold", type=float, default=0.08, help="--local only: vertical threshold")
    p.add_argument("--hold", type=float, default=0.5, help="--local only: seconds before alerting")
    p.add_argument("--saccade-velocity", type=float, default=80.0,
                   help="--local only: deg/s above which movement is a saccade")
    p.add_argument("--test", choices=list(TEST_MODES), help="start this eye-movement test right away")
    p.add_argument("--test-duration", type=float, default=10.0, help="test length in seconds")
    p.add_argument("--binary", action="store_true",
                   help="send raw JPEG instead of JSON (no capture timestamps: speeds less accurate)")
    p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)   # JSON is the default now
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100")
    p.add_argument("--width", type=int, default=960,
                   help="resize frames to this width (eyes need detail; 960+ recommended)")
    p.add_argument("--log", help="write per-frame eye values to this CSV")
    p.add_argument("--no-window", action="store_true", help="print results instead of showing video")
    args = p.parse_args()

    eyes = (LocalEyes(args.threshold, args.vthreshold, args.hold, args.test_duration,
                      args.saccade_velocity) if args.local
            else RemoteEyes(args.url, not args.binary))

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"could not read {args.source}")
        result = eyes(resize_to_width(frame, args.width), args.quality)
        print(json.dumps({k: result.get(k) for k in ("face_detected", "head_yaw", "eyes", "skip_reason")},
                         indent=2))
        print("(a single image can't trigger an alert or measure movement)")
        if not args.no_window:
            cv2.imshow("eyes", draw_eyes(frame, result))
            cv2.waitKey(0)
        eyes.close()
        return

    is_file = not args.source.isdigit()
    cap = open_source(args.source)
    if is_file:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        print(f">> video file: {fps:.0f} fps, using the file's timestamps")
    if args.test:
        eyes.command({"type": "eye_test_start", "mode": args.test, "duration": args.test_duration})

    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["frame", "time_s", "face_detected", "state", "deviation_h", "deviation_v",
                         "score", "alerting", "drifting_eye", "direction", "skip_reason", "head_yaw",
                         "gaze_x_deg", "gaze_y_deg", "velocity_dps", "in_saccade", "saccade_amp_deg",
                         "saccade_peak_dps", "test_mode"])

    plot = TracePlot()
    t_start, n, fps_live = time.time(), 0, 0.0
    file_t0 = time.time() * 1000
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # capture time: the file's own clock for videos, wall clock for a webcam
            ts_ms = (file_t0 + cap.get(cv2.CAP_PROP_POS_MSEC)) if is_file else time.time() * 1000
            frame = resize_to_width(frame, args.width)

            t0 = time.perf_counter()
            r = eyes(frame, args.quality, ts_ms)
            rtt = (time.perf_counter() - t0) * 1000
            fps_live = 0.9 * fps_live + 0.1 * (1000 / max(rtt, 1))
            n += 1
            announce(r)
            plot.add(ts_ms / 1000, r)
            for rep in eyes.take_replies():
                if rep["type"] == "error":
                    print(f">> {rep['error']}")

            mv = r.get("movement") or {}
            sac = mv.get("saccade") or {}
            if writer:
                g = mv.get("gaze_deg") or [None, None]
                writer.writerow([n, round(time.time() - t_start, 3), r.get("face_detected"),
                                 r.get("state"), r.get("deviation_h"), r.get("deviation_v"),
                                 r.get("score"), r.get("alerting"), r.get("drifting_eye"),
                                 r.get("direction"), r.get("skip_reason"), r.get("head_yaw"),
                                 g[0], g[1], mv.get("velocity_dps"), mv.get("in_saccade"),
                                 sac.get("amplitude_deg"), sac.get("peak_velocity_dps"),
                                 (r.get("eye_test") or {}).get("mode")])

            if args.no_window:
                print(f"#{n} {describe(r)}  rtt {rtt:.0f}ms")
                continue

            draw_eyes(frame, r)
            plot.draw(frame, r)
            cv2.putText(frame, f"{fps_live:.1f} fps  rtt {rtt:.0f} ms   [s] saccade test  [p] pursuit test"
                               "  [x] cancel  [c] recalibrate  [q] quit",
                        (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.imshow("eyes (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                eyes.recalibrate()
                print(">> recalibrating: look straight at the camera")
            if key in (ord("s"), ord("p")):
                mode = "saccades" if key == ord("s") else "pursuit"
                eyes.command({"type": "eye_test_start", "mode": mode, "duration": args.test_duration})
            if key == ord("x"):
                eyes.command({"type": "eye_test_cancel"})
    finally:
        cap.release()
        cv2.destroyAllWindows()
        eyes.close()
        if log_file:
            log_file.close()
            print(f"wrote {n} rows to {args.log}")


if __name__ == "__main__":
    main()
