"""Eye test client: stands in for the iPhone app for eye tracking.

Grabs frames from your webcam (or a video file), sends them to the eye server,
draws the tracked irises, and flashes + beeps when one eye drifts out of line.

Usage:
    python eye_server.py                    # terminal 1
    python eye_client.py                    # terminal 2: webcam -> eye server
    python eye_client.py --local            # skip the server, run MediaPipe directly
    python eye_client.py --source clip.mp4  # video file instead of webcam
    python eye_client.py --log eyes.csv     # log drift values per frame (for tuning)

Keys: q / Esc quit, c recalibrate (look straight at the camera first).
Try it: after calibration, cross your eyes for a second -> alert.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

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

    def __call__(self, frame, quality: int = 80) -> dict:
        return self.client(frame, quality)

    def recalibrate(self):
        self.client.send_json({"type": "recalibrate"})

    def close(self):
        self.client.close()


class LocalEyes:
    """Runs MediaPipe in-process (no server)."""

    def __init__(self, threshold: float, vthreshold: float, hold: float):
        from eye_tracker import EyeAnalyzer, MonitorConfig
        self.analyzer = EyeAnalyzer(MonitorConfig(h_threshold=threshold, v_threshold=vthreshold,
                                                  hold_s=hold))

    def __call__(self, frame, quality: int = 80) -> dict:
        return self.analyzer.process_bgr(frame)

    def recalibrate(self):
        self.analyzer.recalibrate()

    def close(self):
        self.analyzer.close()


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
    return text


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


def main():
    p = argparse.ArgumentParser(description="Test client for the eye server")
    p.add_argument("--url", default="ws://localhost:8766")
    p.add_argument("--source", default="0", help="webcam index, video path, or image path")
    p.add_argument("--local", action="store_true", help="run MediaPipe locally, no server")
    p.add_argument("--threshold", type=float, default=0.10, help="--local only: horizontal threshold")
    p.add_argument("--vthreshold", type=float, default=0.08, help="--local only: vertical threshold")
    p.add_argument("--hold", type=float, default=0.5, help="--local only: seconds before alerting")
    p.add_argument("--json", action="store_true",
                   help="send base64 JSON messages instead of binary JPEG")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100")
    p.add_argument("--width", type=int, default=960,
                   help="resize frames to this width (eyes need detail; 960+ recommended)")
    p.add_argument("--log", help="write per-frame eye values to this CSV")
    p.add_argument("--no-window", action="store_true", help="print results instead of showing video")
    args = p.parse_args()

    eyes = (LocalEyes(args.threshold, args.vthreshold, args.hold) if args.local
            else RemoteEyes(args.url, args.json))

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"could not read {args.source}")
        result = eyes(resize_to_width(frame, args.width), args.quality)
        print(json.dumps({k: result.get(k) for k in ("face_detected", "head_yaw", "eyes", "skip_reason")},
                         indent=2))
        print("(a single image can't trigger an alert: alerts need calibration over time)")
        if not args.no_window:
            cv2.imshow("eyes", draw_eyes(frame, result))
            cv2.waitKey(0)
        eyes.close()
        return

    cap = open_source(args.source)
    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["frame", "time_s", "face_detected", "state", "deviation_h", "deviation_v",
                         "score", "alerting", "drifting_eye", "direction", "skip_reason", "head_yaw"])

    t_start, n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_to_width(frame, args.width)

            t0 = time.perf_counter()
            r = eyes(frame, args.quality)
            rtt = (time.perf_counter() - t0) * 1000
            fps = 0.9 * fps + 0.1 * (1000 / max(rtt, 1))
            n += 1
            announce(r)

            if writer:
                writer.writerow([n, round(time.time() - t_start, 3), r.get("face_detected"),
                                 r.get("state"), r.get("deviation_h"), r.get("deviation_v"),
                                 r.get("score"), r.get("alerting"), r.get("drifting_eye"),
                                 r.get("direction"), r.get("skip_reason"), r.get("head_yaw")])

            if args.no_window:
                print(f"#{n} {describe(r)}  rtt {rtt:.0f}ms")
                continue

            draw_eyes(frame, r)
            cv2.putText(frame, f"{fps:.1f} fps  rtt {rtt:.0f} ms   [c] recalibrate  [q] quit",
                        (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("eyes (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                eyes.recalibrate()
                print(">> recalibrating: look straight at the camera")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        eyes.close()
        if log_file:
            log_file.close()
            print(f"wrote {n} rows to {args.log}")


if __name__ == "__main__":
    main()
