"""Pose test client: stands in for the iPhone app.

Grabs frames from your webcam (or a video / image file), sends them to the pose
server exactly the way the iPhone app will, and draws the returned skeleton
and spine angle on screen, plus single-leg balance: stand on one foot and the
window times it, then flashes and beeps when the raised foot touches down.
Duck down quickly and it quacks.

BESS balance test: a side panel has a button for each of the three stances
(feet together, tandem, single leg; 10 s each by default, see BessConfig.duration_s)
and shows the error score per stance and in total. Click a button (or press
1 / 2 / 3), get into position during the countdown, hold until the timer ends.

Usage:
    python server.py                         # terminal 1
    python test_client.py                    # terminal 2: webcam -> server
    python test_client.py --source clip.mp4  # video file instead of webcam
    python test_client.py --source me.jpg    # single image, prints JSON
    python test_client.py --local            # skip the server, run MediaPipe directly
    python test_client.py --log angles.csv   # also log spine angles per frame

Keys: q / Esc quit, r reset balance stats, 1/2/3 start a BESS test, n switch
non-dominant leg, x cancel test, 0 reset BESS scores. Eye tracking is separate: see eye_client.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

from balance import draw_balance
from bess import STANCES
from bess_ui import BessPanel
from pose_analyzer import draw_pose
from quack import play_quack
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


class RemotePose:
    def __init__(self, url: str, use_json: bool):
        self.client = RemoteClient(url, "pose", use_json)

    def __call__(self, frame, quality: int = 80) -> dict:
        return self.client(frame, quality)

    def reset(self):
        self.client.send_json({"type": "recalibrate"})

    def command(self, msg: dict):
        self.client.send_json(msg)

    def take_replies(self) -> list[dict]:
        return self.client.take_replies()

    def close(self):
        self.client.close()


class LocalPose:
    """Runs MediaPipe in-process (no server), for checking the model alone."""

    def __init__(self, model: str, balance: bool = True, balance_mode: str = "world",
                 duck: bool = True, bess: bool = True, bess_eyes: str = "off"):
        from balance import BalanceConfig
        from bess import BessConfig
        from duck import DuckConfig
        from pose_analyzer import PoseAnalyzer
        from pose_pipeline import PosePipeline
        self.analyzer = PoseAnalyzer(model=model)
        self._replies: list[dict] = []
        if balance or duck or bess:
            self.analyzer = PosePipeline(self.analyzer,
                                         BalanceConfig(mode=balance_mode) if balance else None,
                                         DuckConfig() if duck else None,
                                         BessConfig() if bess else None, bess_eyes=bess_eyes)

    def __call__(self, frame, quality: int = 80) -> dict:
        r = self.analyzer.process_bgr(frame)
        return r.to_dict() if hasattr(r, "to_dict") else r

    def reset(self):
        if hasattr(self.analyzer, "recalibrate"):
            self.analyzer.recalibrate()

    def command(self, msg: dict):
        if hasattr(self.analyzer, "handle_command"):
            self._replies += self.analyzer.handle_command(msg)

    def take_replies(self) -> list[dict]:
        r, self._replies = self._replies, []
        return r

    def close(self):
        self.analyzer.close()


def announce_duck(d: dict | None):
    if (d or {}).get("event") == "duck":
        print(f">> QUACK! duck #{d['count']}")
        play_quack()
        announce_duck.flash_until = time.time() + 0.8


def draw_duck(frame, d: dict | None):
    if not d:
        return frame
    h, w = frame.shape[:2]
    if d.get("drop") is not None:
        label = f"duck: drop {d['drop']:.2f} / {d['threshold']:.2f} ({d['mode']})   quacks {d['count']}"
        cv2.putText(frame, label, (w - 430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(frame, label, (w - 430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if time.time() < getattr(announce_duck, "flash_until", 0):
        (tw, th), _ = cv2.getTextSize("QUACK!", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        x, y = (w - tw) // 2, h // 2 + th // 2
        cv2.putText(frame, "QUACK!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 12)
        cv2.putText(frame, "QUACK!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 255), 5)
    return frame


def announce_balance(b: dict | None):
    ev = (b or {}).get("event")
    if ev == "balance_start":
        print(f">> balancing on {b['standing_foot']} foot ({b['lifted_foot']} foot raised)")
    elif ev == "touchdown":
        print(f">> FOOT DOWN: {b['touched_foot']} foot touched the ground after {b['held_s']:.1f}s "
              f"(touches: {b['touch_count']}, best: {b['best_hold_s']:.1f}s)")
        beep()


def announce_bess(b: dict | None):
    for ev in (b or {}).get("events", []):
        k = ev["kind"]
        if k == "bess_started":
            print(f">> BESS {STANCES[ev['stance']]}: get in position, hands on hips "
                  f"(non-dominant: {ev['nondominant']})")
        elif k == "bess_running":
            print(">> BESS: scoring started")
            for w in ev.get("warnings", []):
                print(f"   warning: {w}")
        elif k == "bess_error":
            if ev["counted"]:
                print(f">> BESS +1 at {ev['t']:.1f}s: {ev['label']} (errors: {ev['errors']})")
                beep()
            else:
                print(f">> BESS    {ev['t']:.1f}s: {ev['label']} (not counted: {ev['not_counted_reason']})")
        elif k == "bess_done":
            sc = ev["session"]["scores"]
            fmt = lambda v: "-" if v is None else v  # noqa: E731
            print(f">> BESS {ev['label']} done: {ev['errors']} errors  | feet together {fmt(sc['double'])}, "
                  f"tandem {fmt(sc['tandem'])}, single leg {fmt(sc['single'])} | total {ev['session']['total']}")
        elif k == "bess_failed":
            print(f">> BESS failed: {ev['reason']}")
        elif k == "bess_cancelled":
            print(">> BESS cancelled")


def main():
    p = argparse.ArgumentParser(description="Test client for the pose server")
    p.add_argument("--url", default="ws://localhost:8765")
    p.add_argument("--source", default="0", help="webcam index, video path, or image path")
    p.add_argument("--local", action="store_true", help="run MediaPipe locally, no server")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="model for --local mode")
    p.add_argument("--no-balance", action="store_true", help="--local only: skip balance detection")
    p.add_argument("--balance-mode", choices=["world", "image"], default="world",
                   help="--local only: how foot height is measured")
    p.add_argument("--no-duck", action="store_true", help="--local only: skip duck detection")
    p.add_argument("--mute", action="store_true", help="don't play the quack sound")
    p.add_argument("--nondominant", choices=["left", "right"], default="left",
                   help="BESS: the non-dominant leg (stood on in single leg, in back in tandem)")
    p.add_argument("--bess", choices=list(STANCES), help="start this BESS test right away")
    p.add_argument("--no-bess", action="store_true", help="--local only: skip BESS")
    p.add_argument("--bess-eyes", choices=["off", "auto", "manual"], default="off",
                   help="--local only: BESS eye tracking (off by default)")
    p.add_argument("--json", action="store_true",
                   help="send base64 JSON messages instead of binary JPEG")
    p.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    p.add_argument("--width", type=int, default=640, help="resize frames to this width")
    p.add_argument("--log", help="write per-frame spine angles to this CSV")
    p.add_argument("--no-window", action="store_true", help="print results instead of showing video")
    args = p.parse_args()

    if args.mute:
        global play_quack
        play_quack = lambda: None  # noqa: E731
    pose = (LocalPose(args.model, not args.no_balance, args.balance_mode, not args.no_duck,
                      not args.no_bess, args.bess_eyes)
            if args.local
            else RemotePose(args.url, args.json))

    # Single image: run once, print the result, show it.
    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(args.source)
        if frame is None:
            raise SystemExit(f"could not read {args.source}")
        result = pose(frame, args.quality)
        print(json.dumps({k: result.get(k) for k in ("detected", "spine", "inference_ms")}, indent=2))
        if not args.no_window:
            cv2.imshow("pose", draw_pose(frame, result))
            cv2.waitKey(0)
        pose.close()
        return

    cap = open_source(args.source)
    log_file = writer = None
    if args.log:
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["frame", "time_s", "detected", "trunk_angle_deg", "inclination_deg",
                         "flexion_deg", "lateral_deg", "trunk_visibility", "reliable",
                         "balance_state", "balance_event", "left_foot_height", "right_foot_height",
                         "lifted_foot", "balance_time_s", "touch_count",
                         "floor_calibrated",
                         "duck_mode", "duck_drop", "duck_event", "duck_count",
                         "bess_phase", "bess_hip_left", "bess_hip_right", "bess_errors"])

    panel = BessPanel(args.nondominant)
    window = "posecam pose (q quit)"
    if not args.no_window:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, lambda ev, x, y, fl, _: panel.on_mouse(ev, x, y, video_w))
    video_w = 0
    if args.bess:
        pose.command(panel.command_for(args.bess))

    t_start, n, fps = time.time(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_to_width(frame, args.width)

            t0 = time.perf_counter()
            result = pose(frame, args.quality)
            rtt = (time.perf_counter() - t0) * 1000
            fps = 0.9 * fps + 0.1 * (1000 / max(rtt, 1))
            n += 1
            b = result.get("balance")
            d = result.get("duck")
            bess = result.get("bess")
            announce_balance(b)
            announce_duck(d)
            announce_bess(bess)
            panel.update(bess)
            for r in pose.take_replies():
                if r["type"] == "error":
                    print(f">> {r['error']}")

            s = result.get("spine") or {}
            if writer:
                fh = (b or {}).get("foot_heights") or {}
                writer.writerow([n, round(time.time() - t_start, 3), result["detected"],
                                 s.get("trunk_angle_deg"), s.get("inclination_deg"),
                                 s.get("flexion_deg"), s.get("lateral_deg"),
                                 s.get("trunk_visibility"), s.get("reliable"),
                                 (b or {}).get("state"), (b or {}).get("event"),
                                 fh.get("left"), fh.get("right"), (b or {}).get("lifted_foot"),
                                 (b or {}).get("balance_time_s"), (b or {}).get("touch_count"),
                                 (b or {}).get("floor_calibrated"),
                                 (d or {}).get("mode"), (d or {}).get("drop"),
                                 (d or {}).get("event"), (d or {}).get("count"),
                                 (bess or {}).get("phase"),
                                 ((bess or {}).get("hip_angles") or {}).get("left"),
                                 ((bess or {}).get("hip_angles") or {}).get("right"),
                                 (bess or {}).get("errors")])

            if args.no_window:
                if result["detected"]:
                    line = (f"#{n} trunk {s['trunk_angle_deg']:+6.1f}  incline {s['inclination_deg']:5.1f}"
                            f"  flex {s['flexion_deg']:+6.1f}  lat {s['lateral_deg']:+6.1f}")
                else:
                    line = f"#{n} no person"
                if b:
                    line += f"  balance {b['state']}"
                    if b.get("balance_time_s") is not None:
                        line += f" {b['balance_time_s']:.1f}s"
                print(f"{line}  rtt {rtt:.0f}ms")
                continue

            draw_pose(frame, result)
            draw_balance(frame, b)
            draw_duck(frame, d)
            cv2.putText(frame, f"{fps:.1f} fps  rtt {rtt:.0f} ms  model {result['inference_ms']:.0f} ms",
                        (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            video_w = frame.shape[1]
            cv2.imshow(window, panel.render(frame, bess) if bess is not None else frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                pose.reset()
                print(">> balance stats and duck count reset")
            actions = panel.take_actions()
            if bess is not None and panel.key_action(key):
                actions.append(panel.key_action(key))
            for a in actions:
                cmd = panel.command_for(a)
                if cmd:
                    pose.command(cmd)
                elif a == "toggle_nd":
                    print(f">> non-dominant leg: {panel.nondominant}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        if log_file:
            log_file.close()
            print(f"wrote {n} rows to {args.log}")


if __name__ == "__main__":
    main()
