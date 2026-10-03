"""Replay a recorded iPhone session, or a video file, through the pose pipeline.

Same processing as server.py (pose, balance, duck, BESS, sway), run locally, with
the recording's own timestamps, so results are repeatable.

    python playback.py recordings/20261003-101500.jsonl      # iPhone session (incl. depth)
    python playback.py clip.mov --test quiet@0:20             # video: quiet stance at 0 s, 20 s
    python playback.py clip.mov --test romberg_eo@0 --test romberg_ec@35
    python playback.py rec.jsonl --out results.json           # save sway/BESS results

Recordings come from the app's Record button or `server.py --record`. A test given
with --test starts its countdown at that time in the video (default countdown 5 s).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

log = logging.getLogger("playback")


def parse_test(spec: str) -> tuple[float, dict]:
    """'quiet@3:20' -> (3.0, {"type": "sway_start", "test": "quiet", "duration": 20})"""
    name, _, rest = spec.partition("@")
    at, _, dur = rest.partition(":")
    cmd = {"type": "sway_start", "test": name}
    if name in ("double", "tandem_bess", "single"):
        cmd = {"type": "bess_start", "stance": "tandem" if name == "tandem_bess" else name}
    elif dur:
        cmd["duration"] = float(dur)
    return float(at or 0), cmd


def make_pipeline(model: str, sway_duration: float, countdown: float):
    from balance import BalanceConfig
    from bess import BessConfig
    from duck import DuckConfig
    from pose_analyzer import PoseAnalyzer
    from pose_pipeline import PosePipeline
    from sway import SwayConfig
    return PosePipeline(PoseAnalyzer(model=model), BalanceConfig(), DuckConfig(),
                        BessConfig(countdown_s=countdown),
                        sway=SwayConfig(duration_s=sway_duration, countdown_s=countdown))


def frames_from_jsonl(path: str):
    """Yield ('frame', frame, meta) or ('command', None, msg) from a recording."""
    from ws_server import decode_message
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame, meta = decode_message(line)
            yield ("frame", frame, meta) if frame is not None else ("command", None, meta)


def frames_from_video(path: str, width: int):
    from ws_client import open_source, resize_to_width
    cap = open_source(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield "frame", resize_to_width(frame, width), {"timestamp_ms": t_ms}
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="recordings/*.jsonl from the app, or a video file")
    p.add_argument("--test", action="append", default=[],
                   help="video only: TEST@START[:DURATION], e.g. quiet@0:20 (quiet, tandem, "
                        "romberg_eo, romberg_ec; or BESS: double, tandem_bess, single)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    p.add_argument("--sway-duration", type=float, default=30.0)
    p.add_argument("--countdown", type=float, default=5.0)
    p.add_argument("--width", type=int, default=640, help="video only: resize frames to this width")
    p.add_argument("--out", help="write all results (sway_done / bess_done) to this JSON file")
    p.add_argument("--quiet", action="store_true", help="only print results")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s", stream=sys.stdout)

    from pose_pipeline import pose_events
    pipe = make_pipeline(args.model, args.sway_duration, args.countdown)
    is_rec = Path(args.source).suffix.lower() == ".jsonl"
    source = frames_from_jsonl(args.source) if is_rec else frames_from_video(args.source, args.width)
    scheduled = sorted((parse_test(s) for s in args.test), key=lambda x: x[0])
    if is_rec and scheduled:
        p.error("--test is for video files; recordings already contain the app's commands")

    results, n, t0, depth_frames = [], 0, None, 0
    for kind, frame, meta in source:
        if kind == "command":
            if meta.get("type") == "recalibrate":
                pipe.recalibrate()
            elif meta.get("type") not in ("ping", "record_start", "record_stop"):
                for r in pipe.handle_command(meta):
                    if r["type"] == "error":
                        log.warning("command %s: %s", meta.get("type"), r["error"])
            continue
        ts = meta.get("timestamp_ms")
        t = ts / 1000.0 if isinstance(ts, (int, float)) else n / 30.0
        t0 = t if t0 is None else t0
        while scheduled and t - t0 >= scheduled[0][0]:
            for r in pipe.handle_command(scheduled.pop(0)[1]):
                if r["type"] == "error":
                    log.warning("%s", r["error"])
        out = pipe.process_bgr(frame, t=t, meta=meta)
        n += 1
        depth_frames += meta.get("_depth") is not None
        for m in pose_events(out, n):
            if m.get("kind") in ("sway_done", "bess_done"):
                results.append({k: v for k, v in m.items() if k not in ("session",)})

    pipe.close()
    print(f"\n{n} frames, {depth_frames} with depth")
    for r in results:
        print(json.dumps(r, indent=2))
    if pipe.sway is not None and pipe.sway.summary()["romberg"]:
        print("Romberg (eyes closed / eyes open):")
        print(json.dumps(pipe.sway.summary()["romberg"], indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"results": results,
                                              "sway_session": pipe.sway.summary() if pipe.sway else None},
                                             indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
