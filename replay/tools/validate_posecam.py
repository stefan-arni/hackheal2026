# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.21", "numpy<2", "opencv-python-headless>=4.9,<4.11"]
# ///
"""Run teammates' posecam pipeline (unmodified, imported from a checkout) on a recorded clip.

    git worktree add --detach /tmp/wt-cv origin/cv
    uv run tools/validate_posecam.py data/IMG_9691.mov --posecam /tmp/wt-cv/posecam \\
        --bess single:left:39.0 --out data/reports/IMG_9691/posecam_run.json

Frames are decoded with ffmpeg (rotation applied), resized to 640 px wide like posecam's
client, and fed to PosePipeline.process_bgr(frame, t) with t = the clip's own timestamp, so
balance/BESS events land on the clip clock. Saves posecam's messages (foot_touchdown,
bess_error, ...) and a per-frame series (2D landmarks in full-frame px, world landmarks,
spine angles, foot heights). Needs pose_landmarker_full.task in <posecam>/models/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def clip_times(clip: Path) -> np.ndarray:
    ts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(clip)],
                        capture_output=True, text=True, check=True).stdout.split()
    return np.array([float(x.split(",")[0]) for x in ts if x.split(",")[0]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--posecam", type=Path, required=True, help="path to a posecam checkout (not modified)")
    p.add_argument("--bess", action="append", default=[], help="STANCE:NONDOMINANT:START_S, e.g. single:left:39.0")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    sys.path.insert(0, str(a.posecam.resolve()))
    import cv2
    from balance import BalanceConfig
    from bess import BessConfig
    from pose_analyzer import PoseAnalyzer
    from pose_pipeline import PosePipeline, pose_events

    W, H = 1080, 1920
    w, h = a.width, int(round(a.width * H / W / 2)) * 2
    t_all = clip_times(a.clip)
    pipe = PosePipeline(PoseAnalyzer("full"), balance=BalanceConfig(), bess=BessConfig())
    schedule = sorted((float(s.split(":")[2]), s.split(":")[0], s.split(":")[1]) for s in a.bess)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(a.clip), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    msgs, frames = [], []
    sx, sy = W / w, H / h
    for i, t in enumerate(t_all):
        buf = ff.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        while schedule and t >= schedule[0][0]:
            _, stance, nondom = schedule.pop(0)
            for m in pipe.handle_command({"type": "bess_start", "stance": stance, "nondominant": nondom}):
                msgs.append({"t_clip_s": round(float(t), 3), **m})
        frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        out = pipe.process_bgr(frame, float(t))
        for m in pose_events(out, frame_id=i):
            msgs.append({"t_clip_s": round(float(t), 3), **m})
        rec = {"i": i, "t": round(float(t), 4), "detected": bool(out.get("detected"))}
        if out.get("detected"):
            rec["xyv"] = [[round(l["x"] * W, 1), round(l["y"] * H, 1), round(l["visibility"], 3)] for l in out["landmarks"]]
            rec["world"] = [[round(l["x"], 4), round(l["y"], 4), round(l["z"], 4)] for l in out["world_landmarks"]]
            rec["spine"] = out.get("spine")
        b = out.get("balance") or {}
        rec["balance"] = {k: b.get(k) for k in ("state", "event", "lifted_foot", "foot_heights", "skip_reason")}
        bs = out.get("bess") or {}
        rec["bess"] = {k: bs.get(k) for k in ("phase", "stance", "time_left", "errors", "active")}
        frames.append(rec)
        if i % 300 == 0:
            print(f"  frame {i}/{len(t_all)}", flush=True)
    ff.kill()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"source": a.clip.name, "posecam": str(a.posecam), "frame_size": [W, H],
                                 "analysis_size": [w, h], "messages": msgs, "frames": frames}))
    kinds = {}
    for m in msgs:
        kinds[m.get("kind")] = kinds.get(m.get("kind"), 0) + 1
    print(f"{len(frames)} frames, messages: {kinds} -> {a.out}")


if __name__ == "__main__":
    main()
