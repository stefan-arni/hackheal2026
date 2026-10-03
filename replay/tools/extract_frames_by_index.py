"""Extract frames from a clip by SOURCE FRAME INDEX, so the same frame is always byte-identical.

    # still window, every 10th source frame (3 fps), grid anchored at 57.8 s
    uv run python tools/extract_frames_by_index.py data/IMG_9691.mov --out data/frames/still \\
        --anchor 57.8 --select 57.8:64.1:10 --crop 75,429,1045,1920
    # demo segment at 3 fps + a 10 fps burst, same anchor -> shared frames hit the fal cache
    uv run python tools/extract_frames_by_index.py data/IMG_9691.mov --out data/frames/demo \\
        --anchor 57.8 --select 44.5:64.1:10 --select 48.0:50.2:3 --crop 75,429,1045,1920

Grid: index i is selected by START:END:STEP when t(i) is within [START, END] and
(i - anchor_index) % STEP == 0, with anchor_index the first frame at or after --anchor.
The video is always decoded from frame 0 (deterministic indexing), cropped by ffmpeg
and encoded by Pillow with fixed settings. Files are named f<index>.jpg.

timestamps.json (merged across runs) maps file -> src_index, t_clip_ms, and t_ms on a
phone-clock-style timeline (clip creation_time as epoch ms + t_clip_ms).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

JPEG_QUALITY = 92


def clip_info(clip: Path) -> tuple[int, int, np.ndarray, float]:
    """Displayed size, per-frame timestamps (ms from clip start), creation time (epoch ms)."""
    meta = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation:format_tags=creation_time", "-of", "json", str(clip)],
        capture_output=True, text=True, check=True).stdout)
    st = meta["streams"][0]
    rot = next((abs(int(d["rotation"])) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    W, H = (st["height"], st["width"]) if rot in (90, 270) else (st["width"], st["height"])
    ts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(clip)],
                        capture_output=True, text=True, check=True).stdout.split()
    t_ms = np.array([float(x.split(",")[0]) * 1000 for x in ts if x.split(",")[0]])
    created = meta.get("format", {}).get("tags", {}).get("creation_time")
    epoch_ms = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp() * 1000 if created else 0.0
    return W, H, t_ms, epoch_ms


def select(t_ms: np.ndarray, anchor_s: float, specs: list[str]) -> list[int]:
    a0 = int(np.searchsorted(t_ms, anchor_s * 1000 - 0.5))
    chosen: set[int] = set()
    for spec in specs:
        start, end, step = spec.split(":")
        lo, hi, step = float(start) * 1000 - 0.5, float(end) * 1000 + 0.5, int(step)
        chosen |= {i for i in range(len(t_ms)) if lo <= t_ms[i] <= hi and (i - a0) % step == 0}
    return sorted(chosen)


def extract(clip: Path, out: Path, indices: list[int], crop: tuple[int, int, int, int] | None) -> dict:
    W, H, t_ms, epoch_ms = clip_info(clip)
    x0, y0, x1, y1 = crop or (0, 0, W, H)
    w, h = x1 - x0, y1 - y0
    out.mkdir(parents=True, exist_ok=True)
    want = set(indices)
    # Convert to RGB *before* cropping: cropping 4:2:0 video rounds odd offsets/sizes, which
    # silently changes the frame size and misaligns every following frame in the pipe.
    vf = f"format=rgb24,crop={w}:{h}:{x0}:{y0}"
    probe = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", vf, "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    if len(probe) != w * h * 3:
        raise RuntimeError(f"ffmpeg frame is {len(probe)} bytes, expected {w}x{h}x3 = {w * h * 3}")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", vf,
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    stamps_path = out / "timestamps.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    written = 0
    try:
        for i in range(max(indices) + 1):
            buf = ff.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                raise RuntimeError(f"clip ended at frame {i}")
            if i not in want:
                continue
            name = f"f{i:05d}.jpg"
            jpg = io.BytesIO()
            Image.frombuffer("RGB", (w, h), buf).save(jpg, "JPEG", quality=JPEG_QUALITY, optimize=False)
            (out / name).write_bytes(jpg.getvalue())
            stamps[name] = {"src_index": i, "t_clip_ms": round(float(t_ms[i]), 3),
                            "t_ms": round(epoch_ms + float(t_ms[i]), 1), "crop": [x0, y0, x1, y1],
                            "frame_size": [W, H], "source": clip.name}
            written += 1
    finally:
        ff.kill()
        ff.wait()
    stamps_path.write_text(json.dumps(dict(sorted(stamps.items())), indent=1))
    return {"written": written, "dir": str(out)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--anchor", type=float, required=True, help="grid anchor time (s)")
    p.add_argument("--select", action="append", required=True, help="START:END:STEP in seconds / source frames")
    p.add_argument("--crop", default=None, help="x0,y0,x1,y1 in displayed pixels (fixed per trial)")
    p.add_argument("--only", default=None, help="comma-separated subset of the selected indices (e.g. a probe)")
    a = p.parse_args()
    _, _, t_ms, _ = clip_info(a.clip)
    idx = select(t_ms, a.anchor, a.select)
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        missing = keep - set(idx)
        if missing:
            raise SystemExit(f"--only indices not on the grid: {sorted(missing)}")
        idx = sorted(keep)
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    print(f"{len(idx)} frames: indices {idx[:6]}{' ...' if len(idx) > 6 else ''}")
    print(extract(a.clip, a.out, idx, crop))
