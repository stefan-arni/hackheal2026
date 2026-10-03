"""Run a run folder through the full pipeline (geometry.py) to an aligned replay bundle.

    uv run python tools/run_pipeline.py data/fal_out/demo --events data/events/IMG_9691.json \\
        --landmarks data/scout/IMG_9691/landmarks_2d.json --out data/bundles/demo

Outlier rejection uses the MediaPipe 2D timeline; stance (floor fit, noise floor) uses the
events file's stance_intervals_clip_s when present. Without --height-cm SAM's own metric
scale is kept. The bundle (meta.json, faces.bin, verts.bin, frames/<vis>) opens in the viewer:
    python3 -m http.server 8018 -d replay  ->  http://localhost:8018/viewer/?src=/data/bundles/demo/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service import analytics, bundle, pipeline  # noqa: E402


def main(a: argparse.Namespace) -> None:
    run = pipeline.load_run(a.run_dir)
    ev = json.loads(a.events.read_text()) if a.events else {"events": []}
    lm = json.loads(a.landmarks.read_text()) if a.landmarks else None
    stance = None
    if "stance_intervals_clip_s" in ev and "clip_start_epoch_ms" in ev:
        base = ev["clip_start_epoch_ms"] / 1000
        stance = [(base + t0, base + t1) for t0, t1 in ev["stance_intervals_clip_s"]]
    t0 = time.perf_counter()
    res = pipeline.process(run, ev["events"], a.height_cm / 100 if a.height_cm else None,
                           landmarks=lm, stance_intervals_s=stance)
    dt = time.perf_counter() - t0
    res["quality"]["source_run"] = str(a.run_dir)
    if a.synthetic:
        res["quality"]["SYNTHETIC"] = True
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frames").mkdir(exist_ok=True)
    frames = []
    for t, v in zip(res["t_s"], res["vis_files"]):
        if v:
            shutil.copy2(a.run_dir / v, a.out / "frames" / v)
        frames.append({"t": round(t * 1000, 1), "fal_vis_url": f"frames/{v}" if v else None})
    stats = analytics.compute(res, ev["events"], a.expected_stance)
    meta = bundle.write_bundle(a.out, a.out.name, res, events=ev["events"], frames=frames, analytics=stats)
    size = sum(p.stat().st_size for p in a.out.glob("*.bin")) + (a.out / "meta.json").stat().st_size
    q = res["quality"]
    print(f"{len(res['t_s'])} frames -> {a.out}  ({size / 1e6:.1f} MB, verts {meta['verts_dtype']}, process {dt:.2f}s)")
    print(json.dumps({k: v for k, v in q.items() if k != "reprojection"}, indent=1))
    print("analytics:", json.dumps(meta.get("analytics")))
    if "reprojection" in q:
        print("reprojection:", json.dumps({k: v for k, v in q["reprojection"].items() if k != "per_frame_px"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--landmarks", type=Path, default=None)
    p.add_argument("--height-cm", type=float, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--expected-stance", default=None, help="double | tandem | single_left | single_right")
    main(p.parse_args())
