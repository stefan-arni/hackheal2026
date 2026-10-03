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
import io
import json
import shutil
import sys
import time
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service import analytics, bundle, pipeline, report  # noqa: E402

SRC_HEIGHT = 640  # video frames copied into the bundle for the picture-in-picture overlay


def copy_source_frame(src: Path, dst: Path) -> float:
    """Downscale the image that was sent to fal; returns the scale (bundle px / fal-image px)."""
    from PIL import Image
    with Image.open(src) as im:
        k = SRC_HEIGHT / im.height
        im.convert("RGB").resize((round(im.width * k), SRC_HEIGHT), Image.LANCZOS).save(dst, "JPEG", quality=82)
    return k


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
    for t, v, stem in zip(res["t_s"], res["vis_files"], res["stems"]):
        if v:  # fal's 4-panel image, downscaled (originals are ~3880 px wide, ~0.7 MB each)
            from PIL import Image
            with Image.open(a.run_dir / v) as im:
                k = min(1.0, 1940 / im.width)
                im.convert("RGB").resize((round(im.width * k), round(im.height * k)), Image.LANCZOS).save(
                    a.out / "frames" / v, "JPEG", quality=80)
        fr = {"t": round(t * 1000, 1), "fal_vis_url": f"frames/{v}" if v else None}
        src = a.frames_dir / f"{stem}.jpg" if a.frames_dir else None
        if src is not None and src.exists():
            fr["src_scale"] = round(copy_source_frame(src, a.out / "frames" / f"{stem}_src.jpg"), 5)
            fr["src_url"] = f"frames/{stem}_src.jpg"
        frames.append(fr)
    stats = analytics.compute(res, ev["events"], a.expected_stance)
    meta = bundle.write_bundle(a.out, a.out.name, res, events=ev["events"], frames=frames, analytics=stats)
    t_rep = time.perf_counter()
    xv = json.loads(a.cross_validation.read_text()) if a.cross_validation else None
    report.write_report(a.out, meta, stats, res, title=a.title, cross_validation=xv)
    print(f"report: {a.out / 'report.html'} ({time.perf_counter() - t_rep:.1f}s)")
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
    p.add_argument("--frames-dir", type=Path, default=None, help="images that were sent to fal (for the video overlay)")
    p.add_argument("--title", default=None, help="report title (default: bundle name)")
    p.add_argument("--cross-validation", type=Path, default=None, help="tools/cross_validate.py output for the footer")
    main(p.parse_args())
