"""Cache a finished live trial as the dashboard's "previous trial" fallback (replay/cache/demo_live/).

    uv run python tools/cache_trial.py bess-single-1791045891978

Mesh, analytics and metadata only: no camera frames or photos (cache/demo_*/ is committed).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tid = sys.argv[1]
src, dst = ROOT / "data/trials" / tid, ROOT / "cache/demo_live"
meta = json.loads((src / "meta.json").read_text())
shutil.rmtree(dst, ignore_errors=True)
dst.mkdir(parents=True)
for name in ("faces.bin", "verts.bin", "analytics.json"):
    if (src / name).exists():
        shutil.copy(src / name, dst / name)
meta.pop("verts_raw", None)  # only for the video overlay, which needs the frames
meta["frames"] = [{"t": f["t"], "fal_vis_url": None} for f in meta["frames"]]
meta["cached_demo"] = True
(dst / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
size = sum(p.stat().st_size for p in dst.iterdir()) / 1e6
print(f"cached {tid} -> {dst} ({size:.1f} MB, {len(meta['frames'])} frames, coverage {meta.get('coverage')})")
