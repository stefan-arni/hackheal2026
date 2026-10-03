"""Cache a finished trial as a dashboard "previous trial" fallback (replay/cache/demo_<name>/).

    uv run python tools/cache_trial.py bess-single-1791045891978            # -> cache/demo_live/
    uv run python tools/cache_trial.py bess-single-protocol protocol_single  # -> cache/demo_protocol_single/

Mesh, analytics and metadata only: no camera frames or photos (cache/demo_*/ is committed).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tid = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "live"
src, dst = ROOT / "data/trials" / tid, ROOT / f"cache/demo_{name}"
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
