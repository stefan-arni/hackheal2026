"""fal budget and cache housekeeping.

    uv run python tools/fal_budget.py status                 # spend so far, caps, price, cache size
    uv run python tools/fal_budget.py confirm-price 0.015    # after checking the fal dashboard
    uv run python tools/fal_budget.py migrate                # copy existing real fal results into the cache
    uv run python tools/fal_budget.py forget <sha1-key>      # allow re-paying for a lost billed request
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service.fal_budget import EST_USD_PER_CALL, PROBE_CALLS, Budget  # noqa: E402
from service.fal_cache import FalCache, image_key  # noqa: E402
from service.fal_client_wrap import PARAMS  # noqa: E402
from service.runs import RESERVED_JSON, vis_extension  # noqa: E402
from service.sam_result import SamResult  # noqa: E402


def status(budget: Budget, cache: FalCache) -> None:
    usd, confirmed = budget.price()
    entries = budget.entries()
    by_run = Counter(e.get("run") for e in entries if Budget._is_billed(e))
    maybe = sum(e.get("billed") == "maybe" for e in entries)
    print(f"price: ${usd}/call ({'confirmed' if confirmed else f'UNCONFIRMED estimate; only {PROBE_CALLS} calls allowed'})")
    print(f"billed calls: {budget.billed()}/{budget.total_cap} ≈ ${budget.billed() * usd:.3f}"
          + (f"  (incl. {maybe} 'maybe' from lost responses)" if maybe else ""))
    for run, n in sorted(by_run.items(), key=lambda x: -x[1]):
        print(f"  run {run!r}: {n}/{budget.run_cap}")
    print(f"cache: {len(cache.keys())} frames in {cache.root}")
    print(f"log: {budget.log_path}")


def is_synthetic(rec: dict) -> bool:
    url = rec.get("mesh_url") or ""
    return bool(rec.get("SYNTHETIC") or rec.get("metadata", {}).get("SYNTHETIC")
                or url.startswith(("synthetic://", "mock://")))


def migrate(cache: FalCache, frames_dir: Path) -> None:
    folders = sorted({p.parent for p in (REPLAY_ROOT / "data").glob("**/*.json")
                      if p.name not in RESERVED_JSON and "fal_cache" not in p.parts})
    moved = skipped = unresolved = already = 0
    for folder in folders:
        for p in sorted(folder.glob("*.json")):
            if p.name in RESERVED_JSON:
                continue
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if "metadata" not in rec or "frame" not in rec:
                continue
            if is_synthetic(rec):
                skipped += 1
                continue
            key = rec.get("image_sha1")
            if key is None:  # hash the original image if we can find it
                for src in (frames_dir / rec["frame"], folder / "input" / rec["frame"]):
                    if src.exists():
                        key = image_key(src.read_bytes())
                        break
            if key is None:
                unresolved += 1
                print(f"  can't find source image for {p} (frame {rec['frame']})")
                continue
            if cache.has(key):
                already += 1
                continue
            stem = p.stem
            ply = folder / f"{stem}.ply"
            vis = folder / rec["vis_file"] if rec.get("vis_file") else None
            response = {"meshes": [{"url": rec.get("mesh_url")}] if rec.get("mesh_url") else [],
                        "visualization": {"url": rec.get("visualization_url")} if rec.get("visualization_url") else None,
                        "metadata": rec["metadata"]}
            res = SamResult(response=response, latency_s=rec.get("latency_s", 0.0), key=key,
                            ply=ply.read_bytes() if ply.exists() else None,
                            visualization=vis.read_bytes() if vis and vis.exists() else None,
                            request_id=rec.get("request_id"))
            cache.put(key, res, params=PARAMS, vis_ext=vis_extension(response, res.visualization))
            moved += 1
    print(f"migrated {moved}, already cached {already}, skipped synthetic/mock {skipped}, unresolved {unresolved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    cp = sub.add_parser("confirm-price")
    cp.add_argument("usd_per_call", type=float)
    cp.add_argument("--note", default="")
    mp = sub.add_parser("migrate")
    mp.add_argument("--frames", type=Path, default=REPLAY_ROOT / "data/frames")
    fp = sub.add_parser("forget")
    fp.add_argument("key")
    a = p.parse_args()
    budget, cache = Budget(), FalCache()
    if a.cmd == "status":
        status(budget, cache)
    elif a.cmd == "confirm-price":
        if not 0 < a.usd_per_call < 1:
            sys.exit(f"suspicious price {a.usd_per_call}; expected about {EST_USD_PER_CALL}")
        budget.confirm_price(a.usd_per_call, a.note)
        print(f"confirmed ${a.usd_per_call}/call; caps now apply ({budget.total_cap} total, {budget.run_cap}/run)")
    elif a.cmd == "migrate":
        migrate(cache, a.frames)
    elif a.cmd == "forget":
        rid = budget.pending_request_id(a.key)
        if rid is None:
            sys.exit(f"no pending billed request for {a.key}")
        budget.log("forgotten", key=a.key, request_id=rid, billed=False)
        print(f"forgot request {rid}; the next --live run may pay for {a.key} again")
