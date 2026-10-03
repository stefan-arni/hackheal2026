"""Send extracted frames through fal (budget-guarded) and save meshes, metadata, visualizations.

    uv run python tools/send_frames.py --limit 3 --out data/fal_out/probe          # cache only (default)
    uv run python tools/send_frames.py --limit 3 --out data/fal_out/probe --live   # the 3-frame probe
    uv run python tools/send_frames.py --start 6 --limit 20 --out data/fal_out/still --live

Every frame is looked up by content hash in data/fal_cache/ first; identical frames are
never sent twice. Without --live nothing is sent: missing frames are listed and the
script exits 1. With --live it prints "about to send N new frames ≈ $X" and asks
(--yes skips the prompt). Caps: 500 calls total, 80 per run (--override-budget).
Until the price is confirmed (tools/fal_budget.py confirm-price), only 3 calls are allowed.

Per frame in --out: <stem>.ply, <stem>.json, <stem>_vis.<ext>, plus summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service.fal_budget import PROBE_CALLS, Budget, BudgetError  # noqa: E402
from service.fal_cache import FalCache, image_key  # noqa: E402
from service.fal_client_wrap import CacheMiss, SamBodyClient  # noqa: E402
from service.runs import write_frame, write_summary  # noqa: E402


def save(out: Path, frame: Path, t_ms: float, size: list[int], res, stamp: dict | None = None) -> dict:
    return write_frame(
        out, frame.stem, t_ms=t_ms, image_size=size, response=res.response, latency_s=res.latency_s,
        ply=res.ply, visualization=res.visualization,
        extra={"image_sha1": res.key, "from_cache": res.from_cache, "request_id": res.request_id,
               **({k: stamp[k] for k in ("src_index", "t_clip_ms", "crop", "frame_size", "source") if k in stamp}
                  if stamp else {})},
    )


async def main(args: argparse.Namespace) -> int:
    all_frames = sorted(args.frames.glob("*.jpg"))
    if not all_frames:
        sys.exit(f"no .jpg frames in {args.frames}")
    # Timestamps come from position in the full sorted sequence, so subsets keep real times.
    indexed = list(enumerate(all_frames))[args.start:]
    if args.limit:
        indexed = indexed[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    run = args.run or args.out.name
    todo = [(i, f) for i, f in indexed if args.force or not (args.out / f"{f.stem}.json").exists()]

    cache, budget = FalCache(), Budget(override=args.override_budget)
    # timestamps.json (from extract_frames_by_index.py) beats the uniform-fps assumption
    stamps_path = args.frames / "timestamps.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    items = []  # (t_ms, frame, jpeg, mask, key, size)
    for i, f in todo:
        jpeg = f.read_bytes()
        mpath = args.masks / f"{f.stem}.png" if args.masks else None
        mask = mpath.read_bytes() if mpath and mpath.exists() else None
        with Image.open(io.BytesIO(jpeg)) as im:
            size = list(im.size)
        t_ms = stamps[f.name]["t_ms"] if f.name in stamps else i * 1000.0 / args.fps
        items.append((t_ms, f, jpeg, mask, image_key(jpeg, mask), size))

    hits = [it for it in items if cache.has(it[4])]
    misses = [it for it in items if not cache.has(it[4])]
    new_keys = {it[4] for it in misses if budget.pending_request_id(it[4]) is None}
    recover_keys = {it[4] for it in misses} - new_keys
    print(f"{len(indexed)} frames selected, {len(todo)} to process -> {args.out} (run {run!r})")
    print(f"  cached: {len(hits)}   to recover (already billed): {len(recover_keys)}   "
          f"new: {len(new_keys)} unique ({len(misses) - len(recover_keys)} frames)")
    print(f"  [fal $] {budget.status_line(run)}")

    async with SamBodyClient(live=args.live, run=run, cache=cache, budget=budget,
                             concurrency=args.concurrency) as sam:
        for t_ms, f, jpeg, mask, key, size in hits:
            save(args.out, f, t_ms, size, await sam.reconstruct(jpeg, mask), stamps.get(f.name))

        if misses and not args.live:
            if recover_keys:
                print("  (already-billed results can only be fetched with --live; that fetch is free)")
            print(f"\nMISSING from fal cache ({len(misses)} frames); nothing was sent. Re-run with --live to pay for them:")
            for _, f, _, _, key, _ in misses:
                print(f"  {f.name}  sha1 {key[:12]}{'  (billed, recoverable)' if key in recover_keys else ''}")
            write_summary(args.out, {"last_batch": {"missing": [m[1].name for m in misses]}})
            return 1

        if new_keys:
            try:
                budget.confirm_batch(len(new_keys), run, yes=args.yes)
            except BudgetError as e:
                print(f"\nNOT SENT: {e}")
                return 2

        records, errors = [], []
        t0 = time.perf_counter()

        async def one(t_ms, f, jpeg, mask, key, size):
            try:
                res = await sam.reconstruct(jpeg, mask)
            except BudgetError as e:  # cap reached mid-batch: stop, don't keep trying
                errors.append({"frame": f.name, "error": f"budget: {e}"})
                return
            except (CacheMiss, Exception) as e:  # one bad frame shouldn't kill the batch
                errors.append({"frame": f.name, "error": repr(e)})
                print(f"  {f.name}: ERROR {e!r}")
                return
            try:  # the result is already cached (paid for); a write problem must not lose the batch
                rec = save(args.out, f, t_ms, size, res, stamps.get(f.name))
            except Exception as e:
                errors.append({"frame": f.name, "error": f"saving: {e!r}"})
                print(f"  {f.name}: cached, but saving failed: {e!r}")
                return
            records.append(rec)
            src = "cache" if res.from_cache else f"{res.latency_s:.2f}s"
            print(f"  {f.name}: {src} people={rec['num_people']} V={rec['vertex_count']}"
                  + ("" if rec["usable"] else "  <-- not usable"))

        await asyncio.gather(*(one(*it) for it in misses))
        wall = time.perf_counter() - t0

    summary = write_summary(args.out, {"last_batch": {
        "cached": len(hits), "sent": len(new_keys), "ok": len(records), "wall_s": round(wall, 2), "errors": errors}})
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=1))
    print(f"[fal $] {budget.status_line(run)}")
    if len(summary["vertex_counts"]) > 1:
        print("WARNING: vertex count differs across frames — fixed-topology assumption broken")
    _, confirmed = budget.price()
    if new_keys and not confirmed:
        print(f"\n*** STOP: check the real price before sending more. ***\n"
              f"fal dashboard -> usage & billing: confirm these {budget.billed()} calls cost ~$0.015 each, then\n"
              f"    uv run python tools/fal_budget.py confirm-price <usd_per_call>\n"
              f"Until then only {PROBE_CALLS} calls are allowed in total.")
    return 1 if errors else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=REPLAY_ROOT / "data/frames")
    p.add_argument("--out", type=Path, default=REPLAY_ROOT / "data/fal_out/run")
    p.add_argument("--run", default=None, help="run name for the budget (default: --out folder name)")
    p.add_argument("--masks", type=Path, default=None, help="dir of <stem>.png masks (white = person)")
    p.add_argument("--start", type=int, default=0, help="index of first frame in sorted order")
    p.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    p.add_argument("--fps", type=float, default=3.0, help="extraction rate, for timestamps")
    p.add_argument("--concurrency", type=int, default=None, help="default: REPLAY_FAL_CONCURRENCY or 2")
    p.add_argument("--force", action="store_true", help="re-process frames that already have output (still cache-first)")
    p.add_argument("--live", action="store_true", help="allow billed fal calls for frames not in the cache")
    p.add_argument("--yes", action="store_true", help="don't ask before sending a live batch")
    p.add_argument("--override-budget", action="store_true", help="allow exceeding the 500-total / 80-per-run caps")
    sys.exit(asyncio.run(main(p.parse_args())))
