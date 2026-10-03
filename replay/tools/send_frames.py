"""Send extracted frames through fal and save meshes, metadata and visualizations.

    uv run python tools/send_frames.py                          # all of data/frames
    uv run python tools/send_frames.py --limit 3 --out data/fal_out/probe
    uv run python tools/send_frames.py --start 6 --limit 20 --out data/fal_out/still

Per frame in --out: <stem>.ply, <stem>.json, <stem>_vis.<ext>. Plus summary.json
with latency, people counts and vertex-count consistency. Already-processed
frames are skipped (use --force to redo); each frame costs ~$0.015.
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

from service.fal_client_wrap import SamBodyClient  # noqa: E402
from service.runs import write_frame, write_summary  # noqa: E402


async def process_frame(
    sam: SamBodyClient, frame: Path, mask: Path | None, t_ms: float, out: Path
) -> dict:
    jpeg = frame.read_bytes()
    with Image.open(io.BytesIO(jpeg)) as im:
        size = list(im.size)
    res = await sam.reconstruct(jpeg, mask.read_bytes() if mask else None)
    return write_frame(
        out,
        frame.stem,
        t_ms=t_ms,
        image_size=size,
        response=res.response,
        latency_s=res.latency_s,
        ply=res.ply,
        visualization=res.visualization,
    )


async def main(args: argparse.Namespace) -> None:
    all_frames = sorted(args.frames.glob("*.jpg"))
    if not all_frames:
        sys.exit(f"no .jpg frames in {args.frames}")
    # Timestamps come from position in the full sorted sequence, so subsets keep real times.
    indexed = list(enumerate(all_frames))[args.start :]
    if args.limit:
        indexed = indexed[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [(i, f) for i, f in indexed if args.force or not (args.out / f"{f.stem}.json").exists()]
    print(f"{len(indexed)} frames selected, {len(todo)} to send -> {args.out}")

    records: list[dict] = []
    errors: list[dict] = []
    t0 = time.perf_counter()
    async with SamBodyClient(concurrency=args.concurrency) as sam:

        async def one(i: int, frame: Path) -> None:
            mask = args.masks / f"{frame.stem}.png" if args.masks else None
            if mask is not None and not mask.exists():
                mask = None
            try:
                rec = await process_frame(sam, frame, mask, i * 1000.0 / args.fps, args.out)
            except Exception as e:  # keep going; one bad frame shouldn't kill the batch
                errors.append({"frame": frame.name, "error": repr(e)})
                print(f"  {frame.name}: ERROR {e!r}")
                return
            records.append(rec)
            flag = "" if rec["usable"] else "  <-- not usable"
            print(
                f"  {frame.name}: {rec['latency_s']:.2f}s people={rec['num_people']} "
                f"V={rec['vertex_count']}{flag}"
            )

        await asyncio.gather(*(one(i, f) for i, f in todo))
    wall = time.perf_counter() - t0

    summary = write_summary(
        args.out,
        {"last_batch": {"sent": len(todo), "ok": len(records), "wall_s": round(wall, 2), "errors": errors}},
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=1))
    if len(summary["vertex_counts"]) > 1:
        print("WARNING: vertex count differs across frames — fixed-topology assumption broken")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=REPLAY_ROOT / "data/frames")
    p.add_argument("--out", type=Path, default=REPLAY_ROOT / "data/fal_out/run")
    p.add_argument("--masks", type=Path, default=None, help="dir of <stem>.png masks (white = person)")
    p.add_argument("--start", type=int, default=0, help="index of first frame in sorted order")
    p.add_argument("--limit", type=int, default=0, help="max frames to send (0 = all)")
    p.add_argument("--fps", type=float, default=3.0, help="extraction rate, for timestamps")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--force", action="store_true", help="re-send frames that already have output")
    asyncio.run(main(p.parse_args()))
