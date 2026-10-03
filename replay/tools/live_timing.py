"""Simulate frame landing times for a live trial against fal's measured latency.

    uv run python tools/live_timing.py

Uses the real PrioritySemaphore (fal concurrency slots) with time scaled down, so the
scheduling behaviour is the service's own. Per-frame latency is drawn from the measured
probe range (submit + queue + inference, plus the result download).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.fal_client_wrap import PrioritySemaphore  # noqa: E402

SCALE = 25.0  # simulated seconds per real second


def frames(trial_s: float, uniform_fps: float, error_s: float, burst_fps: float, burst_half_s: float):
    """(capture_t, send_t, kind). Pre-error burst frames come from the phone's ring buffer, sent
    when the error is detected; post-error burst frames are sent as captured."""
    out = [(i / uniform_fps, i / uniform_fps, "uniform") for i in range(int(trial_s * uniform_fps))]
    n = int(round(2 * burst_half_s * burst_fps)) + 1
    for k in range(n):
        t = error_s - burst_half_s + k / burst_fps
        out.append((t, max(t, error_s), "burst"))
    return sorted(out, key=lambda f: f[1])


async def simulate(fr, concurrency, lat_lo, lat_hi, priority, rng):
    sem = PrioritySemaphore(concurrency)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    landed = []

    async def one(capture_t, send_t, kind):
        await asyncio.sleep(send_t / SCALE)
        async with sem(0 if (priority and kind == "burst") else 1):
            await asyncio.sleep(rng.uniform(lat_lo, lat_hi) / SCALE)
        landed.append(((loop.time() - t0) * SCALE, kind))

    await asyncio.gather(*(one(*f) for f in fr))
    return landed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--lat", type=float, nargs=2, default=(5.45, 6.95), help="per-frame seconds (min max)")
    p.add_argument("--trial", type=float, default=20.0)
    p.add_argument("--error", type=float, default=12.0)
    p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()
    print(f"trial {a.trial:.0f} s, error at {a.error:.0f} s with a ±0.5 s 10 fps burst, concurrency {a.concurrency}, "
          f"latency {a.lat[0]}–{a.lat[1]} s; times are seconds after the trial ends (median of {a.runs})")
    print(f"{'uniform fps':>11} {'priority':>8} {'frames':>6} {'error replay':>12} {'full replay':>11}  throughput limit")
    for fps in (1.5, 2.0, 3.0):
        for prio in (True, False):
            err, full = [], []
            fr = frames(a.trial, fps, a.error, 10, 0.5)
            for r in range(a.runs):
                landed = asyncio.run(simulate(fr, a.concurrency, *a.lat, prio, random.Random(r)))
                err.append(max(t for t, k in landed if k == "burst") - a.trial)
                full.append(max(t for t, _ in landed) - a.trial)
            cap = a.concurrency / statistics.mean(a.lat)
            print(f"{fps:>11} {'bursts' if prio else 'FIFO':>8} {len(fr):>6} {statistics.median(err):>+11.1f}s "
                  f"{statistics.median(full):>+10.1f}s  {cap:.2f} frames/s vs {len(fr) / a.trial:.2f} offered")


if __name__ == "__main__":
    main()
