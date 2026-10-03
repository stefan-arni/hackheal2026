"""Replay a run's frames to the service in real time, like the capture page would.

    uv run python tools/fake_phone.py                                    # SYNTHETIC run, 3 fps real time
    uv run python tools/fake_phone.py data/fal_out/synthetic --speed 4   # faster

Sends input/<stem>.jpg at each frame's timestamp (phone clock = wall-clock ms at start
+ t_ms), POSTs /end with the events (from ground_truth.json when present, as the
metrics pipeline would), then polls /status and reports how long after the trial ended
the replay was ready.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service.runs import load_records  # noqa: E402


def main(a: argparse.Namespace) -> None:
    recs = [r for r in load_records(a.run_dir) if (a.run_dir / "input" / r["frame"]).exists()]
    if not recs:
        sys.exit(f"no frames with input/<stem>.jpg in {a.run_dir}")
    gt_path = a.run_dir / "ground_truth.json"
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else {}
    if gt.get("SYNTHETIC"):
        print("*** sending SYNTHETIC frames ***")
    trial = a.trial or f"fake-{int(time.time())}"
    W, H = recs[0]["image_size"]
    crop = [0, 0, W, H]  # fixed for the whole trial
    phone_t0 = time.time() * 1000  # phone clock at trial start
    t0_rel = recs[0]["t_ms"]

    with httpx.Client(base_url=a.url, timeout=30) as http:
        wall0 = time.perf_counter()
        for r in recs:
            due = (r["t_ms"] - t0_rel) / 1000 / a.speed
            time.sleep(max(0.0, due - (time.perf_counter() - wall0)))
            resp = http.post(
                f"/replay/{trial}/frame",
                files={"jpeg": (r["frame"], (a.run_dir / "input" / r["frame"]).read_bytes(), "image/jpeg")},
                data={"t": str(phone_t0 + r["t_ms"]), "crop": json.dumps(crop),
                      "frame_size": json.dumps([W, H]), "kind": "uniform"},
            )
            resp.raise_for_status()
            ack = resp.json()
            print(f"  t={r['t_ms'] / 1000:5.2f}s -> {ack['accepted']} inflight={ack['inflight']}")

        events = [{**e, "t": phone_t0 + e["t"]} for e in gt.get("events", [])]
        height_cm = a.height_cm or 100 * gt.get("patient_height_m", 1.75)
        t_end = time.perf_counter()
        print(f"trial end after {t_end - wall0:.1f}s: {http.post(f'/replay/{trial}/end', json={'events': events, 'patient_height_cm': height_cm}).json()}")

        while True:
            st = http.get(f"/replay/{trial}/status").json()
            if st["state"] in ("ready", "failed"):
                break
            time.sleep(0.2)
    print(f"\n{st['state'].upper()} {time.perf_counter() - t_end:.2f}s after trial end: {json.dumps(st)}")
    if st["state"] == "ready":
        print(f"bundle on disk: data/trials/{trial}/  ->  {a.url}/viewer/?src=/replay/{trial}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, nargs="?", default=REPLAY_ROOT / "data/fal_out/synthetic")
    p.add_argument("--url", default="http://localhost:8017")
    p.add_argument("--trial", default=None)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--height-cm", type=float, default=None)
    main(p.parse_args())
