"""Hour-one decision gate: SAM foot-vertex jitter during a still stance, per axis.

    uv run python tools/noise_floor.py data/fal_out/still

Feet don't move during stance, so any frame-to-frame motion of foot vertices
is SAM noise. Axes are camera frame: x = image horizontal (side-to-side when the
patient faces the camera), y = vertical, z = depth (forward/back).

Steps: put meshes in the camera frame (conventions.json from convention_check.py),
focal-normalize depth (tz' = tz·f̄/f_i, the pipeline does the same), select foot
vertices near heel/toe/ankle keypoints in the first frame, and track those indices
(fixed topology). Reports per-vertex jitter and foot-centroid jitter, raw and
focal-normalized.

Gate: centroid depth jitter < 1 cm -> forward/back sway + full stability margin are
real features; otherwise AP is "estimated", visualization only.
Units are SAM's (≈ meters) before metric scaling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runio import find_keypoints, load_conventions, load_run, to_camera  # noqa: E402

REPLAY_ROOT = Path(__file__).resolve().parents[1]
GATE_M = 0.01


def per_axis(x: np.ndarray) -> dict[str, float]:
    """x: (F, ..., 3) over frames. Std over frames, median over remaining points, per axis."""
    sd = x.std(axis=0, ddof=1).reshape(-1, 3)
    mad = 1.4826 * np.median(np.abs(x - np.median(x, axis=0)), axis=0).reshape(-1, 3)
    return {
        **{a: float(np.median(sd[:, i])) for i, a in enumerate("xyz")},
        **{f"{a}_robust": float(np.median(mad[:, i])) for i, a in enumerate("xyz")},
    }


def fmt(d: dict[str, float]) -> str:
    return "  ".join(f"{a}={d[a] * 100:5.2f}cm (robust {d[a + '_robust'] * 100:5.2f})" for a in "xyz")


def main(args: argparse.Namespace) -> None:
    frames, names = load_run(args.run_dir)
    if len(frames) < 5:
        sys.exit(f"need >= 5 usable frames, got {len(frames)}")
    conv = load_conventions(args.run_dir) or (
        json.loads(args.conventions.read_text()) if args.conventions.exists() else None
    )
    if conv is None:
        print("WARNING: no conventions.json — assuming mesh and keypoints already in camera frame")
        conv = {"mesh": {"flip": "identity", "add_cam_t": False}, "keypoints_3d": {"flip": "identity", "add_cam_t": False}}
    mc, kc = conv["mesh"], conv["keypoints_3d"]

    V = [fr.verts() for fr in frames]
    if len({len(v) for v in V}) != 1:
        sys.exit(f"vertex counts differ across frames: {sorted({len(v) for v in V})}")
    cam_t = np.stack([fr.cam_t for fr in frames])
    focal = np.array([fr.focal for fr in frames])
    f_bar = np.median(focal)
    cam_t_norm = cam_t.copy()
    cam_t_norm[:, 2] *= f_bar / focal

    # body-relative verts (axes fixed), then translate by raw or focal-normalized cam_t
    body = np.stack([to_camera(v, ct, mc["flip"], mc["add_cam_t"]) - ct for v, ct in zip(V, cam_t)])
    raw = body + cam_t[:, None, :]
    norm = body + cam_t_norm[:, None, :]

    # foot vertex selection on frame 0
    foot_kp = find_keypoints(names, "heel", "toe", "ankle")
    if foot_kp and frames[0].kp3d is not None:
        kp0 = to_camera(frames[0].kp3d, frames[0].cam_t, kc["flip"], kc["add_cam_t"])[foot_kp]
        d = np.linalg.norm(raw[0][:, None, :] - kp0[None], axis=2).min(axis=1)
        foot = np.flatnonzero(d < args.radius)
        how = f"within {args.radius * 100:.0f} cm of {len(foot_kp)} foot keypoints ({', '.join(names[i] for i in foot_kp)})"
    else:
        foot = np.array([], dtype=int)
    if len(foot) < 20:
        # fallback: lowest vertices along the vertical axis (largest y if y-down)
        y = raw[0][:, 1] * (1 if conv.get("camera_y_down", True) is not False else -1)
        foot = np.argsort(y)[-int(len(y) * args.lowest_frac) :]
        how = f"lowest {args.lowest_frac:.0%} of vertices (foot keypoints not found)"
    print(f"{len(frames)} frames over {(frames[-1].t_ms - frames[0].t_ms) / 1000:.1f} s; {len(foot)} foot vertices: {how}")
    print(f"focal: median {f_bar:.1f} px, CV {focal.std() / focal.mean():.2%}; pred_cam_t z std raw {cam_t[:, 2].std() * 100:.2f} cm, normalized {cam_t_norm[:, 2].std() * 100:.2f} cm")

    results = {}
    for label, X in (("raw", raw), ("focal_normalized", norm)):
        fv = X[:, foot]
        results[label] = {"per_vertex": per_axis(fv), "centroid": per_axis(fv.mean(axis=1, keepdims=True))}
        print(f"\n[{label}]")
        print(f"  per-vertex  {fmt(results[label]['per_vertex'])}")
        print(f"  centroid    {fmt(results[label]['centroid'])}")
    if foot_kp and all(fr.kp3d is not None for fr in frames):
        kp = np.stack([to_camera(fr.kp3d, ct, kc["flip"], kc["add_cam_t"]) - fr.cam_t + ctn for fr, ct, ctn in zip(frames, cam_t, cam_t_norm)])[:, foot_kp]
        results["foot_keypoints_normalized"] = per_axis(kp)
        print(f"\n[foot keypoints, focal-normalized]\n  per-kp      {fmt(results['foot_keypoints_normalized'])}")

    c = results["focal_normalized"]["centroid"]
    ap_real = c["z"] < GATE_M
    print("\n=== DECISION GATE ===")
    print(f"depth (z) foot-centroid jitter {c['z'] * 100:.2f} cm vs {GATE_M * 100:.0f} cm threshold")
    print("=> forward/back sway + full stability margin are REAL features" if ap_real
          else "=> AP is ESTIMATED: label it, visualization only; side-to-side metrics still valid")

    out = {
        # bundle-format noise floor (meters, camera axes, before metric scaling)
        "noise_floor": {a: c[a] for a in "xyz"},
        "ap_real": ap_real,
        "gate_m": GATE_M,
        "frames": len(frames),
        "foot_vertices": int(len(foot)),
        "foot_selection": how,
        "focal_cv": float(focal.std() / focal.mean()),
        **results,
    }
    (args.run_dir / "noise_floor.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.run_dir / 'noise_floor.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--conventions", type=Path, default=REPLAY_ROOT / "data/conventions.json")
    p.add_argument("--radius", type=float, default=0.08, help="foot-vertex selection radius (m)")
    p.add_argument("--lowest-frac", type=float, default=0.03)
    main(p.parse_args())
