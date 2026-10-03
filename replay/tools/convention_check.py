"""Hour-one convention check: which coordinate frame are keypoints_3d and mesh vertices in?

    uv run python tools/convention_check.py data/fal_out/probe

1. Keypoints: project keypoints_3d with a pinhole camera
       u = f·X/Z + W/2,   v = f·Y/Z + H/2
   under each axis-flip variant, with and without adding pred_cam_t, and compare
   to keypoints_2d. As-is wins → camera space; +cam_t wins → body-relative.
2. Mesh: with keypoints in the camera frame, find the transform of the .ply
   vertices that puts the keypoints on the mesh (median nearest-vertex distance).
3. Axis signs: is image-down +y (head above feet ⇒ smaller y)? Body height in mesh units.

Writes conventions.json into the run dir and data/conventions.json (read by noise_floor.py).
Self-contained on purpose so it runs before geometry.py is implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runio import FLIPS, find_keypoints, load_run, to_camera  # noqa: E402

REPLAY_ROOT = Path(__file__).resolve().parents[1]


def project(P: np.ndarray, f: float, W: int, H: int) -> np.ndarray:
    Z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = f * P[:, :2] / Z[:, None] + np.array([W / 2, H / 2])
    uv[Z <= 0] = np.nan  # behind the camera
    return uv


def main(args: argparse.Namespace) -> None:
    frames, names = load_run(args.run_dir)
    frames = [f for f in frames if f.kp3d is not None][: args.max_frames]
    if not frames:
        sys.exit("no usable frames with keypoints_3d — check include_3d_keypoints / include_mhr_params")
    print(f"{len(frames)} frames, {frames[0].W}x{frames[0].H}, {len(names)} keypoint names")
    if args.names:
        for i, n in enumerate(names):
            print(f"  {i:2d} {n}")

    # --- 1. keypoints_3d vs keypoints_2d ----------------------------------
    rows = []
    for flip in FLIPS:
        for add in (False, True):
            errs, offsets = [], []
            for fr in frames:
                uv = project(to_camera(fr.kp3d, fr.cam_t, flip, add), fr.focal, fr.W, fr.H)
                d = uv - fr.kp2d
                errs.append(np.nanmedian(np.linalg.norm(d, axis=1)))
                offsets.append(np.nanmedian(d, axis=0))
            rows.append((float(np.nanmedian(errs)), flip, add, np.nanmedian(offsets, axis=0)))
    rows.sort(key=lambda r: (np.isnan(r[0]), r[0]))

    print("\nkeypoints_3d -> 2D reprojection (median px error over keypoints, then frames)")
    for err, flip, add, off in rows:
        print(f"  {flip:9s} {'+cam_t' if add else 'as-is ':6s}  {err:9.2f} px   mean offset (du,dv)=({off[0]:+.1f},{off[1]:+.1f})")
    kp_err, kp_flip, kp_add, kp_off = rows[0]
    diag = float(np.hypot(frames[0].W, frames[0].H))
    verdict = "camera space" if not kp_add else "body-relative (add pred_cam_t)"
    print(f"=> keypoints_3d are {verdict}, axes {kp_flip}; error {kp_err:.1f} px = {100 * kp_err / diag:.2f}% of diagonal")
    if kp_err > 0.02 * diag:
        print("   WARNING: best error > 2% of image diagonal — no variant fits; check principal point / crop / image size")
    if np.hypot(*kp_off) > 0.01 * diag:
        print(f"   NOTE: systematic offset {kp_off.round(1)} px — principal point may not be (W/2, H/2)")

    # --- 2. mesh vertices ------------------------------------------------
    mesh_rows = []
    verts = [fr.verts() for fr in frames[: min(5, len(frames))]]
    for flip in FLIPS:
        for add in (False, True):
            ds = []
            for fr, V in zip(frames, verts):
                kp_cam = to_camera(fr.kp3d, fr.cam_t, kp_flip, kp_add)
                tree = cKDTree(to_camera(V, fr.cam_t, flip, add))
                ds.append(np.median(tree.query(kp_cam)[0]))
            mesh_rows.append((float(np.median(ds)), flip, add))
    mesh_rows.sort()
    print("\nmesh vertices: median distance from camera-frame keypoints to nearest vertex")
    for d, flip, add in mesh_rows[:6]:
        print(f"  {flip:9s} {'+cam_t' if add else 'as-is ':6s}  {d * 100:8.2f} cm")
    m_d, m_flip, m_add = mesh_rows[0]
    print(f"=> mesh vertices: axes {m_flip}, {'add pred_cam_t' if m_add else 'already camera frame'}")
    print(f"   vertex count {len(verts[0])}, identical across checked frames: {len({len(v) for v in verts}) == 1}")

    # --- 3. axis signs & scale -------------------------------------------
    fr0 = frames[0]
    kp_cam = to_camera(fr0.kp3d, fr0.cam_t, kp_flip, kp_add)
    V_cam = to_camera(verts[0], fr0.cam_t, m_flip, m_add)
    head = find_keypoints(names, "nose", "head", "eye")
    feet = find_keypoints(names, "ankle", "heel", "toe")
    y_down = None
    if head and feet:
        y_down = bool(kp_cam[head, 1].mean() < kp_cam[feet, 1].mean())
        print(f"\nhead y={kp_cam[head, 1].mean():+.3f}, feet y={kp_cam[feet, 1].mean():+.3f} -> camera y points {'DOWN' if y_down else 'UP'}")
    else:
        print("\ncould not find head/foot keypoints by name; rerun with --names")
    ext = V_cam.max(0) - V_cam.min(0)
    print(f"mesh extent (x,y,z) = {ext.round(3)}  (expect ~body height on the vertical axis if meters)")
    print(f"focal {np.median([f.focal for f in frames]):.1f} px, pred_cam_t z {np.median([f.cam_t[2] for f in frames]):.2f}")

    result = {
        "keypoints_3d": {"flip": kp_flip, "add_cam_t": kp_add, "median_px": kp_err, "offset_px": kp_off.tolist()},
        "mesh": {"flip": m_flip, "add_cam_t": m_add, "median_nn_m": m_d},
        "camera_y_down": y_down,
        "image_size": [fr0.W, fr0.H],
        "vertex_count": len(verts[0]),
        "mesh_extent": ext.tolist(),
        "frames_checked": len(frames),
    }
    for out in (args.run_dir / "conventions.json", REPLAY_ROOT / "data/conventions.json"):
        out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {args.run_dir / 'conventions.json'} and data/conventions.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--max-frames", type=int, default=10)
    p.add_argument("--names", action="store_true", help="print all keypoint names")
    main(p.parse_args())
