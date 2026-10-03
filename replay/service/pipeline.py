"""Post-processing glue: run a trial's frames through geometry.py in spec order.

No math lives here — only data plumbing and the choices of *which* points feed each
step (stance frames, foot keypoints, thresholds). Used by the service (bundles) and by
tests/test_pipeline_e2e.py (against synthetic ground truth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from service import geometry as g
from service.runs import DEFAULT_CONVENTIONS, RESERVED_JSON, to_camera

# OpenCV camera (y down, z forward) -> y-up camera (z toward viewer). The floor fit runs in
# the y-up frame so the rotation to +y is small (~camera pitch). In the y-down frame it is
# ~180°, where Rodrigues' axis n×ŷ is dominated by noise and tiny normal errors become
# large yaw errors (mixing side-to-side and forward-back).
Y_UP = np.diag([1.0, -1.0, -1.0])

# Tunables (glue choices, not math)
SMOOTH_SIGMA_S = 0.3
EVENT_MARGIN_S = 1.0  # frames this close to an event are not "stance"
FLOOR_TOL_M = 0.015  # contact threshold for BOS
FLOOR_PERCENTILE = 1.0  # floor height = this percentile of vertex heights above the fitted foot-keypoint plane
SOLE_BAND_M = 0.012  # pass-2 floor fit uses vertices within this of the pass-1 floor
FOOT_HEIGHT_M = 0.08  # vertices below this (aligned, metric) count as foot for the noise floor


@dataclass
class Run:
    """One trial's usable frames, everything in SAM's camera frame."""

    t_s: np.ndarray  # (F,)
    focal: np.ndarray  # (F,)
    cam_t: np.ndarray  # (F,3)
    kp3d: np.ndarray  # (F,K,3)
    verts: np.ndarray  # (F,V,3)
    faces: np.ndarray  # (Fc,3)
    kp_names: list[str]
    stems: list[str]
    vis_files: list[str | None]


def load_run(run_dir: Path, conventions: dict | None = None) -> Run:
    """Load a run folder (send_frames / service / synthetic format) into the camera frame."""
    conv = conventions or _find_conventions(run_dir) or DEFAULT_CONVENTIONS
    mc, kc = conv["mesh"], conv["keypoints_3d"]
    recs = sorted(
        (json.loads(p.read_text()) for p in run_dir.glob("*.json") if p.name not in RESERVED_JSON),
        key=lambda r: r["t_ms"],
    )
    recs = [r for r in recs if r["usable"]]
    if not recs:
        raise ValueError(f"no usable frames in {run_dir}")

    t, focal, cam_t, kp3d, verts, stems, vis = [], [], [], [], [], [], []
    faces = None
    for r in recs:
        person = r["metadata"]["people"][0]
        ct = np.asarray(person["pred_cam_t"], float)
        stem = Path(r["frame"]).stem
        mesh = trimesh.load(run_dir / f"{stem}.ply", file_type="ply", process=False)
        if faces is None:
            faces = np.asarray(mesh.faces)
        elif len(mesh.vertices) != len(verts[0]):
            raise ValueError(f"{stem}: vertex count {len(mesh.vertices)} != {len(verts[0])} (topology must be fixed)")
        t.append(r["t_ms"] / 1000)
        focal.append(float(person["focal_length"]))
        cam_t.append(ct)
        kp3d.append(to_camera(np.asarray(person["keypoints_3d"], float)[:, :3], ct, kc["flip"], kc["add_cam_t"]))
        verts.append(to_camera(np.asarray(mesh.vertices, float), ct, mc["flip"], mc["add_cam_t"]))
        stems.append(stem)
        vis.append(r.get("vis_file"))
    return Run(
        t_s=np.array(t), focal=np.array(focal), cam_t=np.stack(cam_t), kp3d=np.stack(kp3d),
        verts=np.stack(verts), faces=faces, kp_names=recs[0]["metadata"].get("keypoint_names") or [],
        stems=stems, vis_files=vis,
    )


def _find_conventions(run_dir: Path) -> dict | None:
    for p in (run_dir / "conventions.json", Path(__file__).resolve().parents[1] / "data/conventions.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def _kp(names: list[str], *patterns: str) -> list[int]:
    return [i for i, n in enumerate(names) if any(s in n.lower() for s in patterns)]


def stance_mask(t_s: np.ndarray, events: list[dict], margin_s: float = EVENT_MARGIN_S) -> np.ndarray:
    ev = np.array([e["t"] / 1000 for e in events])
    if ev.size == 0:
        return np.ones(len(t_s), bool)
    return np.abs(t_s[:, None] - ev[None]).min(axis=1) > margin_s


def process(run: Run, events: list[dict], patient_height_m: float) -> dict[str, Any]:
    """Spec math pipeline, steps 2–11. Returns arrays for the bundle plus diagnostics."""
    names, t = run.kp_names, run.t_s
    stance = stance_mask(t, events)
    if stance.sum() < 3:
        raise ValueError("fewer than 3 stance frames")

    # 2. focal normalization: shift each frame along camera z to the normalized depth
    shift = g.normalize_focal(run.cam_t, run.focal) - run.cam_t
    V = (run.verts + shift[:, None, :]) @ Y_UP
    K = (run.kp3d + shift[:, None, :]) @ Y_UP

    # 3. outlier rejection needs MediaPipe landmarks (not wired yet) -> keep all frames

    # 4. floor fit. Pass 1: heel/toe keypoints from stance frames (spec). In tandem stance
    # these lie nearly on one line, so roll is poorly constrained; pass 2 refits on the
    # sole vertices near that floor, which span the full foot width.
    foot_kp = _kp(names, "heel", "toe")
    head_kp = _kp(names, "nose", "head", "eye")
    floor_pts = K[stance][:, foot_kp].reshape(-1, 3)
    up_hint = K[stance][:, head_kp].reshape(-1, 3).mean(0) - floor_pts.mean(0)
    centroid, n = g.fit_plane(floor_pts, up_hint)
    R = g.rotation_to_y(n)
    heights = g.align_to_floor(V[stance], R, centroid)[..., 1]
    floor_y = float(np.median(np.percentile(heights, FLOOR_PERCENTILE, axis=1)))
    sole = heights < floor_y + SOLE_BAND_M
    if sole.sum() >= 3 * len(floor_pts):
        sole_pts = V[stance][sole]
        centroid, n = g.fit_plane(sole_pts, up_hint)
        R = g.rotation_to_y(n)
        heights = g.align_to_floor(V[stance], R, centroid)[..., 1]
        floor_y = float(np.median(np.percentile(heights, FLOOR_PERCENTILE, axis=1)))
        floor_source = "sole_vertices"
    else:
        floor_source = "foot_keypoints"
    # origin: mean of per-foot centers (heel ↔ mean of toes), on the floor
    foot_centers = []
    for side in ("left", "right"):
        heel, toes = _kp(names, f"{side}_heel"), _kp(names, f"{side}_big_toe", f"{side}_small_toe")
        if heel and toes:
            foot_centers.append((K[stance][:, heel].mean((0, 1)) + K[stance][:, toes].mean((0, 1))) / 2)
    center = np.mean(foot_centers, axis=0) if foot_centers else centroid
    center_floor = g.align_to_floor(center, R, centroid)
    origin = centroid + R.T @ np.array([center_floor[0], floor_y, center_floor[2]])
    Va = g.align_to_floor(V, R, origin)

    # 5. metric scale
    s = g.metric_scale(patient_height_m, Va[stance][..., 1].max(axis=1))
    Va = Va * s

    # 6. temporal smoothing (non-uniform timestamps)
    Vs = g.gaussian_smooth(t, Va, SMOOTH_SIGMA_S)

    # 7–9. COM, BOS, stability margin per frame
    com = np.stack([g.center_of_mass(v, run.faces) for v in Vs])
    bos = [g.support_hull(g.contact_points(v, FLOOR_TOL_M)) for v in Vs]
    margin = np.array([g.stability_margin(c[[0, 2]], h) for c, h in zip(com, bos)])
    comps = np.array([g.margin_components(c[[0, 2]], h) for c, h in zip(com, bos)])

    # 10. sway heatmap; 11. noise floor on (unsmoothed) stance foot vertices
    heatmap = g.sway_heatmap(Vs)
    foot_v = Va[stance][..., 1].mean(axis=0) < FOOT_HEIGHT_M
    nf = g.noise_floor(Va[stance][:, foot_v])

    return {
        "t_s": t,
        "verts": Vs,
        "faces": run.faces,
        "com": com,
        "bos": bos,
        "margin": margin,
        "margin_ml": comps[:, 0],
        "margin_ap": comps[:, 1],
        "heatmap": heatmap,
        "noise_floor": nf,
        # reported in SAM's (OpenCV) camera frame
        "floor_normal_cam": Y_UP @ n,
        "origin_cam": Y_UP @ origin,
        "R_cam_to_floor": R @ Y_UP,
        "scale": s,
        "quality": {
            "aligned": True,
            "frames": len(t),
            "stance_frames": int(stance.sum()),
            "floor_points": int(sole.sum()) if floor_source == "sole_vertices" else len(floor_pts),
            "floor_source": floor_source,
            "watertight": g.is_watertight(run.faces),
            "scale": float(s),
            "ap_real": bool(nf["z"] < 0.01),
        },
    }


def raw_display(run: Run) -> dict[str, Any]:
    """Fallback when geometry.py isn't available: camera-frame verts flipped to y-up, no metrics."""
    return {
        "t_s": run.t_s,
        "verts": run.verts @ Y_UP,
        "faces": run.faces,
        "quality": {"aligned": False, "frames": len(run.t_s), "frame": "camera, y-up (unaligned)"},
    }
