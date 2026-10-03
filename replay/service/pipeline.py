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
FLOOR_TOL_M = 0.015  # minimum contact threshold for BOS (spec); widened to CONTACT_NOISE_K × vertical noise
CONTACT_NOISE_K = 3.0  # real SAM: ~0.9–1.3 cm vertical foot noise -> ~3 cm band; synthetic (3 mm) keeps 1.5 cm
FLOOR_TOL_MAX_M = 0.035
FLOOR_PERCENTILE = 1.0  # floor height = this percentile of vertex heights above the fitted foot-keypoint plane
SOLE_BAND_M = 0.012  # pass-2 floor fit uses vertices within this of the pass-1 floor
FOOT_HEIGHT_M = 0.08
GROUND_ANCHOR_MAX_M = 0.05  # re-anchor a frame's lowest point to the floor if it floats/sinks less than this
GROUND_ANCHOR_PCT = 0.2  # "lowest point" = this percentile of vertex heights (robust to a few spikes)  # vertices below this (aligned, metric) count as foot for the noise floor


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
    image_size: np.ndarray | None = None  # (F,2) [W, H] of the image sent to fal (the crop)
    crop: np.ndarray | None = None  # (F,4) crop box in full-frame pixels, None if unknown
    t_clip_ms: np.ndarray | None = None  # (F,) source-clip time, for matching MediaPipe timelines


def subset(run: Run, keep: np.ndarray) -> Run:
    """The same run restricted to frames where keep is True."""
    idx = np.flatnonzero(keep)
    pick = lambda a: None if a is None else a[idx]  # noqa: E731
    return Run(t_s=run.t_s[idx], focal=run.focal[idx], cam_t=run.cam_t[idx], kp3d=run.kp3d[idx],
               verts=run.verts[idx], faces=run.faces, kp_names=run.kp_names,
               stems=[run.stems[i] for i in idx], vis_files=[run.vis_files[i] for i in idx],
               image_size=pick(run.image_size), crop=pick(run.crop), t_clip_ms=pick(run.t_clip_ms))


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
    sizes, crops, t_clip = [], [], []
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
        sizes.append(r.get("image_size"))
        crops.append(r.get("crop"))
        t_clip.append(r.get("t_clip_ms"))
    return Run(
        t_s=np.array(t), focal=np.array(focal), cam_t=np.stack(cam_t), kp3d=np.stack(kp3d),
        verts=np.stack(verts), faces=faces,
        kp_names=[norm_name(n) for n in recs[0]["metadata"].get("keypoint_names") or []],
        stems=stems, vis_files=vis,
        image_size=np.array(sizes, float) if all(x is not None for x in sizes) else None,
        crop=np.array(crops, float) if all(x is not None for x in crops) else None,
        t_clip_ms=np.array(t_clip, float) if all(x is not None for x in t_clip) else None,
    )


def _find_conventions(run_dir: Path) -> dict | None:
    for p in (run_dir / "conventions.json", Path(__file__).resolve().parents[1] / "data/conventions.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def norm_name(name: str) -> str:
    """fal's MHR names are hyphenated ("left-big-toe-tip"); compare as "left_big_toe_tip"."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _kp(names: list[str], *patterns: str) -> list[int]:
    pats = [norm_name(p) for p in patterns]
    return [i for i, n in enumerate(names) if any(p in norm_name(n) for p in pats)]


# SAM (MHR, normalized names) -> MediaPipe Pose landmark names, for reprojection checks
MP_PAIRS = [("nose", "nose")] + [
    (f"{s}_{j}", f"{s}_{j}") for s in ("left", "right")
    for j in ("eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel")
] + [("left_big_toe_tip", "left_foot_index"), ("right_big_toe_tip", "right_foot_index")]
OUTLIER_MIN_PX = 20.0  # never call a frame an outlier below this median joint error
OUTLIER_FACTOR = 3.0  # ... or below this × the trial's median error


def reprojection_vs_mediapipe(run: Run, landmarks: dict, min_vis: float = 0.5) -> np.ndarray:
    """Per-frame median pixel error (F,) between SAM's keypoints (projected, full-frame px) and
    MediaPipe's 2D landmarks at the nearest timestamp. NaN where no comparison is possible."""
    if run.crop is None or run.image_size is None or run.t_clip_ms is None:
        raise ValueError("run lacks crop / image_size / t_clip_ms (frames not from extract_frames_by_index)")
    mp_names = list(landmarks["names"])
    mp_xyv = np.asarray(landmarks["xyv"], float)  # (M, 33, 3) full-frame px + visibility
    pairs = [(run.kp_names.index(a), mp_names.index(b)) for a, b in MP_PAIRS
             if a in run.kp_names and b in mp_names]
    sam_i, mp_i = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    near = g.nearest_timestamp_index(run.t_clip_ms, np.asarray(landmarks["t_ms"], float))
    err = np.full(len(run.t_s), np.nan)
    for f in range(len(run.t_s)):
        W, H = run.image_size[f]
        to_full = (run.crop[f, 2] - run.crop[f, 0]) / W  # >1 when the frame was downscaled for fal
        uv = g.project_pinhole(run.kp3d[f, sam_i], run.focal[f], W, H) * to_full + run.crop[f, :2]
        ref = mp_xyv[near[f], mp_i]
        ok = ref[:, 2] >= min_vis
        if ok.sum() >= 5:
            err[f] = float(np.nanmedian(np.linalg.norm(uv[ok] - ref[ok, :2], axis=1)))
    return err


def stance_mask(t_s: np.ndarray, events: list[dict], margin_s: float = EVENT_MARGIN_S) -> np.ndarray:
    ev = np.array([e["t"] / 1000 for e in events])
    if ev.size == 0:
        return np.ones(len(t_s), bool)
    return np.abs(t_s[:, None] - ev[None]).min(axis=1) > margin_s


def process(
    run: Run,
    events: list[dict],
    patient_height_m: float | None,
    *,
    landmarks: dict | None = None,
    stance_intervals_s: list[tuple[float, float]] | None = None,
    up_cam: np.ndarray | None = None,
    up_source: str = "external",
    up_check: np.ndarray | None = None,
) -> dict[str, Any]:
    """Spec math pipeline, steps 2–11. Returns arrays for the bundle plus diagnostics.

    landmarks: MediaPipe 2D timeline (scout's landmarks_2d.json) -> step 3 outlier rejection.
    stance_intervals_s: [t0, t1] (same clock as run.t_s) where both feet are planted; default is
    every frame more than EVENT_MARGIN_S from an event. patient_height_m None keeps SAM's scale.
    """
    quality: dict[str, Any] = {}
    # 3. outlier rejection: SAM keypoints vs MediaPipe 2D at the nearest timestamp
    if landmarks is not None:
        err = reprojection_vs_mediapipe(run, landmarks)
        thr = max(OUTLIER_MIN_PX, OUTLIER_FACTOR * float(np.nanmedian(err)))
        keep = g.inlier_mask(err, thr)
        quality["reprojection"] = {
            "median_px": round(float(np.nanmedian(err)), 1), "p95_px": round(float(np.nanpercentile(err, 95)), 1),
            "threshold_px": round(thr, 1), "dropped": [s for s, k in zip(run.stems, keep) if not k],
            "per_frame_px": {s: (None if np.isnan(e) else round(float(e), 1)) for s, e in zip(run.stems, err)},
        }
        run = subset(run, keep)
    names, t = run.kp_names, run.t_s
    if stance_intervals_s is not None:
        stance = np.zeros(len(t), bool)
        for t0, t1 in stance_intervals_s:
            stance |= (t >= t0) & (t <= t1)
    else:
        stance = stance_mask(t, events)
    if stance.sum() < 3:
        raise ValueError(f"fewer than 3 stance frames ({int(stance.sum())})")

    # 2. focal normalization: shift each frame along camera z to the normalized depth
    shift = g.normalize_focal(run.cam_t, run.focal) - run.cam_t
    V = (run.verts + shift[:, None, :]) @ Y_UP
    K = (run.kp3d + shift[:, None, :]) @ Y_UP

    # 4. floor fit. Pass 1: heel/toe keypoints from stance frames (spec). In tandem stance
    # these lie nearly on one line, so roll is poorly constrained; pass 2 refits on the
    # sole vertices near that floor, which span the full foot width.
    foot_kp = _kp(names, "heel", "toe")
    head_kp = _kp(names, "nose", "head", "eye")
    floor_pts = K[stance][:, foot_kp].reshape(-1, 3)
    up_hint = K[stance][:, head_kp].reshape(-1, 3).mean(0) - floor_pts.mean(0)
    if up_cam is not None:  # independent gravity: only height / origin come from the feet
        centroid = floor_pts.mean(axis=0)
        n = Y_UP @ (np.asarray(up_cam, float) / np.linalg.norm(up_cam))
        if n @ up_hint < 0:
            raise ValueError("up_cam points toward the feet; it must point up (y < 0 in OpenCV axes)")
        R = g.rotation_to_y(n)
        heights = g.align_to_floor(V[stance], R, centroid)[..., 1]
        floor_y = float(np.median(np.percentile(heights, FLOOR_PERCENTILE, axis=1)))
        sole = heights < floor_y + SOLE_BAND_M
        fit_c, fit_n = g.fit_plane(V[stance][sole], up_hint)  # for the report only
        quality["feet_fit_vs_up_deg"] = round(float(np.degrees(np.arccos(np.clip(fit_n @ n, -1, 1)))), 2)
        floor_source = up_source
    else:
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
    if up_check is not None:
        nc = Y_UP @ (np.asarray(up_check, float) / np.linalg.norm(up_check))
        quality["gravity_vs_feet_deg"] = round(float(np.degrees(np.arccos(np.clip(nc @ n, -1, 1)))), 2)
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
    Ka = g.align_to_floor(K, R, origin)  # keypoints follow the mesh through every step

    # 5. metric scale
    heights = Va[stance][..., 1].max(axis=1)
    s = g.metric_scale(patient_height_m, heights) if patient_height_m else 1.0
    quality["mesh_height_m"] = round(float(np.median(heights)), 3)
    Va = Va * s
    Ka = Ka * s

    # Ground contact prior: in a balance trial at least one foot is on the floor. SAM's per-frame
    # vertical noise (~1–3 cm) makes the stance foot float or sink, which empties or floods the
    # 1.5 cm contact band; shift each frame so its lowest point sits on the floor (|shift| < 5 cm).
    lowest = np.percentile(Va[..., 1], GROUND_ANCHOR_PCT, axis=1)
    anchor = np.where(np.abs(lowest) < GROUND_ANCHOR_MAX_M, lowest, 0.0)
    Va = Va - anchor[:, None, None] * np.array([0.0, 1.0, 0.0])
    Ka = Ka - anchor[:, None, None] * np.array([0.0, 1.0, 0.0])
    quality["ground_anchor_cm"] = {"median_abs": round(float(np.median(np.abs(anchor))) * 100, 2),
                                   "max_abs": round(float(np.max(np.abs(anchor))) * 100, 2),
                                   "frames_not_anchored": int((np.abs(lowest) >= GROUND_ANCHOR_MAX_M).sum())}

    # 6. temporal smoothing (non-uniform timestamps)
    Vs = g.gaussian_smooth(t, Va, SMOOTH_SIGMA_S)
    Ks = g.gaussian_smooth(t, Ka, SMOOTH_SIGMA_S)

    # 11 (early). noise floor on unsmoothed stance foot vertices; it also sets the contact band
    foot_v = Va[stance][..., 1].mean(axis=0) < FOOT_HEIGHT_M
    nf = g.noise_floor(Va[stance][:, foot_v])
    tol = float(np.clip(CONTACT_NOISE_K * nf["y"], FLOOR_TOL_M, FLOOR_TOL_MAX_M))
    quality["contact_band_cm"] = round(tol * 100, 2)

    # 7–9. COM, BOS, stability margin per frame
    com = np.stack([g.center_of_mass(v, run.faces) for v in Vs])
    bos = [g.support_hull(g.contact_points(v, tol)) for v in Vs]
    margin = np.array([g.stability_margin(c[[0, 2]], h) for c, h in zip(com, bos)])
    comps = np.array([g.margin_components(c[[0, 2]], h) for c, h in zip(com, bos)])

    # 10. sway heatmap
    heatmap = g.sway_heatmap(Vs)

    return {
        "t_s": t,
        "verts": Vs,
        "verts_raw": Va,  # per-frame fit before smoothing: what lines up with the video frame
        "keypoints": Ks,  # (F,K,3) floor frame, metric, smoothed; names in kp_names
        "kp_names": names,
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
        # floor frame -> SAM's raw camera frame (OpenCV), per frame: p_cam = M @ p_floor + t_f.
        # Inverts every step (anchor, scale, rotation, origin, Y_UP, focal normalization), so
        # projecting with that frame's SAM focal and its image center lands on the video frame.
        "camera": {
            "M": (Y_UP @ R.T / s).tolist(),
            "t": [(Y_UP @ (R.T @ (np.array([0.0, a, 0.0]) / s) + origin) - sh).tolist()
                  for a, sh in zip(anchor, shift)],
            "focal": run.focal.tolist(),
            "image_size": None if run.image_size is None else run.image_size.tolist(),
            "crop": None if run.crop is None else run.crop.tolist(),
        },
        "stems": run.stems,
        "vis_files": run.vis_files,
        "quality": {
            **quality,
            "aligned": True,
            "frames": len(t),
            "stance_frames": int(stance.sum()),
            "floor_points": int(sole.sum()) if floor_source == "sole_vertices" else len(floor_pts),
            "floor_source": floor_source,
            "watertight": g.is_watertight(run.faces),
            "scale": float(s),
            "patient_height_cm": round(patient_height_m * 100, 1) if patient_height_m else None,
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
