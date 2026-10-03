# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe==0.10.21", "numpy<2", "opencv-python-headless>=4.9,<4.11", "matplotlib>=3.8"]
# ///
"""Scout a recorded clip locally with MediaPipe (free) before spending fal credit.

    uv run tools/scout_clip.py data/IMG_9691.mov

Runs in its own environment (inline metadata above): MediaPipe 0.10.21 on CPU. Newer
MediaPipe (1.0.x) on macOS initialises Metal even with the CPU delegate and aborts
where no GPU service is available.

Pass 1 (cached in data/scout/<clip>/timeline.npz): every frame through MediaPipe Pose
(VIDEO mode, up to 2 people, segmentation mask), plus camera motion measured by
tracking background features (person masked out) against the first frame.
Pass 2: report — camera static? feet visible? framing, longest planted stretch
(noise-floor window), candidate lift/step events, and footage problems.

Also writes landmarks_2d.json (t_ms + 33 landmarks per frame, full-res pixels): the
MediaPipe 2D timeline the pipeline can use for outlier rejection.

Needs data/models/pose_landmarker_full.task (MediaPipe's pose_landmarker_full model).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPLAY_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPLAY_ROOT / "data/models/pose_landmarker_full.task"

# MediaPipe Pose landmark indices
NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = 25, 26, 27, 28, 29, 30, 31, 32
FOOT = {"left": (L_ANK, L_HEEL, L_TOE), "right": (R_ANK, R_HEEL, R_TOE)}
NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
         "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
         "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
         "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]


# --------------------------------------------------------------------------------- pass 1

def probe(clip: Path) -> tuple[int, int, np.ndarray]:
    """Display (rotated) size and per-frame timestamps in ms."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation", "-of", "json", str(clip)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    rot = next((abs(int(d["rotation"])) for d in st.get("side_data_list", []) if "rotation" in d), 0)
    w, h = (st["height"], st["width"]) if rot in (90, 270) else (st["width"], st["height"])
    ts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(clip)],
                        capture_output=True, text=True, check=True).stdout.split()
    return w, h, np.array([float(x.split(",")[0]) * 1000 for x in ts if x.split(",")[0]])


def extract(clip: Path, out_dir: Path, scale: float) -> Path:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    W, H, t_ms = probe(clip)
    w, h = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    print(f"{clip.name}: {W}x{H} displayed, {len(t_ms)} frames, {t_ms[-1] / 1000:.1f} s; analysing at {w}x{h}")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL), delegate=BaseOptions.Delegate.CPU),
                                 running_mode=RunningMode.VIDEO,
                                 num_poses=2, output_segmentation_masks=True)
    F = len(t_ms)
    lm = np.full((F, 33, 4), np.nan, np.float32)  # x px, y px (full res), visibility, presence
    world = np.full((F, 33, 3), np.nan, np.float32)  # meters, hip-centred
    n_people = np.zeros(F, np.int8)
    cam = np.full((F, 4), np.nan, np.float32)  # dx, dy (full-res px), rotation deg, scale vs frame 0
    shake = np.full(F, np.nan, np.float32)  # background shift vs previous sampled frame (px)
    orb = cv2.ORB_create(1500)
    ref = prev = None
    with PoseLandmarker.create_from_options(opts) as det:
        for i in range(F):
            buf = ff.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                F = i
                break
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t_ms[i]))
            n_people[i] = len(res.pose_landmarks)
            person = np.zeros((h, w), np.uint8)
            if res.pose_landmarks:
                for j, p in enumerate(res.pose_landmarks[0]):
                    lm[i, j] = (p.x * W, p.y * H, p.visibility, p.presence)
                for j, p in enumerate(res.pose_world_landmarks[0]):
                    world[i, j] = (p.x, p.y, p.z)
                for m in res.segmentation_masks or []:
                    v = m.numpy_view()
                    person |= ((v[..., 0] if v.ndim == 3 else v) > 0.3).astype(np.uint8)
            if i % 3 == 0:  # camera motion on background features, every 3rd frame
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                bg = (1 - cv2.dilate(person, np.ones((25, 25), np.uint8))) * 255
                kp, des = orb.detectAndCompute(gray, bg)
                if ref is None:
                    ref = (kp, des)
                for target, store in ((ref, "ref"), (prev, "prev")):
                    if target is None or des is None or target[1] is None:
                        continue
                    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(target[1], des)
                    if len(matches) < 12:
                        continue
                    a = np.float32([target[0][m.queryIdx].pt for m in matches])
                    b = np.float32([kp[m.trainIdx].pt for m in matches])
                    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is None or inl.sum() < 10:
                        continue
                    sc = float(np.hypot(M[0, 0], M[1, 0]))
                    if store == "ref":
                        cam[i] = (M[0, 2] / scale, M[1, 2] / scale, np.degrees(np.arctan2(M[1, 0], M[0, 0])), sc)
                    else:
                        shake[i] = float(np.hypot(M[0, 2], M[1, 2]) / scale)
                prev = (kp, des)
            if i % 300 == 0:
                print(f"  frame {i}/{len(t_ms)}")
    ff.stdout.close()
    ff.wait()
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "timeline.npz"
    np.savez_compressed(npz, t_ms=t_ms[:F], lm=lm[:F], world=world[:F], n_people=n_people[:F], cam=cam[:F],
                        shake=shake[:F], size=np.array([W, H]))
    with open(out_dir / "landmarks_2d.json", "w") as f:
        json.dump({"source": clip.name, "image_size": [W, H], "names": NAMES, "t_ms": t_ms[:F].round(1).tolist(),
                   "xyv": np.round(lm[:F, :, :3], 2).tolist()}, f)
    return npz


# --------------------------------------------------------------------------------- pass 2

def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    m = np.concatenate([[False], mask, [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2]))


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps of at most max_gap frames inside True runs."""
    out = mask.copy()
    for s_, e_ in runs(~mask):
        if 0 < s_ and e_ < len(mask) and e_ - s_ <= max_gap:
            out[s_:e_] = True
    return out


def hysteresis(x: np.ndarray, up: float, down: float) -> np.ndarray:
    out = np.zeros(len(x), bool)
    on = False
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = on
            continue
        on = v > up if not on else v > down
        out[i] = on
    return out


def smooth_nan(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average along axis 0, ignoring NaNs."""
    if k <= 1:
        return x.copy()
    v = np.nan_to_num(x)
    w = (~np.isnan(x)).astype(float)
    ker = np.ones(k)
    num = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, v)
    den = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def fmt(t_ms) -> str:
    return f"{t_ms / 1000:5.2f}s"


def analyze(npz: Path, a: argparse.Namespace) -> dict:
    d = np.load(npz)
    t, lm, n_people, cam, shake = d["t_ms"], d["lm"], d["n_people"], d["cam"], d["shake"]
    # (world landmarks used for lift detection below)
    W, H = d["size"]
    F = len(t)
    fps = (F - 1) / ((t[-1] - t[0]) / 1000)
    report: dict = {"clip": npz.parent.name, "frames": F, "duration_s": round((t[-1] - t[0]) / 1000, 2),
                    "fps": round(fps, 2), "size": [int(W), int(H)], "problems": [], "notes": []}
    P, N = report["problems"], report["notes"]

    # --- detection & people
    det = ~np.isnan(lm[:, NOSE, 0]) | ~np.isnan(lm[:, L_HIP, 0])
    report["person_detected_pct"] = round(100 * det.mean(), 1)
    if det.mean() < 0.98:
        P.append(f"person not detected in {100 * (1 - det.mean()):.1f}% of frames: "
                 + ", ".join(f"{fmt(t[s])}-{fmt(t[e - 1])}" for s, e in runs(~det)[:6]))
    multi = n_people >= 2
    if multi.any():
        P.append(f"second person detected in {multi.sum()} frames ({100 * multi.mean():.1f}%): "
                 + ", ".join(f"{fmt(t[s])}-{fmt(t[e - 1])}" for s, e in runs(multi)[:6]) + " — SAM may pick the wrong one; send a mask or crop")

    # --- camera static
    shift = np.hypot(cam[:, 0], cam[:, 1])
    ok = ~np.isnan(shift)
    report["camera"] = {
        "sampled_frames": int(ok.sum()),
        "max_shift_px": round(float(np.nanmax(shift)), 1) if ok.any() else None,
        "p95_shift_px": round(float(np.nanpercentile(shift, 95)), 1) if ok.any() else None,
        "max_rotation_deg": round(float(np.nanmax(np.abs(cam[:, 2]))), 2) if ok.any() else None,
        "max_scale_change_pct": round(float(np.nanmax(np.abs(cam[:, 3] - 1)) * 100), 2) if ok.any() else None,
        "max_frame_to_frame_px": round(float(np.nanmax(shake)), 1) if np.isfinite(shake).any() else None,
    }
    if ok.mean() < 0.2 and ok.any() is False:
        P.append("camera motion could not be measured (too few background features)")
    elif ok.any() and (np.nanmax(shift) > a.cam_tol_px or np.nanmax(np.abs(cam[:, 2])) > 0.5):
        moved = shift > a.cam_tol_px
        P.append(f"camera moved: up to {np.nanmax(shift):.1f} px / {np.nanmax(np.abs(cam[:, 2])):.2f}° vs the first frame"
                 + (" at " + ", ".join(f"{fmt(t[s])}-{fmt(t[e - 1])}" for s, e in runs(moved)[:5]) if moved.any() else ""))

    # --- feet visibility & framing
    vis = {s: np.nanmin(lm[:, list(idx), 2], axis=1) for s, idx in FOOT.items()}
    inside = {s: np.all((lm[:, list(idx), 0] > 0) & (lm[:, list(idx), 0] < W) & (lm[:, list(idx), 1] > 0)
                        & (lm[:, list(idx), 1] < H), axis=1) for s, idx in FOOT.items()}
    feet_ok = (vis["left"] > a.vis) & (vis["right"] > a.vis) & inside["left"] & inside["right"] & det
    report["feet_visible_pct"] = round(100 * feet_ok.mean(), 1)
    report["foot_visibility_median"] = {s: round(float(np.nanmedian(v)), 3) for s, v in vis.items()}
    if feet_ok.mean() < 0.98:
        P.append(f"feet not clearly visible in {100 * (1 - feet_ok.mean()):.1f}% of frames: "
                 + ", ".join(f"{fmt(t[s])}-{fmt(t[e - 1])}" for s, e in runs(~feet_ok)[:6]))
    ys = lm[det, :, 1]
    top = np.nanmin(lm[det][:, :11, 1], axis=1)  # face landmarks; real head top is ~0.07 H above
    feet_y = np.nanmax(lm[det][:, 27:, 1], axis=1)
    body_px = feet_y - top
    report["framing"] = {
        "person_height_px_median": int(np.nanmedian(body_px)),
        "person_height_pct_of_frame": round(100 * float(np.nanmedian(body_px)) / H, 1),
        "min_margin_top_px": int(np.nanmin(top)), "min_margin_bottom_px": int(H - np.nanmax(feet_y)),
        "min_margin_left_px": int(np.nanmin(lm[det][:, :, 0])), "min_margin_right_px": int(W - np.nanmax(lm[det][:, :, 0])),
    }
    if np.nanmin(top) < 0.06 * H:
        P.append(f"head close to the top edge (face landmarks {int(np.nanmin(top))} px from it) — top of head may be cut off")
    if H - np.nanmax(feet_y) < 0.02 * H:
        P.append(f"feet within {int(H - np.nanmax(feet_y))} px of the bottom edge")
    if np.nanmedian(body_px) < 0.4 * H:
        N.append("person is small in frame; SAM detail and keypoint accuracy will suffer")

    # --- crop box for fal (fixed per trial)
    x0, x1 = np.nanpercentile(lm[det][:, :, 0], 0.5), np.nanpercentile(lm[det][:, :, 0], 99.5)
    y0, y1 = np.nanpercentile(top, 0.5) - 0.12 * np.nanmedian(body_px), np.nanpercentile(feet_y, 99.5)
    mx, my = 0.15 * (x1 - x0), 0.06 * (y1 - y0)
    box = [max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my)]
    report["suggested_fixed_crop"] = [int(v) for v in box]

    # --- planted feet. Heel + toe only: the ankle landmark swings with the shin during
    # ankle-strategy sway even when the foot is flat. Per foot: smoothed centre speed is low
    # and the foot isn't lifted; a stretch must also not creep (drift <= tol from its median).
    leg = np.nanmedian(np.hypot(*(lm[:, [L_HIP, R_HIP], :2].mean(1) - lm[:, [L_ANK, R_ANK], :2].mean(1)).T))
    report["leg_length_px"] = round(float(leg), 1)
    k = max(1, int(round(fps * 0.2)))
    sm = smooth_nan(lm[:, [L_HEEL, L_TOE, R_HEEL, R_TOE], :2], k)  # (F, 4, 2)
    centre = {"left": sm[:, 0:2].mean(1), "right": sm[:, 2:4].mean(1)}
    lowest = {"left": np.nanmax(sm[:, 0:2, 1], 1), "right": np.nanmax(sm[:, 2:4, 1], 1)}
    raw_c = {"left": lm[:, [L_HEEL, L_TOE], :2].mean(1), "right": lm[:, [R_HEEL, R_TOE], :2].mean(1)}
    jitter = float(np.nanmedian([np.nanstd(raw_c[f] - centre[f], axis=0) for f in FOOT]))
    tol = max(a.plant_tol * leg, 4 * jitter)
    report["foot_jitter_px"] = round(jitter, 2)
    report["planted_tolerance_px"] = round(float(tol), 1)
    # Lift from MediaPipe world landmarks (meters, y down): each foot's lowest point vs the
    # other foot's. Image-space height is fooled by depth (a foot placed further back looks
    # higher). Resting offset between the feet removed; hysteresis up/down thresholds.
    wy = smooth_nan(d["world"][:, [L_HEEL, L_TOE, R_HEEL, R_TOE], 1], k)
    low_w = {"left": np.nanmax(wy[:, 0:2], 1), "right": np.nanmax(wy[:, 2:4], 1)}
    rel = low_w["right"] - low_w["left"]  # > 0: left foot higher than right
    calm = (np.abs(rel - np.nanmedian(rel)) < a.lift_down_m) & feet_ok
    rel = rel - (np.nanmedian(rel[calm]) if calm.any() else 0.0)
    height = {"left": rel, "right": -rel}  # height of this foot above the other (m)
    lifted = {f: hysteresis(height[f], a.lift_up_m, a.lift_down_m) for f in FOOT}
    floor = {f: height[f] for f in FOOT}  # kept for plotting
    speed = {f: np.r_[0, np.linalg.norm(np.diff(centre[f], axis=0), axis=1)] * fps / leg for f in FOOT}  # leg/s
    still = {f: (speed[f] < a.still_speed) & ~lifted[f] for f in FOOT}
    both = still["left"] & still["right"] & feet_ok
    both = close_gaps(both, int(fps * 0.2))
    stretches = []
    for s_, e_ in runs(both):
        while e_ - s_ > fps:  # trim until the stretch doesn't creep
            seg = np.concatenate([centre["left"][s_:e_], centre["right"][s_:e_]], axis=1)
            dev = np.nanmax(np.abs(seg - np.nanmedian(seg, axis=0)), axis=1)
            if np.nanmax(dev) <= tol:
                break
            worst = int(np.nanargmax(dev))  # index within [s_, e_)
            if worst < (e_ - s_) / 2:
                s_ = s_ + worst + 1
            else:
                e_ = s_ + worst
        if (t[e_ - 1] - t[s_]) >= 1000:
            stretches.append((s_, e_))
    stretches.sort(key=lambda r: t[r[0]])
    report["planted_stretches_1s"] = [[round(t[x] / 1000, 2), round(t[y - 1] / 1000, 2), round((t[y - 1] - t[x]) / 1000, 2)]
                                      for x, y in stretches]
    if stretches:
        s, e = max(stretches, key=lambda r: t[r[1] - 1] - t[r[0]])
    else:
        s, e = 0, 1
    report["longest_planted"] = {"start_s": round(t[s] / 1000, 2), "end_s": round(t[e - 1] / 1000, 2),
                                 "duration_s": round((t[e - 1] - t[s]) / 1000, 2), "frames_30fps": int(e - s),
                                 "frames_at_3fps": int((t[e - 1] - t[s]) / 1000 * 3) + 1}
    if (t[e - 1] - t[s]) / 1000 < 5:
        P.append(f"no 5 s stretch with both feet planted (longest {(t[e - 1] - t[s]) / 1000:.1f} s at {fmt(t[s])}) — "
                 "noise floor would rest on too few frames; record ~10 s of still stance")
    seg = lm[s:e][:, [L_HEEL, L_TOE, R_HEEL, R_TOE], :2]
    jit = np.nanmedian(np.nanstd(seg, axis=0), axis=0)
    report["mediapipe_foot_jitter_px_in_planted"] = {"x": round(float(jit[0]), 2), "y": round(float(jit[1]), 2)}
    report["_series"] = {"centre": centre, "lowest": lowest, "height": height, "lifted": lifted, "still": still,
                         "both": both, "feet_ok": feet_ok, "stretches": stretches}

    # --- candidate events: lift = lowest foot point rises above its rolling floor; step = the
    # foot's planted position relocates (median before vs after differ by > step_frac × leg)
    events = []
    for f in FOOT:
        for s_, e_ in runs(close_gaps(lifted[f] & feet_ok, int(fps * 0.15))):
            if (t[e_ - 1] - t[s_]) >= 150:
                events.append({"t_start_s": round(t[s_] / 1000, 2), "t_end_s": round(t[e_ - 1] / 1000, 2),
                               "kind": "foot_lift", "side": f,
                               "peak_height_cm": round(float(np.nanmax(height[f][s_:e_])) * 100, 1)})
                # inside a lift, a dip close to the floor may be a brief touchdown / toe touch
                dip = height[f][s_:e_] < a.touch_dip_m
                for ds, de in runs(dip):
                    if (t[s_ + de - 1] - t[s_ + ds]) >= 150 and 0 < ds and de < e_ - s_:
                        events.append({"t_start_s": round(t[s_ + ds] / 1000, 2), "t_end_s": round(t[s_ + de - 1] / 1000, 2),
                                       "kind": "possible_touchdown", "side": f,
                                       "min_height_cm": round(float(np.nanmin(height[f][s_ + ds:s_ + de])) * 100, 1)})
        st_runs = [r for r in runs(still[f]) if (t[r[1] - 1] - t[r[0]]) >= 400]
        for (s0, e0), (s1, e1) in zip(st_runs, st_runs[1:]):
            move = float(np.linalg.norm(np.nanmedian(centre[f][s1:e1], 0) - np.nanmedian(centre[f][s0:e0], 0)))
            if move > a.step_frac * leg:
                events.append({"t_start_s": round(t[e0 - 1] / 1000, 2), "t_end_s": round(t[s1] / 1000, 2),
                               "kind": "step", "side": f, "moved_leg_frac": round(move / leg, 3)})
    report["candidate_events"] = sorted(events, key=lambda ev: (ev["t_start_s"], ev["kind"]))
    if not events:
        N.append("no foot lifts or steps detected; fine for a noise floor, but no error moments to replay")

    # --- whole-body sway (hip-centre motion) as context
    hipc = lm[:, [L_HIP, R_HIP], :2].mean(1)
    sway = np.nanstd(hipc, axis=0)
    report["hip_sway_std_px"] = {"x": round(float(sway[0]), 1), "y": round(float(sway[1]), 1)}
    return report


def plot(report: dict, npz: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz)
    t = d["t_ms"] / 1000
    ser = report["_series"]
    leg = report["leg_length_px"]
    fig, ax = plt.subplots(4, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1, 1]})
    colors = {"left": "#2563eb", "right": "#dc2626"}
    for f, c in colors.items():
        ax[0].plot(t, ser["height"][f] * 100, c, lw=1, label=f"{f} foot height above the other (cm, MediaPipe world)")
        ax[1].plot(t, (ser["centre"][f][:, 0] - np.nanmedian(ser["centre"][f][:, 0])) / leg, c, lw=1, label=f"{f} foot x")
    hip = d["lm"][:, [23, 24], 0].mean(1)
    ax[1].plot(t, (hip - np.nanmedian(hip)) / leg, "#6b7280", lw=1, label="hip centre x")
    ax[2].plot(t, np.nanmin(d["lm"][:, [29, 31], 2], 1), colors["left"], lw=1)
    ax[2].plot(t, np.nanmin(d["lm"][:, [30, 32], 2], 1), colors["right"], lw=1)
    ax[2].axhline(0.3, color="k", lw=0.8, ls="--")
    ax[2].set_ylabel("foot visibility")
    cam = d["cam"]
    ok = ~np.isnan(cam[:, 0])
    ax[3].plot(t[ok], np.hypot(cam[ok, 0], cam[ok, 1]), "#059669", lw=1)
    ax[3].set_ylabel("camera shift px")
    for a_ in ax:
        for s_, e_ in ser["stretches"]:
            a_.axvspan(t[s_], t[e_ - 1], color="#22c55e", alpha=0.18)
        for s_, e_ in runs(~ser["feet_ok"]):
            a_.axvspan(t[s_], t[e_ - 1], color="#f59e0b", alpha=0.15)
    for ev in report["candidate_events"]:
        c = colors[ev["side"]]
        ax[0].axvspan(ev["t_start_s"], ev["t_end_s"], ymin=0.92 if ev["kind"] == "step" else 0.85,
                      ymax=1.0 if ev["kind"] == "step" else 0.92, color=c, alpha=0.8)
    ax[0].axhline(6, color="k", lw=0.8, ls="--")
    ax[0].set_ylabel("foot height (cm)")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_ylabel("x offset (leg lengths)")
    ax[-1].set_xlabel("time (s)   green = both feet planted, amber = feet not clearly visible, bars = events (top: step, lower: lift)")
    ax[0].set_title(f"{report['clip']}: scout timeline")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: data/scout/<clip stem>/")
    p.add_argument("--scale", type=float, default=0.5, help="analysis resolution (fraction of full)")
    p.add_argument("--redo", action="store_true", help="re-run MediaPipe even if the timeline is cached")
    p.add_argument("--vis", type=float, default=0.3, help="min landmark visibility for 'visible'")
    p.add_argument("--plant-tol", type=float, default=0.03, help="planted: foot points stay within this × leg length")
    p.add_argument("--lift-up-m", type=float, default=0.06, help="lift starts when a foot is this far above the other (m)")
    p.add_argument("--lift-down-m", type=float, default=0.03, help="lift ends below this (m)")
    p.add_argument("--touch-dip-m", type=float, default=0.10, help="flag dips below this during a lift (m)")
    p.add_argument("--step-frac", type=float, default=0.08, help="step: foot centre relocates this × leg length")
    p.add_argument("--still-speed", type=float, default=0.25, help="planted: foot centre speed below this (leg lengths/s)")
    p.add_argument("--cam-tol-px", type=float, default=8.0, help="camera counts as static below this shift (px)")
    a = p.parse_args()
    if not MODEL.exists():
        sys.exit(f"missing {MODEL} (MediaPipe pose_landmarker_full.task)")
    out = a.out or REPLAY_ROOT / "data/scout" / a.clip.stem
    npz = out / "timeline.npz"
    if a.redo or not npz.exists():
        extract(a.clip, out, a.scale)
    rep = analyze(npz, a)
    plot(rep, npz, out / "timeline.png")
    rep.pop("_series")
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"\nwrote {out / 'report.json'}, {out / 'timeline.png'}, {out / 'landmarks_2d.json'}")


if __name__ == "__main__":
    main()
