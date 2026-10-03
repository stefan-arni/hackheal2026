# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy<2", "opencv-python-headless>=4.9,<4.11"]
# ///
"""Floor normal from floorboard vanishing points (free, no fal), compared with SAM's floor fit.

    uv run tools/floor_vanishing.py data/IMG_9691.mov --frames 1745,1825,1905 \\
        --crop 75,429,1045,1920 --focal 1384 --sam-normal 0.0129,-0.9989,-0.0448

Per frame (person masked out): RANSAC vanishing point of the room's vertical edges (walls,
pillar, furniture; above the floor) -> up = K⁻¹v_vertical in the TRUE camera (principal point
at the full-frame center). The floorboards' vanishing point gives a floor direction d = K⁻¹v
that must be perpendicular to up: a check on the focal length and on the vertical VP.
(Board-end joints are too short and few for a reliable second floor direction.)
SAM assumes the principal point is at the center of the image it was given (the crop), so
its camera frame is rotated relative to the true one; n is rotated into SAM's frame (the
true ray through the crop center becomes SAM's optical axis) before comparing.
Axes: OpenCV (x right, y down, z forward); normals point up (y < 0).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

REPLAY_ROOT = Path(__file__).resolve().parents[1]


def read_frame(clip: Path, index: int) -> np.ndarray:
    W, H = 1080, 1920
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"select=eq(n\\,{index}),format=gray",
                          "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W)


def segments(gray: np.ndarray, person_box: tuple[int, int, int, int], floor_top: int, min_len: float):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(cv2.GaussianBlur(gray, (3, 3), 0))[0]
    if lines is None:
        return np.zeros((0, 4))
    L = lines[:, 0, :]
    mid = (L[:, :2] + L[:, 2:]) / 2
    length = np.linalg.norm(L[:, 2:] - L[:, :2], axis=1)
    x0, y0, x1, y1 = person_box
    in_person = (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)
    keep = (length >= min_len) & (mid[:, 1] > floor_top) & ~in_person
    return L[keep]


def vp_ransac(L: np.ndarray, iters: int = 3000, tol_deg: float = 1.0, seed: int = 0):
    """Vanishing point (homogeneous) of a family of segments: RANSAC on pairwise intersections,
    inliers = segments pointing at the VP within tol, then least-squares refinement."""
    rng = np.random.default_rng(seed)
    p1 = np.c_[L[:, :2], np.ones(len(L))]
    p2 = np.c_[L[:, 2:], np.ones(len(L))]
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    mid = (L[:, :2] + L[:, 2:]) / 2
    seg_dir = (L[:, 2:] - L[:, :2]) / np.linalg.norm(L[:, 2:] - L[:, :2], axis=1, keepdims=True)
    w = np.linalg.norm(L[:, 2:] - L[:, :2], axis=1)

    def inliers(v):
        if abs(v[2]) < 1e-9:  # VP at infinity: direction (vx, vy)
            d = np.tile(v[:2] / np.linalg.norm(v[:2]), (len(L), 1))
        else:
            d = v[:2] / v[2] - mid
            d /= np.linalg.norm(d, axis=1, keepdims=True)
        cosang = np.abs(np.sum(d * seg_dir, axis=1))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1))) < tol_deg

    best, best_n = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(L), 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if np.linalg.norm(v) < 1e-12:
            continue
        v = v / np.linalg.norm(v)
        m = inliers(v)
        if (n := int((w * m).sum())) > best_n:
            best, best_n = m, n
    A = lines[best] * w[best, None]
    v = np.linalg.svd(A)[2][-1]
    return v / np.linalg.norm(v), best


def rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    k = np.cross(a, b)
    s, c = np.linalg.norm(k), a @ b
    if s < 1e-12:
        return np.eye(3)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def pitch_roll(n_up: np.ndarray) -> tuple[float, float]:
    """Camera pitch (degrees, + = looking down) and roll from an up normal in camera coordinates.
    Looking down tilts the optical axis away from 'up', so up · forward = n_z = -sin(pitch)."""
    pitch = -np.degrees(np.arcsin(np.clip(n_up[2], -1, 1)))
    roll = np.degrees(np.arctan2(n_up[0], -n_up[1]))
    return float(pitch), float(roll)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", type=Path)
    p.add_argument("--frames", default="1825")
    p.add_argument("--crop", default="75,429,1045,1920")
    p.add_argument("--focal", type=float, default=1384.0)
    p.add_argument("--person-box", default=None, help="x0,y0,x1,y1 full-frame px to ignore (default: from crop)")
    p.add_argument("--floor-top", type=int, default=1250, help="floorboard segments: below this row")
    p.add_argument("--wall-bottom", type=int, default=1150, help="vertical edges: above this row")
    p.add_argument("--max-perp-err", type=float, default=2.0, help="drop frames whose floorboards aren't ⟂ up")
    p.add_argument("--sam-normal", default=None, help="SAM floor normal in SAM's camera frame (up, OpenCV axes)")
    p.add_argument("--out", type=Path, default=REPLAY_ROOT / "data/scout/IMG_9691/floor_vanishing")
    a = p.parse_args()
    W, H = 1080, 1920
    f = a.focal
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)
    cx0, cy0, cx1, cy1 = (int(v) for v in a.crop.split(","))
    crop_c = np.array([(cx0 + cx1) / 2, (cy0 + cy1) / 2, 1.0])
    R_true_to_sam = rot_a_to_b(Kinv @ crop_c, np.array([0, 0, 1.0]))
    box = tuple(int(v) for v in a.person_box.split(",")) if a.person_box else (330, 400, 760, 1820)
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for idx in (int(x) for x in a.frames.split(",")):
        gray = read_frame(a.clip, idx)
        L = segments(gray, box, a.floor_top, min_len=35)
        ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
        long_fam = L[np.abs(ang - 90) < 40]  # boards running away from the camera (fan toward a VP)
        v1, in1 = vp_ransac(long_fam)
        U = segments(gray, box, 0, min_len=60)
        U = U[((U[:, 1] + U[:, 3]) / 2) < a.wall_bottom]  # above the floor
        angU = np.degrees(np.arctan2(U[:, 3] - U[:, 1], U[:, 2] - U[:, 0])) % 180
        vert = U[np.abs(angU - 90) < 12]
        vv, inv = vp_ransac(vert, tol_deg=0.6)
        up = Kinv @ vv
        up /= np.linalg.norm(up)
        if up[1] > 0:
            up = -up  # vertical VP may be above or below; up has y < 0 in OpenCV axes
        d1 = Kinv @ v1
        d1 /= np.linalg.norm(d1)
        perp_err = float(abs(90 - np.degrees(np.arccos(np.clip(abs(up @ d1), -1, 1)))))
        n = up - (up @ d1) * d1  # enforce floor direction ⟂ up (small correction)
        n /= np.linalg.norm(n)
        n_sam = R_true_to_sam @ n
        pr_true, pr_sam = pitch_roll(n), pitch_roll(n_sam)
        rec = {"frame": idx,
               "segments": {"floorboards": int(len(long_fam)), "floorboard_inliers": int(in1.sum()),
                            "vertical": int(len(vert)), "vertical_inliers": int(inv.sum())},
               "vp_floorboards_px": (v1[:2] / v1[2]).round(1).tolist() if abs(v1[2]) > 1e-9 else "infinity",
               "vp_vertical_px": (vv[:2] / vv[2]).round(0).tolist() if abs(vv[2]) > 1e-9 else "infinity",
               "floorboards_vs_up_perpendicularity_err_deg": round(perp_err, 2),
               "normal_true_cam": n.round(4).tolist(), "pitch_roll_true_deg": [round(x, 2) for x in pr_true],
               "normal_sam_cam": n_sam.round(4).tolist(), "pitch_roll_sam_deg": [round(x, 2) for x in pr_sam]}
        trans_fam, in2 = vert, inv  # drawn in the debug image
        results.append(rec)
        # debug image
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for seg, col, inl in ((long_fam, (0, 200, 255), in1), (trans_fam, (255, 120, 0), in2)):
            for s_, ok in zip(seg, inl):
                cv2.line(vis, tuple(int(x) for x in s_[:2]), tuple(int(x) for x in s_[2:]), col if ok else (90, 90, 90), 3)
        cv2.rectangle(vis, box[:2], box[2:], (0, 0, 255), 3)
        cv2.imwrite(str(a.out / f"lines_{idx}.jpg"), cv2.resize(vis, (540, 960)))
    good = [r for r in results if r["floorboards_vs_up_perpendicularity_err_deg"] < a.max_perp_err]
    if not good:
        raise SystemExit("no frame passed the floorboard-perpendicularity check")
    normals = np.array([r["normal_sam_cam"] for r in good])
    n_mean = np.median(normals, axis=0)
    n_mean /= np.linalg.norm(n_mean)
    true_mean = np.median(np.array([r["normal_true_cam"] for r in good]), axis=0)
    true_mean /= np.linalg.norm(true_mean)
    spread = max(float(np.degrees(np.arccos(np.clip(n @ n_mean, -1, 1)))) for n in normals)
    summary = {"frames": results, "frames_used": [r["frame"] for r in good], "max_spread_deg": round(spread, 2),
               "normal_true_cam": true_mean.round(4).tolist(), "pitch_roll_true_deg": [round(x, 2) for x in pitch_roll(true_mean)],
               "normal_sam_cam_mean": n_mean.round(4).tolist(),
               "pitch_roll_sam_deg": [round(x, 2) for x in pitch_roll(n_mean)],
               "principal_point_offset_deg": round(float(np.degrees(np.arccos((Kinv @ crop_c / np.linalg.norm(Kinv @ crop_c))[2]))), 2)}
    if a.sam_normal:
        ns = np.array([float(x) for x in a.sam_normal.split(",")])
        ns /= np.linalg.norm(ns)
        summary["sam_fit_normal"] = ns.round(4).tolist()
        summary["sam_fit_pitch_roll_deg"] = [round(x, 2) for x in pitch_roll(ns)]
        summary["angle_vp_vs_sam_fit_deg"] = round(float(np.degrees(np.arccos(np.clip(n_mean @ ns, -1, 1)))), 2)
    (a.out / "result.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
