"""Generate a SYNTHETIC run with known ground truth, in send_frames.py's output format.

    uv run python tools/make_synthetic_run.py                  # -> data/fal_out/synthetic/
    uv run python tools/make_synthetic_run.py --body-relative  # mesh/keypoints3d without pred_cam_t

*** SYNTHETIC DATA — not a real person, not real fal output. ***

Scene (world frame: y up, floor y = 0, origin = mean stance-foot center, patient faces
the camera along +z, patient's left = +x = image right):
  - crude humanoid from boxes/capsules/sphere, ~1.75 m tall, fixed topology,
    tandem stance (left foot in front), hands on hips
  - inverted-pendulum sway about the ankles (~1 cm side-to-side, ~1.5 cm forward-back)
  - optional slow lean toward +x from 10.5 s (COM drifts toward the BOS edge before the error)
  - rear (right) foot lifts at 12.0 s for 1.0 s
Camera: OpenCV convention (x right, y down, z forward), 2.5 m away at 1.3 m height,
pitched down at the hips, pinhole u = f·X/Z + W/2, v = f·Y/Z + H/2.
SAM-like noise per frame: focal estimate off by up to ±5 % with the coupled depth error
(tz_i = tz · f_i / f), plus ~3 mm iid jitter on every vertex and keypoint.

Outputs: per-frame .ply/.json/_vis.png + summary.json (same as send_frames.py),
input/<stem>.jpg (what a phone would send; used by fake_phone.py), ground_truth.json.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service.runs import write_frame, write_summary  # noqa: E402

LABEL = "SYNTHETIC"

# --- scene constants ------------------------------------------------------------

W, H = 720, 1280  # image sent to fal (fixed crop)
F_TRUE = 1300.0  # px
CAM_POS = np.array([0.0, 1.3, 2.5])
CAM_TARGET = np.array([0.0, 0.9, 0.0])

ANKLE_L = np.array([0.0, 0.08, 0.06])  # front foot
ANKLE_R = np.array([0.0, 0.08, -0.20])  # rear foot
PIVOT = (ANKLE_L + ANKLE_R) / 2  # inverted-pendulum axis point
HIP_L, HIP_R = np.array([0.09, 0.92, 0.0]), np.array([-0.09, 0.92, 0.0])
KNEE_L, KNEE_R = np.array([0.06, 0.50, 0.05]), np.array([-0.06, 0.50, -0.07])
SHOULDER_L, SHOULDER_R = np.array([0.20, 1.45, 0.0]), np.array([-0.20, 1.45, 0.0])
ELBOW_L, ELBOW_R = np.array([0.30, 1.18, -0.06]), np.array([-0.30, 1.18, -0.06])
WRIST_L, WRIST_R = np.array([0.19, 1.00, -0.02]), np.array([-0.19, 1.00, -0.02])
HEAD_C, HEAD_R = np.array([0.0, 1.635, 0.0]), 0.115  # top of head at 1.75 m

LIFT_START_S, LIFT_DUR_S, LIFT_HEIGHT_M, LIFT_RAMP_S = 12.0, 1.0, 0.06, 0.15


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World->camera rotation, OpenCV axes (rows = camera x right, y down, z forward in world)."""
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


R_WC = look_at(CAM_POS, CAM_TARGET)


def to_cam(p: np.ndarray) -> np.ndarray:
    return (p - CAM_POS) @ R_WC.T


# --- humanoid ------------------------------------------------------------------


def capsule(p0, p1, r) -> trimesh.Trimesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c = trimesh.creation.capsule(height=np.linalg.norm(p1 - p0), radius=r, count=[12, 12])
    T = trimesh.geometry.align_vectors([0, 0, 1], p1 - p0)
    T[:3, 3] = (p0 + p1) / 2
    return c.apply_transform(T)


def box(extents, center, subdiv=0) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents)
    for _ in range(subdiv):
        b = b.subdivide()
    return b.apply_translation(center)


@dataclass
class Body:
    verts: np.ndarray  # (V,3) rest pose, world
    faces: np.ndarray  # (Fc,3)
    group: np.ndarray  # (V,) 0 = swaying body, 1 = left foot (fixed), 2 = right foot+shin (lifts)
    parts: dict[str, list[int]]  # part name -> [start, stop) vertex range
    kp: np.ndarray  # (70,3) rest pose keypoints
    kp_group: np.ndarray  # (70,)
    kp_names: list[str]


def build_body() -> Body:
    parts = [
        ("left_foot", box([0.09, 0.07, 0.26], [0.0, 0.035, 0.13], 3), 1),
        ("right_foot", box([0.09, 0.07, 0.26], [0.0, 0.035, -0.13], 3), 2),
        ("right_shin", capsule(ANKLE_R + [0, 0.04, 0], KNEE_R, 0.05), 2),
        ("left_shin", capsule(ANKLE_L + [0, 0.04, 0], KNEE_L, 0.05), 0),
        ("left_thigh", capsule(KNEE_L, HIP_L, 0.07), 0),
        ("right_thigh", capsule(KNEE_R, HIP_R, 0.07), 0),
        ("pelvis", box([0.32, 0.20, 0.20], [0.0, 0.95, 0.0], 1), 0),
        ("torso", box([0.36, 0.50, 0.21], [0.0, 1.25, 0.0], 1), 0),
        ("neck", capsule([0, 1.48, 0], [0, 1.56, 0], 0.05), 0),
        ("head", trimesh.creation.icosphere(subdivisions=2, radius=HEAD_R).apply_translation(HEAD_C), 0),
        ("left_upper_arm", capsule(SHOULDER_L, ELBOW_L, 0.045), 0),
        ("right_upper_arm", capsule(SHOULDER_R, ELBOW_R, 0.045), 0),
        ("left_forearm", capsule(ELBOW_L, WRIST_L, 0.04), 0),
        ("right_forearm", capsule(ELBOW_R, WRIST_R, 0.04), 0),
    ]
    verts, faces, group, ranges, n = [], [], [], {}, 0
    for name, m, g in parts:
        verts.append(m.vertices)
        faces.append(m.faces + n)
        group.append(np.full(len(m.vertices), g))
        ranges[name] = [n, n + len(m.vertices)]
        n += len(m.vertices)

    # 70 MHR-like keypoints: 21 body/feet, 21 per hand, 7 extra. Names are a best guess
    # at the real MHR list; feet and head names match the substrings the tools search for.
    kp: list[tuple[str, np.ndarray, int]] = [
        ("nose", [0.0, 1.63, 0.11], 0),
        ("left_eye", [0.035, 1.66, 0.10], 0), ("right_eye", [-0.035, 1.66, 0.10], 0),
        ("left_ear", [0.11, 1.64, 0.0], 0), ("right_ear", [-0.11, 1.64, 0.0], 0),
        ("left_shoulder", SHOULDER_L, 0), ("right_shoulder", SHOULDER_R, 0),
        ("left_elbow", ELBOW_L, 0), ("right_elbow", ELBOW_R, 0),
        ("left_hip", HIP_L, 0), ("right_hip", HIP_R, 0),
        ("left_knee", KNEE_L, 0), ("right_knee", KNEE_R, 0),
        ("left_ankle", ANKLE_L, 1), ("right_ankle", ANKLE_R, 2),
        ("left_big_toe", [-0.025, 0.02, 0.25], 1), ("left_small_toe", [0.03, 0.02, 0.23], 1),
        ("left_heel", [0.0, 0.02, 0.01], 1),
        ("right_big_toe", [0.025, 0.02, -0.01], 2), ("right_small_toe", [-0.03, 0.02, -0.03], 2),
        ("right_heel", [0.0, 0.02, -0.25], 2),
    ]
    offs = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 3), [0.0], np.linspace(0.0, 0.06, 7)), -1).reshape(-1, 3)
    for side, wrist in (("right", WRIST_R), ("left", WRIST_L)):
        kp += [(f"{side}_hand_{i}", wrist + o, 0) for i, o in enumerate(offs)]
    kp += [
        ("left_olecranon", ELBOW_L + [0, 0, -0.03], 0), ("right_olecranon", ELBOW_R + [0, 0, -0.03], 0),
        ("left_cubital_fossa", ELBOW_L + [0, 0, 0.03], 0), ("right_cubital_fossa", ELBOW_R + [0, 0, 0.03], 0),
        ("left_acromion", SHOULDER_L + [0, 0.03, 0], 0), ("right_acromion", SHOULDER_R + [0, 0.03, 0], 0),
        ("neck", [0.0, 1.50, 0.0], 0),
    ]
    assert len(kp) == 70
    return Body(
        verts=np.concatenate(verts),
        faces=np.concatenate(faces),
        group=np.concatenate(group),
        parts=ranges,
        kp=np.array([p for _, p, _ in kp], float),
        kp_group=np.array([g for _, _, g in kp]),
        kp_names=[nm for nm, _, _ in kp],
    )


# --- motion -------------------------------------------------------------------


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sway_signal(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sum of slow sines (0.08–0.3 Hz), peak-normalized to 1."""
    freqs, amps, phases = rng.uniform(0.08, 0.3, 3), rng.uniform(0.5, 1.0, 3), rng.uniform(0, 2 * np.pi, 3)

    def s(x):
        return sum(a * np.sin(2 * np.pi * f * x + ph) for f, a, ph in zip(freqs, amps, phases))

    dense = np.linspace(0, t.max(), 4000)  # normalize on a dense grid, not just the samples
    return s(t) / np.max(np.abs(s(dense)))


def lift_height(t: np.ndarray) -> np.ndarray:
    up = smoothstep((t - LIFT_START_S) / LIFT_RAMP_S)
    down = smoothstep((LIFT_START_S + LIFT_DUR_S - t) / LIFT_RAMP_S)
    return LIFT_HEIGHT_M * np.minimum(up, down)


def lean_profile(t: np.ndarray) -> np.ndarray:
    return smoothstep((t - 10.5) / 1.5) * smoothstep((14.0 - t) / 1.0)


def rot(theta_x: float, theta_z: float) -> np.ndarray:
    cx, sx, cz, sz = np.cos(theta_x), np.sin(theta_x), np.cos(theta_z), np.sin(theta_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx


def pose(points: np.ndarray, groups: np.ndarray, R: np.ndarray, lift: float) -> np.ndarray:
    out = points.copy()
    body = groups == 0
    out[body] = (points[body] - PIVOT) @ R.T + PIVOT
    out[groups == 2] += [0.0, lift, 0.0]
    return out


# --- rendering (vis + phone input images) ---------------------------------------


def project(P: np.ndarray, f: float) -> np.ndarray:
    return f * P[:, :2] / P[:, 2:3] + [W / 2, H / 2]


def render(V: np.ndarray, faces: np.ndarray, f: float, kp2d: np.ndarray | None = None) -> Image.Image:
    """Flat-shaded painter's-algorithm render of camera-space verts."""
    img = Image.new("RGB", (W, H), (58, 63, 75))
    d = ImageDraw.Draw(img)
    uv = project(V, f)
    tri = V[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    to_light = np.array([-0.3, -0.5, -0.8]) / np.linalg.norm([-0.3, -0.5, -0.8])  # up-left, toward the camera
    light = np.clip(n @ to_light, 0, 1)
    for i in np.argsort(-tri[:, :, 2].mean(1)):
        if n[i, 2] > 0:  # back-facing
            continue
        c = int(60 + 190 * light[i])
        d.polygon([tuple(p) for p in uv[faces[i]]], fill=(c, c, int(c * 0.95)))
    if kp2d is not None:
        for u, v in kp2d[:21]:
            d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 64, 64))
    d.rectangle([0, 0, W, 44], fill=(20, 20, 20))
    d.text((12, 12), LABEL, fill=(255, 200, 0), font_size=26)
    return img


def side_view(V: np.ndarray, faces: np.ndarray, f: float) -> Image.Image:
    """Same mesh rotated 90° about the vertical through its center, like fal's side panel."""
    c = V.mean(0)
    up = R_WC @ np.array([0.0, 1.0, 0.0])
    Rs = trimesh.transformations.rotation_matrix(np.pi / 2, up)[:3, :3]
    return render((V - c) @ Rs.T + c, faces, f)


def png_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, quality=90)
    return buf.getvalue()


# --- main -------------------------------------------------------------------------


def generate(
    out: Path,
    *,
    seed: int = 0,
    duration_s: float = 20.0,
    fps: float = 3.0,
    sway_ml_m: float = 0.01,
    sway_ap_m: float = 0.015,
    lean_m: float = 0.035,
    focal_err: float = 0.05,
    vertex_noise_m: float = 0.003,
    body_relative: bool = False,
    render_images: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    if out.exists():
        shutil.rmtree(out)
    (out / "input").mkdir(parents=True)

    body = build_body()
    t = np.arange(0, duration_s, 1 / fps)

    # Pendulum length from the body-group COM so the requested sway is the COM displacement.
    rest = trimesh.Trimesh(body.verts, body.faces, process=False)
    L = rest.center_mass[1] - PIVOT[1]
    d_ml = sway_ml_m * sway_signal(t, rng) + lean_m * lean_profile(t)
    d_ap = sway_ap_m * sway_signal(t, rng)
    lift = lift_height(t)

    gt_com_world, gt_lift = [], []
    for k, tk in enumerate(t):
        R = rot(theta_x=d_ap[k] / L, theta_z=-d_ml[k] / L)
        Vw = pose(body.verts, body.group, R, lift[k])
        Kw = pose(body.kp, body.kp_group, R, lift[k])
        gt_com_world.append(trimesh.Trimesh(Vw, body.faces, process=False).center_mass)
        gt_lift.append(lift[k])

        # SAM-like estimate: focal off by u, depth off by the same factor, plus jitter.
        u = rng.uniform(-focal_err, focal_err)
        f_k = F_TRUE * (1 + u)
        pelvis_cam = to_cam(pose(np.array([[0.0, 0.95, 0.0]]), np.array([0]), R, 0.0))[0]
        dz = np.array([0.0, 0.0, pelvis_cam[2] * u])
        Vc = to_cam(Vw) + dz + rng.normal(scale=vertex_noise_m, size=Vw.shape)
        Kc = to_cam(Kw) + dz + rng.normal(scale=vertex_noise_m, size=Kw.shape)
        cam_t = pelvis_cam + dz
        kp2d = project(Kc, f_k) + rng.normal(scale=0.5, size=(70, 2))

        stem = f"frame_{k + 1:06d}"
        person = {
            "person_id": 0,
            "bbox": [*kp2d.min(0).tolist(), *kp2d.max(0).tolist()],
            "focal_length": f_k,
            "pred_cam_t": cam_t.tolist(),
            "keypoints_2d": kp2d.tolist(),
            "keypoints_3d": (Kc - cam_t if body_relative else Kc).tolist(),
        }
        meta = {"SYNTHETIC": True, "num_people": 1, "keypoint_names": body.kp_names, "people": [person]}
        Vstore = Vc - cam_t if body_relative else Vc
        ply = trimesh.Trimesh(Vstore, body.faces, process=False).export(file_type="ply")

        vis = None
        if render_images:
            front = render(Vc, body.faces, f_k)
            (out / "input" / f"{stem}.jpg").write_bytes(png_bytes(front, "JPEG"))
            panel = Image.new("RGB", (2 * W, H))
            panel.paste(render(Vc, body.faces, f_k, kp2d), (0, 0))
            panel.paste(side_view(Vc, body.faces, f_k), (W, 0))
            vis = png_bytes(panel.resize((W, H // 2)))
        response = {
            "meshes": [{"url": f"synthetic://{stem}.ply"}],
            "visualization": {"url": f"synthetic://{stem}_vis.png", "content_type": "image/png"},
            "metadata": meta,
        }
        write_frame(
            out, stem, t_ms=float(tk * 1000), image_size=[W, H], response=response,
            latency_s=0.0, ply=ply, visualization=vis, extra={"SYNTHETIC": True},
        )

    events = [
        {"t": LIFT_START_S * 1000, "kind": "foot_lift", "side": "right"},
        {"t": (LIFT_START_S + LIFT_DUR_S) * 1000, "kind": "foot_down", "side": "right"},
    ]
    write_summary(out, {"SYNTHETIC": True, "events": events})  # true events, for viewer ticks
    # self-describing axis conventions, so the pipeline never applies real fal's to synthetic data
    conv = {"add_cam_t": body_relative, "flip": "identity"}
    (out / "conventions.json").write_text(json.dumps(
        {"SYNTHETIC": True, "mesh": conv, "keypoints_3d": conv, "camera_y_down": True}, indent=1))

    # Ground truth. Floor frame == world frame here: the camera has no roll/yaw, so the
    # minimal rotation taking the camera-frame floor normal to +y is exactly camera->world.
    n_cam = R_WC @ np.array([0.0, 1.0, 0.0])
    p_cam = to_cam(np.zeros(3))
    com_world = np.array(gt_com_world)
    left, right = body.parts["left_foot"], body.parts["right_foot"]
    gt = {
        "SYNTHETIC": True,
        "description": "Ground truth for a synthetic run; NOT real data.",
        "seed": seed,
        "fps": fps,
        "t_ms": (t * 1000).tolist(),
        "image_size": [W, H],
        "focal_true": F_TRUE,
        "mesh_frame": "body_relative (add pred_cam_t)" if body_relative else "camera (OpenCV: x right, y down, z forward)",
        "camera": {"position_world": CAM_POS.tolist(), "R_world_to_cam": R_WC.tolist()},
        "floor_plane_camera": {
            "normal": n_cam.tolist(),  # unit, points toward the head
            "point": p_cam.tolist(),  # world origin (stance-foot center on the floor)
            "d": float(-n_cam @ p_cam),  # n·x + d = 0
        },
        "floor_frame": {
            "definition": "y up, floor y=0, origin = mean stance-foot center, x = patient's left (image right), z = toward camera",
            "R_cam_to_floor": R_WC.T.tolist(),
            "origin_camera": p_cam.tolist(),
        },
        "patient_height_m": float(body.verts[:, 1].max()),
        "com_floor": com_world.tolist(),  # noise-free mesh, volumetric (divergence theorem)
        "com_note": "COM of the concatenated part meshes; overlapping parts are double-counted, same as center_of_mass on this mesh",
        "sway_m": {"ml": d_ml.tolist(), "ap": d_ap.tolist(), "ml_peak": sway_ml_m, "ap_peak": sway_ap_m, "lean_m": lean_m},
        "events": events,
        "lift_height_m": gt_lift,
        "noise": {"focal_err_max": focal_err, "vertex_noise_m": vertex_noise_m, "keypoint_2d_noise_px": 0.5},
        "vertex_count": int(len(body.verts)),
        "face_count": int(len(body.faces)),
        "vertex_ranges": body.parts,
        "foot_vertices": {"left": list(range(*left)), "right": list(range(*right))},
    }
    (out / "ground_truth.json").write_text(json.dumps(gt))
    return gt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPLAY_ROOT / "data/fal_out/synthetic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--body-relative", action="store_true", help="store mesh/keypoints_3d without pred_cam_t")
    p.add_argument("--no-lean", action="store_true", help="disable the pre-error lean toward the BOS edge")
    a = p.parse_args()
    gt = generate(a.out, seed=a.seed, body_relative=a.body_relative, lean_m=0.0 if a.no_lean else 0.035)
    com = np.array(gt["com_floor"])
    print(f"{LABEL} run -> {a.out}")
    print(f"  {len(gt['t_ms'])} frames, V={gt['vertex_count']}, height {gt['patient_height_m']:.3f} m")
    print(f"  COM height {com[:, 1].mean():.3f} m; COM range x {np.ptp(com[:, 0]) * 100:.1f} cm, z {np.ptp(com[:, 2]) * 100:.1f} cm")
    print(f"  events {gt['events']}")
