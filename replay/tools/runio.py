"""Load a send_frames.py output directory for the hour-one diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class Frame:
    stem: str
    t_ms: float
    W: int
    H: int
    focal: float
    cam_t: np.ndarray  # (3,)
    kp2d: np.ndarray  # (70, 2)
    kp3d: np.ndarray | None  # (70, 3)
    ply_path: Path

    def verts(self) -> np.ndarray:
        mesh = trimesh.load(self.ply_path, file_type="ply", process=False)
        return np.asarray(mesh.vertices, dtype=np.float64)


def load_run(run_dir: Path) -> tuple[list[Frame], list[str]]:
    """Usable single-person frames sorted by time, plus the keypoint names."""
    frames: list[Frame] = []
    names: list[str] = []
    for p in sorted(run_dir.glob("*.json")):
        if p.name in ("summary.json", "conventions.json", "noise_floor.json"):
            continue
        rec = json.loads(p.read_text())
        if not rec.get("usable"):
            continue
        meta = rec["metadata"]
        names = names or meta.get("keypoint_names") or []
        person = meta["people"][0]
        kp3d = person.get("keypoints_3d")
        W, H = rec["image_size"]
        frames.append(
            Frame(
                stem=p.stem,
                t_ms=rec["t_ms"],
                W=W,
                H=H,
                focal=float(person["focal_length"]),
                cam_t=np.asarray(person["pred_cam_t"], dtype=np.float64),
                kp2d=np.asarray(person["keypoints_2d"], dtype=np.float64)[:, :2],
                kp3d=None if kp3d is None else np.asarray(kp3d, dtype=np.float64)[:, :3],
                ply_path=run_dir / f"{p.stem}.ply",
            )
        )
    frames.sort(key=lambda f: f.t_ms)
    return frames, names


def find_keypoints(names: list[str], *patterns: str) -> list[int]:
    """Indices of keypoints whose name contains any of the patterns (case-insensitive)."""
    pats = [s.lower() for s in patterns]
    return [i for i, n in enumerate(names) if any(s in n.lower() for s in pats)]


# Axis sign variants tried by the convention check. Applied before adding cam_t.
FLIPS: dict[str, tuple[float, float, float]] = {
    "identity": (1, 1, 1),
    "flip_y": (1, -1, 1),
    "flip_yz": (1, -1, -1),
    "flip_x": (-1, 1, 1),
    "flip_xy": (-1, -1, 1),
}


def to_camera(points: np.ndarray, cam_t: np.ndarray, flip: str, add_cam_t: bool) -> np.ndarray:
    out = points * np.asarray(FLIPS[flip])
    return out + cam_t if add_cam_t else out


def load_conventions(run_dir: Path) -> dict | None:
    p = run_dir / "conventions.json"
    return json.loads(p.read_text()) if p.exists() else None
