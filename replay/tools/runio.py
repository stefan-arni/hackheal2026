"""Load a send_frames.py output directory for the hour-one diagnostics."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.runs import FLIPS, to_camera  # noqa: E402,F401


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
        if p.name in ("summary.json", "conventions.json", "noise_floor.json", "ground_truth.json"):
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
    norm = lambda x: x.lower().replace("-", "_")  # fal's MHR names are hyphenated
    pats = [norm(s) for s in patterns]
    return [i for i, n in enumerate(names) if any(s in norm(n) for s in pats)]


def load_conventions(run_dir: Path) -> dict | None:
    p = run_dir / "conventions.json"
    return json.loads(p.read_text()) if p.exists() else None
