"""Run-folder format shared by send_frames.py, the service and the synthetic generator.

Per frame: <stem>.ply, <stem>.json, <stem>_vis.<ext>. Plus summary.json, whose
`frames` list (sorted by time) is what the viewer uses to find files.
"""

from __future__ import annotations

import io
import json
import mimetypes
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# Axis sign variants tried by the convention check. Applied before adding cam_t.
FLIPS: dict[str, tuple[float, float, float]] = {
    "identity": (1, 1, 1),
    "flip_y": (1, -1, 1),
    "flip_yz": (1, -1, -1),
    "flip_x": (-1, 1, 1),
    "flip_xy": (-1, -1, 1),
}

# What conventions.json says when the check hasn't been run: fal documents camera space.
DEFAULT_CONVENTIONS = {
    "mesh": {"flip": "identity", "add_cam_t": False},
    "keypoints_3d": {"flip": "identity", "add_cam_t": False},
}


def to_camera(points: np.ndarray, cam_t: np.ndarray, flip: str, add_cam_t: bool) -> np.ndarray:
    out = points * np.asarray(FLIPS[flip])
    return out + cam_t if add_cam_t else out


RESERVED_JSON = {"summary.json", "conventions.json", "noise_floor.json", "ground_truth.json"}


def mesh_counts(ply: bytes) -> tuple[int, int] | None:
    """(vertices, faces), or None if the bytes aren't a readable mesh."""
    try:
        mesh = trimesh.load(io.BytesIO(ply), file_type="ply", process=False)
        return len(mesh.vertices), len(mesh.faces)
    except Exception:
        return None


def vis_extension(response: dict[str, Any], data: bytes | None = None) -> str:
    """Image extension: sniffed from the bytes first (fal serves visualizations as
    application/octet-stream), then the declared content type, then the URL."""
    if data:
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
    vis = response.get("visualization") or {}
    ext = mimetypes.guess_extension(vis.get("content_type") or "")
    if ext in (None, ".bin"):
        ext = Path(vis.get("url", "")).suffix
    return ext if ext and ext != ".bin" else ".png"


def write_frame(
    out: Path,
    stem: str,
    *,
    t_ms: float,
    image_size: list[int],
    response: dict[str, Any],
    latency_s: float,
    ply: bytes | None,
    visualization: bytes | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one frame's files from a fal response; returns the .json record."""
    meta = response.get("metadata") or {}
    num_people = int(meta.get("num_people", 0))
    meshes = response.get("meshes") or []
    vis = response.get("visualization") or {}
    record: dict[str, Any] = {
        "frame": f"{stem}.jpg",
        "t_ms": t_ms,
        "image_size": image_size,  # [W, H] of the image sent to fal (the crop)
        "latency_s": round(latency_s, 3),
        "num_people": num_people,
        "usable": num_people == 1 and ply is not None,
        "vertex_count": None,
        "face_count": None,
        "mesh_url": meshes[0]["url"] if meshes else None,
        "visualization_url": vis.get("url"),
        "vis_file": None,
        **(extra or {}),
        "metadata": meta,
    }
    if ply is not None:
        (out / f"{stem}.ply").write_bytes(ply)
        counts = mesh_counts(ply)
        if counts is None:
            record["usable"], record["mesh_error"] = False, "unreadable .ply"
        else:
            record["vertex_count"], record["face_count"] = counts
    if visualization is not None:
        record["vis_file"] = f"{stem}_vis{vis_extension(response, visualization)}"
        (out / record["vis_file"]).write_bytes(visualization)
    (out / f"{stem}.json").write_text(json.dumps(record, indent=1))
    return record


def load_records(out: Path) -> list[dict[str, Any]]:
    recs = [json.loads(p.read_text()) for p in out.glob("*.json") if p.name not in RESERVED_JSON]
    return sorted(recs, key=lambda r: r["t_ms"])


def write_summary(out: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize every frame in the folder (including earlier batches) into summary.json."""
    done = load_records(out)
    lat = sorted(r["latency_s"] for r in done)
    summary = {
        "frames_total": len(done),
        "usable": sum(r["usable"] for r in done),
        "num_people_counts": {str(k): sum(r["num_people"] == k for r in done) for k in {r["num_people"] for r in done}},
        "vertex_counts": sorted({r["vertex_count"] for r in done if r["vertex_count"] is not None}),
        "face_counts": sorted({r["face_count"] for r in done if r["face_count"] is not None}),
        "latency_s": {
            "p50": statistics.median(lat) if lat else None,
            "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
            "max": lat[-1] if lat else None,
        },
        **(extra or {}),
        "frames": [
            {"stem": Path(r["frame"]).stem, "t_ms": r["t_ms"], "usable": r["usable"], "vis_file": r.get("vis_file")}
            for r in done
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
