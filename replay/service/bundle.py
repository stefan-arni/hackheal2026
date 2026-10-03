"""Pack/unpack replay bundles (REPLAY_SPEC.md "Bundle format").

    <dir>/meta.json   trialId, t [F] (ms), counts, events, com, bos, margin, heatmap, noise_floor, quality, frames
    <dir>/faces.bin   Uint32 [faces_count * 3]
    <dir>/verts.bin   Float32 [F * V * 3], little-endian, frame-major
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _list(x):
    """Array -> nested lists with NaN/inf as None (JSON has no NaN)."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    return np.where(np.isfinite(a), a, None).tolist()


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_bundle(
    out: Path,
    trial_id: str,
    result: dict[str, Any],
    *,
    events: list[dict] | None = None,
    frames: list[dict] | None = None,
) -> dict[str, Any]:
    """Write a bundle from pipeline.process() (or pipeline.raw_display()) output."""
    out.mkdir(parents=True, exist_ok=True)
    verts = np.ascontiguousarray(result["verts"], dtype="<f4")
    faces = np.ascontiguousarray(result["faces"], dtype="<u4")
    F, V, _ = verts.shape
    meta = {
        "trialId": trial_id,
        "t": (np.asarray(result["t_s"]) * 1000).round(1).tolist(),
        "frames_count": F,
        "vertex_count": V,
        "faces_count": len(faces),
        "events": events or [],
        "com": _list(result.get("com")),
        "bos": [_list(h) for h in result["bos"]] if result.get("bos") is not None else None,
        "margin": _list(result.get("margin")),
        "margin_ml": _list(result.get("margin_ml")),
        "margin_ap": _list(result.get("margin_ap")),
        "heatmap": _list(result.get("heatmap")),
        "noise_floor": _clean(result.get("noise_floor")),
        "quality": _clean(result.get("quality", {})),
        "frames": frames or [],
    }
    (out / "verts.bin").write_bytes(verts.tobytes())
    (out / "faces.bin").write_bytes(faces.tobytes())
    (out / "meta.json").write_text(json.dumps(meta, allow_nan=False))
    return meta


def read_bundle(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """-> (meta, faces (Fc,3) uint32, verts (F,V,3) float32)."""
    meta = json.loads((path / "meta.json").read_text())
    faces = np.frombuffer((path / "faces.bin").read_bytes(), dtype="<u4").reshape(-1, 3)
    verts = np.frombuffer((path / "verts.bin").read_bytes(), dtype="<f4").reshape(
        meta["frames_count"], meta["vertex_count"], 3
    )
    assert len(faces) == meta["faces_count"]
    return meta, faces, verts
