"""Pack/unpack replay bundles (REPLAY_SPEC.md "Bundle format").

    <dir>/meta.json   trialId, t [F] (ms), counts, events, com, bos, margin, heatmap, noise_floor, quality, frames
    <dir>/verts_raw.bin  same layout, before temporal smoothing (lines up with the video frames)
    <dir>/analytics.json  per-trial analytics (service/analytics.py), when the bundle is aligned
    <dir>/faces.bin   Uint32 [faces_count * 3]
    <dir>/verts.bin   Float32 [F * V * 3], little-endian, frame-major; Float16 when the float32
                      file would exceed FLOAT16_ABOVE_BYTES (meta.verts_dtype says which)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FLOAT16_ABOVE_BYTES = 8 * 1024 * 1024  # float16 ≈ 1 mm precision at 1–2 m, far below SAM's noise


def _list(x):
    """Array -> nested lists with NaN/inf as None (JSON has no NaN)."""
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    return np.where(np.isfinite(a), a, None).tolist()


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
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
    analytics: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bundle from pipeline.process() (or pipeline.raw_display()) output."""
    out.mkdir(parents=True, exist_ok=True)
    verts = np.ascontiguousarray(result["verts"], dtype="<f4")
    faces = np.ascontiguousarray(result["faces"], dtype="<u4")
    F, V, _ = verts.shape
    dtype = "float32"
    if verts.nbytes > FLOAT16_ABOVE_BYTES:
        verts, dtype = verts.astype("<f2"), "float16"
    meta = {
        "trialId": trial_id,
        "t": (np.asarray(result["t_s"]) * 1000).round(1).tolist(),
        "frames_count": F,
        "vertex_count": V,
        "verts_dtype": dtype,
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
    if result.get("camera") is not None:
        meta["camera"] = _clean(result["camera"])  # floor -> SAM camera per frame (video overlay)
    if analytics is not None:
        meta["touchdowns"] = _clean(analytics.get("touchdowns", []))
        meta["series"] = _clean(analytics.get("series", {}))
        (out / "analytics.json").write_text(json.dumps(_clean(analytics), indent=1, allow_nan=False))
        meta["analytics"] = {"url": "analytics.json", "quality": analytics["quality"]["label"],
                             "min_margin_cm": analytics.get("margin", {}).get("min_cm"),
                             "time_outside_bos_s": analytics.get("margin", {}).get("time_outside_bos_s"),
                             "ml_sway_rms_cm": analytics["ml_sway"]["rms_cm"]}
    (out / "verts.bin").write_bytes(verts.tobytes())
    if result.get("verts_raw") is not None:  # unsmoothed, for overlays on the video frame
        raw = np.ascontiguousarray(result["verts_raw"], dtype="<f4")
        (out / "verts_raw.bin").write_bytes((raw.astype("<f2") if dtype == "float16" else raw).tobytes())
        meta["verts_raw"] = "verts_raw.bin"
    (out / "faces.bin").write_bytes(faces.tobytes())
    if extra_meta:
        meta.update(_clean(extra_meta))
    (out / "meta.json").write_text(json.dumps(meta, allow_nan=False))
    return meta


def read_bundle(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """-> (meta, faces (Fc,3) uint32, verts (F,V,3) float32 — float16 files are widened)."""
    meta = json.loads((path / "meta.json").read_text())
    faces = np.frombuffer((path / "faces.bin").read_bytes(), dtype="<u4").reshape(-1, 3)
    dt = "<f2" if meta.get("verts_dtype") == "float16" else "<f4"
    verts = np.frombuffer((path / "verts.bin").read_bytes(), dtype=dt).astype(np.float32).reshape(
        meta["frames_count"], meta["vertex_count"], 3
    )
    assert len(faces) == meta["faces_count"]
    return meta, faces, verts
