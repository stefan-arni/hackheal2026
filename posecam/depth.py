"""Depth frames from the iPhone (LiDAR or TrueDepth) -> metric 3D torso position.

A JSON frame can carry a depth map next to the JPEG:

    {"type": "frame", "image": "<base64 JPEG>", "timestamp_ms": ..., "rotate": 90,
     "depth": "<base64>", "depth_format": "uint16_mm" | "float16_m",
     "depth_size": [w, h],                 # depth map size before rotation
     "intrinsics": [fx, fy, cx, cy],       # in depth-map pixels, before rotation
     "camera": "lidar" | "truedepth"}

The depth map must cover the same field of view as the image (it does when both
come from one AVCaptureDataOutputSynchronizer with a 4:3 video format). It's
rotated with the same `rotate` as the image, and the intrinsics with it, so
landmark coordinates (normalized 0-1 on the upright image) index it directly.
"""

from __future__ import annotations

import base64

import numpy as np

ROT_K = {0: 0, 90: -1, 180: 2, 270: 1}   # np.rot90 k for a clockwise rotation
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
MIN_Z, MAX_Z = 0.2, 8.0                     # plausible depth range in metres


def rotate_intrinsics(K, w: int, h: int, rot: int):
    """Intrinsics (fx, fy, cx, cy) of a w x h image after rotating it `rot`
    degrees clockwise (same as cv2.rotate / ws_server's `rotate`)."""
    fx, fy, cx, cy = K
    rot %= 360
    if rot == 90:
        return fy, fx, h - 1 - cy, cx
    if rot == 180:
        return fx, fy, w - 1 - cx, h - 1 - cy
    if rot == 270:
        return fy, fx, cy, w - 1 - cx
    return fx, fy, cx, cy


def decode_depth(meta: dict, rot: int = 0):
    """Pop the depth fields from a frame's metadata. Returns (depth in metres as
    float32 HxW, intrinsics) after rotation, or (None, None) if there's none."""
    raw = meta.pop("depth", None)
    if raw is None:
        return None, None
    w, h = (int(v) for v in meta.get("depth_size", (0, 0)))
    fmt = meta.get("depth_format", "uint16_mm")
    buf = base64.b64decode(raw)
    if fmt == "uint16_mm":
        d = np.frombuffer(buf, "<u2").astype(np.float32) / 1000.0
    elif fmt == "float16_m":
        d = np.frombuffer(buf, "<f2").astype(np.float32)
    else:
        raise ValueError(f"unknown depth_format {fmt!r}")
    if w * h != d.size:
        raise ValueError(f"depth_size {w}x{h} doesn't match {d.size} values")
    d = d.reshape(h, w)
    K = meta.get("intrinsics")
    if K is None or len(K) != 4:
        raise ValueError("depth frames need intrinsics [fx, fy, cx, cy]")
    rot %= 360
    d = np.ascontiguousarray(np.rot90(d, ROT_K.get(rot, 0)))
    return d, rotate_intrinsics([float(k) for k in K], w, h, rot)


def torso_point(landmarks, depth: np.ndarray, K, hip_weight: float = 0.65,
                min_visibility: float = 0.5, grid: int = 9) -> dict | None:
    """Metric 3D position of the torso (camera coordinates: x right, y down,
    z away from the camera, metres). Depth is the median over a grid of points
    inside the shoulder-hip quadrilateral (inner 60%), so a few bad depth pixels
    or the background between the arms don't matter. x, y are at the COM point
    (`hip_weight` of the way from shoulder-mid to hip-mid)."""
    vis = [landmarks[i].get("visibility") or 0.0 for i in (L_SH, R_SH, L_HIP, R_HIP)]
    if min(vis) < min_visibility:
        return None
    dh, dw = depth.shape
    P = {i: np.array([landmarks[i]["x"] * dw, landmarks[i]["y"] * dh]) for i in (L_SH, R_SH, L_HIP, R_HIP)}

    zs = []
    for a in np.linspace(0.2, 0.8, grid):           # across (left -> right)
        top = P[L_SH] + a * (P[R_SH] - P[L_SH])
        bot = P[L_HIP] + a * (P[R_HIP] - P[L_HIP])
        for b in np.linspace(0.2, 0.8, grid):       # down (shoulders -> hips)
            u, v = top + b * (bot - top)
            iu, iv = int(round(u)), int(round(v))
            if 0 <= iu < dw and 0 <= iv < dh:
                z = float(depth[iv, iu])
                if MIN_Z < z < MAX_Z and np.isfinite(z):
                    zs.append(z)
    if len(zs) < max(5, grid * grid // 4):
        return None
    z = float(np.median(zs))
    fx, fy, cx, cy = K
    com = hip_weight * (P[L_HIP] + P[R_HIP]) / 2 + (1 - hip_weight) * (P[L_SH] + P[R_SH]) / 2
    x = (com[0] - cx) * z / fx
    y = (com[1] - cy) * z / fy
    q1, q3 = np.percentile(zs, [25, 75])
    return {"torso_m": [round(x, 4), round(y, 4), round(z, 4)],
            "valid_points": len(zs), "depth_spread_m": round(float(q3 - q1), 4)}


L_ANK, R_ANK, L_HEEL, R_HEEL = 27, 28, 29, 30


def ankle_depths(landmarks, depth: np.ndarray, radius: int = 2,
                 min_visibility: float = 0.3) -> dict | None:
    """Distance (metres) from the camera to each ankle: the median depth around
    the ankle and around the heel point, whichever is nearer (a point that
    slipped off the foot reads the floor or wall behind it, which is further).
    None if either foot can't be read."""
    dh, dw = depth.shape
    out = {}
    for side, idx in (("left", (L_ANK, L_HEEL)), ("right", (R_ANK, R_HEEL))):
        per_point = []
        for i in idx:
            lm = landmarks[i]
            if (lm.get("visibility") or 0.0) < min_visibility:
                continue
            u, v = int(round(lm["x"] * dw)), int(round(lm["y"] * dh))
            patch = depth[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1]
            zs = [float(z) for z in patch.ravel() if MIN_Z < z < MAX_Z]
            if len(zs) >= 3:
                per_point.append(float(np.median(zs)))
        if not per_point:
            return None
        out[side] = round(min(per_point), 4)
    return out
