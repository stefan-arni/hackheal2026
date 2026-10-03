"""Tiny software renderer for report images (Pillow only, no GPU or browser).

Flat-shaded, depth-sorted triangles. Two cameras:
  - look_at(...): a virtual camera in the floor frame (hero shot)
  - the per-frame SAM camera from the bundle (mesh drawn over the actual video frame)
Camera convention: OpenCV (x right, y down, z forward), pixel = f·X/Z + (cx, cy).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with p_cam = R @ p_world + t, for a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def project(P_cam: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.c_[f * P_cam[:, 0] / P_cam[:, 2] + cx, f * P_cam[:, 1] / P_cam[:, 2] + cy]


def draw_mesh(img: Image.Image, V_cam: np.ndarray, faces: np.ndarray, f: float, cx: float, cy: float,
              color=(226, 232, 240), alpha: float = 1.0, light=(-0.35, -0.6, -0.72), rim: float = 0.0,
              rim_color=(125, 211, 252)) -> Image.Image:
    """Composite a flat-shaded mesh (camera-frame vertices) onto img; returns the new image."""
    uv = project(V_cam, f, cx, cy)
    tri = V_cam[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    centre = tri.mean(axis=1)
    view = -centre / (np.linalg.norm(centre, axis=1, keepdims=True) + 1e-12)
    facing = np.sum(n * view, axis=1)
    keep = (facing > 0) & (tri[:, :, 2].min(axis=1) > 0.05)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    lam = np.clip(n @ -L, 0, 1)
    shade = 0.42 + 0.58 * lam
    fres = (1 - np.clip(facing, 0, 1)) ** 2.5 * rim  # rim light on silhouettes
    base = np.asarray(color, float)
    rimc = np.asarray(rim_color, float)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    for i in np.flatnonzero(keep)[np.argsort(-centre[keep, 2])]:
        c = np.clip(base * shade[i] + rimc * fres[i], 0, 255).astype(int)
        d.polygon([tuple(p) for p in uv[faces[i]]], fill=(int(c[0]), int(c[1]), int(c[2]), a))
    out = img.convert("RGBA")
    out.alpha_composite(layer)
    return out


def floor_polyline(img: Image.Image, pts_xz, R, t, f, cx, cy, color, width=3, closed=True, fill=None):
    """Draw a polygon/polyline lying on the floor (y = 0) of the floor frame."""
    P = np.c_[np.asarray(pts_xz)[:, 0], np.zeros(len(pts_xz)), np.asarray(pts_xz)[:, 1]]
    uv = project(P @ R.T + t, f, cx, cy)
    d = ImageDraw.Draw(img, "RGBA")
    pts = [tuple(p) for p in uv]
    if fill is not None and len(pts) >= 3:
        d.polygon(pts, fill=fill)
    if closed and len(pts) >= 2:
        pts = pts + [pts[0]]
    d.line(pts, fill=color, width=width, joint="curve")
