"""All replay math. Pure functions on numpy arrays, unit-tested in tests/test_geometry.py.

Conventions after alignment (step 4 on): y up, floor is the x–z plane, origin at the
stance-foot center, meters. Before alignment: SAM camera frame (see conventions.json).
Shapes: F frames, V vertices, K keypoints, N generic points.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

# --- 1. Convention check -----------------------------------------------------


def project_pinhole(points: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """Project camera-frame points (N,3) to pixels (N,2): u = f·X/Z + W/2, v = f·Y/Z + H/2.

    Principal point at the image center; points with Z <= 0 map to NaN.
    """
    P = np.asarray(points, dtype=float)
    Z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = focal * P[:, :2] / Z[:, None] + np.array([W / 2, H / 2])
    uv[Z <= 0] = np.nan
    return uv


def reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """Per-point pixel distance (N,) between projected points_3d (N,3) and points_2d (N,2).

    NaN where a point is behind the camera or the 2D point is missing.
    """
    return np.linalg.norm(project_pinhole(points_3d, focal, W, H) - np.asarray(points_2d, float), axis=1)


# --- 2. Focal normalization --------------------------------------------------


def normalize_focal(cam_t: np.ndarray, focal: np.ndarray) -> np.ndarray:
    """cam_t (F,3), focal (F,) -> cam_t' (F,3) with tz' = tz · f̄ / f_i, f̄ = median focal.

    SAM's focal and depth are coupled (bigger f ⇔ farther away); a fixed camera has one focal,
    so depth is rescaled to the median. tx, ty unchanged.
    """
    out = np.array(cam_t, dtype=float, copy=True)
    focal = np.asarray(focal, dtype=float)
    out[:, 2] *= np.median(focal) / focal
    return out


# --- 3. Outlier rejection ----------------------------------------------------


def nearest_timestamp_index(t_query: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """For each t_query (F,), index into sorted t_ref (M,) of the nearest timestamp.

    Ties go to the earlier reference sample.
    """
    t_query, t_ref = np.asarray(t_query, float), np.asarray(t_ref, float)
    i = np.clip(np.searchsorted(t_ref, t_query), 1, len(t_ref) - 1)
    left_closer = np.abs(t_query - t_ref[i - 1]) <= np.abs(t_ref[i] - t_query)
    return np.where(left_closer, i - 1, i)


def inlier_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (F,) mask: True where the per-frame reprojection error <= threshold.

    NaN (no comparison possible) counts as an outlier.
    """
    return np.nan_to_num(np.asarray(errors, float), nan=np.inf) <= threshold


# --- 4. Floor fit + gravity alignment ----------------------------------------


def fit_plane(points: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through points (N,3) via SVD of the centered points.

    Returns (centroid (3,), unit normal (3,)); the normal is the right-singular vector with the
    smallest singular value, flipped so normal · up_hint > 0 (points toward the head).
    """
    P = np.asarray(points, float)
    centroid = P.mean(axis=0)
    n = np.linalg.svd(P - centroid, full_matrices=False)[2][-1]
    n /= np.linalg.norm(n)
    return centroid, (n if n @ np.asarray(up_hint, float) > 0 else -n)


def rotation_to_y(n: np.ndarray) -> np.ndarray:
    """Rotation matrix R (3,3) with R @ n = ŷ (Rodrigues).

    k = (n×ŷ)/|n×ŷ|, θ = arccos(n·ŷ), R = I + sinθ·K + (1−cosθ)·K². n ≈ ŷ gives I; n ≈ −ŷ gives
    a 180° turn about x. Ill-conditioned near 180°: feed it a roughly upward normal.
    """
    n = np.asarray(n, float) / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    k = np.cross(n, y)
    s, c = np.linalg.norm(k), float(n @ y)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def true_to_sam_rotation(crop_center_px: np.ndarray, frame_size: tuple[int, int], focal: float) -> np.ndarray:
    """Rotation (3,3) taking directions from the true camera frame (principal point at the full-frame
    center) to SAM's frame, which assumes the principal point at the crop center: the true ray
    through the crop center becomes SAM's optical axis. Use it for phone gravity / vanishing points."""
    W, H = frame_size
    r = np.array([(crop_center_px[0] - W / 2) / focal, (crop_center_px[1] - H / 2) / focal, 1.0])
    r /= np.linalg.norm(r)
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(r, z)
    s, c = np.linalg.norm(k), float(r @ z)
    if s < 1e-12:
        return np.eye(3)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def align_to_floor(points: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Map camera-frame points (..., 3) to the floor frame: R @ (p − origin).

    Works on any leading shape (a point, a mesh, a whole trial).
    """
    return (np.asarray(points, float) - origin) @ np.asarray(R).T


# --- 5. Metric scale ---------------------------------------------------------


def metric_scale(patient_height_m: float, mesh_heights: np.ndarray) -> float:
    """s = H_patient / median(mesh_heights), mesh_heights (F,) = floor-to-head-top per frame.

    The median ignores frames where the person crouches or the head is mis-fit.
    """
    return float(patient_height_m / np.median(np.asarray(mesh_heights, float)))


# --- 6. Temporal smoothing ---------------------------------------------------


def gaussian_smooth(t: np.ndarray, X: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """Gaussian-kernel smoothing over non-uniform timestamps (bursts + uniform frames).

    t (F,) seconds, X (F, ...) -> X̂_i = Σ_j w_ij X_j / Σ_j w_ij with w_ij = exp(−(t_i−t_j)²/2σ²).
    Weights are normalized per row, so constants are preserved exactly.
    """
    t = np.asarray(t, float)
    w = np.exp(-((t[:, None] - t[None]) ** 2) / (2 * sigma**2))
    w /= w.sum(axis=1, keepdims=True)
    return np.tensordot(w, np.asarray(X, float), axes=1)


# --- 7. Center of mass -------------------------------------------------------


def center_of_mass(V: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Volumetric COM (3,) of a closed mesh with uniform density.

    Sum of signed tetrahedra from the origin per face (a, b, c): volume a·(b×c)/6, centroid
    (a+b+c)/4; COM = Σ vol·centroid / Σ vol. Matches trimesh's center_mass.
    """
    V = np.asarray(V, float)
    a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    return (vol[:, None] * (a + b + c) / 4.0).sum(axis=0) / vol.sum()


def is_watertight(faces: np.ndarray) -> bool:
    """True if every undirected edge is shared by exactly two faces.

    Required for the divergence-theorem COM to be meaningful.
    """
    f = np.asarray(faces)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


# --- 8. Base of support ------------------------------------------------------


def contact_points(V: np.ndarray, floor_tol: float = 0.015) -> np.ndarray:
    """Floor-frame vertices (V,3) with y <= floor_tol, projected to the floor -> (N,2) as [x, z].

    These are the parts of the feet touching (or within tolerance of) the floor.
    """
    V = np.asarray(V, float)
    return V[V[:, 1] <= floor_tol][:, [0, 2]]


def support_hull(points_xz: np.ndarray) -> np.ndarray:
    """Convex hull vertices (K,2) of floor points (N,2), counter-clockwise in (x, z).

    Fewer than 3 distinct points (or collinear ones) return the points as given.
    """
    P = np.asarray(points_xz, float)
    if len(np.unique(P, axis=0)) < 3:
        return P
    try:
        return P[ConvexHull(P).vertices]  # scipy returns 2D hulls counter-clockwise
    except Exception:  # degenerate (collinear)
        return P


def _edges(hull: np.ndarray):
    return hull, np.roll(hull, -1, axis=0)


def _inside_convex(p: np.ndarray, hull: np.ndarray) -> bool:
    a, b = _edges(hull)
    cross = (b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (p[0] - a[:, 0])
    return bool((cross >= -1e-12).all() or (cross <= 1e-12).all())


def _nearest_on_boundary(p: np.ndarray, hull: np.ndarray) -> np.ndarray:
    a, b = _edges(hull)
    ab = b - a
    tt = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
    q = a + tt[:, None] * ab
    return q[np.argmin(np.linalg.norm(q - p, axis=1))]


# --- 9. Stability margin -----------------------------------------------------


def stability_margin(point_xz: np.ndarray, hull: np.ndarray) -> float:
    """Signed distance from a floor point (2,) to the nearest hull edge. Positive inside.

    Distance is to edge segments, so outside a corner it is the distance to the corner.
    """
    p, h = np.asarray(point_xz, float), np.asarray(hull, float)
    if len(h) < 3:
        return float("nan")
    d = float(np.linalg.norm(_nearest_on_boundary(p, h) - p))
    return d if _inside_convex(p, h) else -d


def margin_components(point_xz: np.ndarray, hull: np.ndarray) -> tuple[float, float]:
    """(side_to_side, forward_back) margins in the floor frame (x, z).

    Inside: distance to the boundary along ±x (and ±z), the smaller of the two sides.
    Outside: per-axis offset to the nearest hull point, made negative.
    """
    p, h = np.asarray(point_xz, float), np.asarray(hull, float)
    if len(h) < 3:
        return float("nan"), float("nan")
    if not _inside_convex(p, h):
        q = _nearest_on_boundary(p, h)
        return -abs(float(p[0] - q[0])), -abs(float(p[1] - q[1]))
    out = []
    for ax in (0, 1):  # ax 0: move along x (line z = p_z); ax 1: move along z (line x = p_x)
        other = 1 - ax
        a, b = _edges(h)
        hits = []
        for (a0, b0) in zip(a, b):
            if (a0[other] - p[other]) * (b0[other] - p[other]) <= 0 and a0[other] != b0[other]:
                u = (p[other] - a0[other]) / (b0[other] - a0[other])
                hits.append(a0[ax] + u * (b0[ax] - a0[ax]))
        out.append(min(p[ax] - min(hits), max(hits) - p[ax]) if hits else float("nan"))
    return float(out[0]), float(out[1])


# --- 10. Sway heatmap --------------------------------------------------------


def sway_heatmap(V: np.ndarray) -> np.ndarray:
    """Per-vertex RMS displacement (V,) from the trial-mean position, V (F,V,3).

    Large values on the head/trunk with small at the feet suggest ankle strategy.
    """
    V = np.asarray(V, float)
    return np.sqrt(((V - V.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))


# --- 11. Noise floor ---------------------------------------------------------


def noise_floor(foot_verts: np.ndarray) -> dict[str, float]:
    """Jitter of static foot vertices (F,K,3) per axis: median over vertices of std over frames.

    Feet don't move during stance, so this is the reconstruction noise reported with every metric.
    """
    sd = np.asarray(foot_verts, float).std(axis=0, ddof=1)
    return {a: float(np.median(sd[:, i])) for i, a in enumerate("xyz")}


# --- 12. Stretch: extrapolated COM -------------------------------------------


def extrapolated_com(x: np.ndarray, v: np.ndarray, com_height: float, g: float = 9.81) -> np.ndarray:
    """Hof's XCoM: ξ = x + v/ω₀, ω₀ = √(g/ℓ), ℓ = com_height.

    Use the 30 Hz side-to-side velocity; 3 fps SAM frames are too coarse for v.
    """
    return np.asarray(x, float) + np.asarray(v, float) / np.sqrt(g / com_height)
