"""All replay math. Pure functions on numpy arrays, unit-tested in tests/test_geometry.py.

Conventions after alignment (step 4 on): y up, floor is the x–z plane, origin at the
stance-foot center, meters. Before alignment: SAM camera frame (see conventions.json).
Shapes: F frames, V vertices, K keypoints, N generic points.
"""

from __future__ import annotations

import numpy as np

# --- 1. Convention check -----------------------------------------------------


def project_pinhole(points: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """Project camera-frame points (N,3) to pixels (N,2): u = f·X/Z + W/2, v = f·Y/Z + H/2.

    Points with Z <= 0 map to NaN.
    """
    raise NotImplementedError


def reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray, focal: float, W: int, H: int) -> np.ndarray:
    """Per-point pixel distance (N,) between projected points_3d (N,3) and points_2d (N,2)."""
    raise NotImplementedError


# --- 2. Focal normalization --------------------------------------------------


def normalize_focal(cam_t: np.ndarray, focal: np.ndarray) -> np.ndarray:
    """cam_t (F,3), focal (F,) -> cam_t' (F,3) with tz' = tz · f̄ / f_i, f̄ = median focal.

    tx, ty unchanged.
    """
    raise NotImplementedError


# --- 3. Outlier rejection ----------------------------------------------------


def nearest_timestamp_index(t_query: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    """For each t_query (F,), index into sorted t_ref (M,) of the nearest timestamp."""
    raise NotImplementedError


def inlier_mask(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (F,) mask: True where the per-frame reprojection error <= threshold. NaN -> False."""
    raise NotImplementedError


# --- 4. Floor fit + gravity alignment ----------------------------------------


def fit_plane(points: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through points (N,3) via SVD of the centered points.

    Returns (centroid (3,), unit normal (3,)). Normal = right-singular vector with the
    smallest singular value, sign chosen so normal · up_hint > 0 (points toward the head).
    """
    raise NotImplementedError


def rotation_to_y(n: np.ndarray) -> np.ndarray:
    """Rotation matrix R (3,3) with R @ n = ŷ (Rodrigues).

    k = (n×ŷ)/|n×ŷ|, θ = arccos(n·ŷ), R = I + sinθ·K + (1−cosθ)·K².
    Handles n ≈ ŷ (identity) and n ≈ −ŷ (180° about any horizontal axis).
    """
    raise NotImplementedError


def align_to_floor(points: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Map camera-frame points (..., 3) to the floor frame: R @ (p − origin)."""
    raise NotImplementedError


# --- 5. Metric scale ---------------------------------------------------------


def metric_scale(patient_height_m: float, mesh_heights: np.ndarray) -> float:
    """s = H_patient / median(mesh_heights), mesh_heights (F,) = floor-to-head-top per frame."""
    raise NotImplementedError


# --- 6. Temporal smoothing ---------------------------------------------------


def gaussian_smooth(t: np.ndarray, X: np.ndarray, sigma: float = 0.3) -> np.ndarray:
    """Gaussian-kernel smoothing over non-uniform timestamps.

    t (F,) seconds, X (F, ...) -> X̂ (F, ...) with X̂_i = Σ_j w_ij X_j / Σ_j w_ij,
    w_ij = exp(−(t_i−t_j)² / 2σ²).
    """
    raise NotImplementedError


# --- 7. Center of mass -------------------------------------------------------


def center_of_mass(V: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Volumetric COM (3,) of a closed mesh with uniform density.

    Signed tetrahedra from the origin per face (a, b, c): volume a·(b×c)/6, centroid
    (a+b+c)/4; COM = Σ vol·centroid / Σ vol. Should match trimesh's center_mass.
    """
    raise NotImplementedError


def is_watertight(faces: np.ndarray) -> bool:
    """True if every edge is shared by exactly two faces."""
    raise NotImplementedError


# --- 8. Base of support ------------------------------------------------------


def contact_points(V: np.ndarray, floor_tol: float = 0.015) -> np.ndarray:
    """Floor-frame vertices (V,3) with y <= floor_tol, projected to the floor -> (N,2) as [x, z]."""
    raise NotImplementedError


def support_hull(points_xz: np.ndarray) -> np.ndarray:
    """Convex hull vertices (K,2) of floor points (N,2), counter-clockwise."""
    raise NotImplementedError


# --- 9. Stability margin -----------------------------------------------------


def stability_margin(point_xz: np.ndarray, hull: np.ndarray) -> float:
    """Signed distance from a floor point (2,) to the nearest hull edge. Positive inside."""
    raise NotImplementedError


def margin_components(point_xz: np.ndarray, hull: np.ndarray) -> tuple[float, float]:
    """(side_to_side, forward_back) margins: signed distance to the hull boundary along ±x and ±z.

    Each is the smaller of the two distances along that axis. Only defined for points inside
    the hull (outside, an axis line may miss the hull entirely); choose and document your
    own behavior there.
    """
    raise NotImplementedError


# --- 10. Sway heatmap --------------------------------------------------------


def sway_heatmap(V: np.ndarray) -> np.ndarray:
    """Per-vertex RMS displacement (V,) from the trial-mean position, V (F,V,3)."""
    raise NotImplementedError


# --- 11. Noise floor ---------------------------------------------------------


def noise_floor(foot_verts: np.ndarray) -> dict[str, float]:
    """Jitter of static foot vertices (F,K,3) per axis: {x, y, z} = median over vertices of std over frames."""
    raise NotImplementedError


# --- 12. Stretch: extrapolated COM -------------------------------------------


def extrapolated_com(x: np.ndarray, v: np.ndarray, com_height: float, g: float = 9.81) -> np.ndarray:
    """Hof's XCoM: ξ = x + v/ω₀, ω₀ = √(g/ℓ), ℓ = com_height."""
    raise NotImplementedError
