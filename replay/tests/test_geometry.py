"""Unit tests for service/geometry.py on synthetic data with known answers."""

import numpy as np
import pytest
import trimesh

from service import geometry as g

rng = np.random.default_rng(0)


def random_unit(n=1):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def shoelace(poly):
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)


# --- 1. projection -----------------------------------------------------------


def test_project_pinhole_known_point():
    uv = g.project_pinhole(np.array([[1.0, 2.0, 4.0]]), focal=100.0, W=200, H=100)
    np.testing.assert_allclose(uv, [[125.0, 100.0]])


def test_project_pinhole_principal_point_and_behind_camera():
    uv = g.project_pinhole(np.array([[0.0, 0.0, 3.0], [1.0, 1.0, -1.0]]), focal=500.0, W=640, H=480)
    np.testing.assert_allclose(uv[0], [320.0, 240.0])
    assert np.isnan(uv[1]).all()


def test_reprojection_error():
    P = np.array([[0.1, -0.2, 2.0], [0.3, 0.4, 3.0]])
    uv = g.project_pinhole(P, 800.0, 640, 480)
    np.testing.assert_allclose(g.reprojection_error(P, uv, 800.0, 640, 480), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(g.reprojection_error(P, uv + [3.0, 4.0], 800.0, 640, 480), [5.0, 5.0])


# --- 2. focal normalization ----------------------------------------------------


def test_normalize_focal():
    cam_t = np.array([[0.1, 0.2, 1.0], [0.3, 0.4, 1.0], [0.5, 0.6, 1.0]])
    out = g.normalize_focal(cam_t, np.array([100.0, 200.0, 300.0]))
    np.testing.assert_allclose(out[:, :2], cam_t[:, :2])
    np.testing.assert_allclose(out[:, 2], [2.0, 1.0, 2.0 / 3.0])


def test_normalize_focal_constant_focal_is_identity():
    cam_t = rng.normal(size=(10, 3))
    np.testing.assert_allclose(g.normalize_focal(cam_t, np.full(10, 1234.0)), cam_t)


# --- 3. outlier rejection ------------------------------------------------------


def test_nearest_timestamp_index():
    idx = g.nearest_timestamp_index(np.array([-5.0, 49.0, 51.0, 260.0]), np.array([0.0, 100.0, 200.0]))
    np.testing.assert_array_equal(idx, [0, 0, 1, 2])


def test_inlier_mask():
    mask = g.inlier_mask(np.array([1.0, 5.0, np.nan, 2.0]), threshold=2.0)
    np.testing.assert_array_equal(mask, [True, False, False, True])


# --- 4. floor fit + alignment --------------------------------------------------


def plane_points(normal, point, n=200, noise=0.0):
    normal = normal / np.linalg.norm(normal)
    a = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(normal, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    uv = rng.uniform(-0.5, 0.5, size=(n, 2))
    return point + uv[:, :1] * a + uv[:, 1:] * b + noise * rng.normal(size=(n, 1)) * normal


def test_fit_plane_recovers_known_plane():
    n_true = np.array([0.1, -0.95, 0.3])
    n_true /= np.linalg.norm(n_true)
    pts = plane_points(n_true, np.array([0.2, 1.5, 3.0]), noise=1e-4)
    centroid, n = g.fit_plane(pts, up_hint=np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(centroid, pts.mean(axis=0))
    np.testing.assert_allclose(np.linalg.norm(n), 1.0)
    np.testing.assert_allclose(n, n_true, atol=1e-3)


def test_fit_plane_normal_follows_up_hint():
    pts = plane_points(np.array([0.0, 1.0, 0.0]), np.zeros(3))
    _, n_up = g.fit_plane(pts, up_hint=np.array([0.0, 1.0, 0.0]))
    _, n_down = g.fit_plane(pts, up_hint=np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(n_up, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(n_down, [0.0, -1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("n", list(random_unit(20)))
def test_rotation_to_y_random(n):
    R = g.rotation_to_y(n)
    np.testing.assert_allclose(R @ n, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("n", [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 1e-12]])
def test_rotation_to_y_degenerate(n):
    n = np.asarray(n) / np.linalg.norm(n)
    R = g.rotation_to_y(n)
    np.testing.assert_allclose(R @ n, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0)


def test_floor_alignment_flattens_tilted_floor():
    # camera-frame floor, tilted, with y-down camera: "up" is roughly -y
    n_true = np.array([0.0, -0.9, -0.4])
    pts = plane_points(n_true, np.array([0.0, 1.6, 3.0]))
    head = np.array([0.0, 0.0, 3.0])  # above the floor
    centroid, n = g.fit_plane(pts, up_hint=head - pts.mean(axis=0))
    R = g.rotation_to_y(n)
    out = g.align_to_floor(pts, R, centroid)
    np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-9)
    assert g.align_to_floor(head, R, centroid)[1] > 0


def test_align_to_floor_batched_shape():
    X = rng.normal(size=(4, 7, 3))
    assert g.align_to_floor(X, np.eye(3), np.zeros(3)).shape == (4, 7, 3)
    np.testing.assert_allclose(g.align_to_floor(X, np.eye(3), np.ones(3)), X - 1.0)


def test_true_to_sam_rotation():
    np.testing.assert_allclose(g.true_to_sam_rotation(np.array([540.0, 960.0]), (1080, 1920), 1400.0), np.eye(3), atol=1e-12)
    R = g.true_to_sam_rotation(np.array([560.0, 1174.5]), (1080, 1920), 1384.0)
    ray = np.array([20.0 / 1384, 214.5 / 1384, 1.0])
    np.testing.assert_allclose(R @ (ray / np.linalg.norm(ray)), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # crop center below the image center: SAM's camera is pitched down relative to the true one,
    # so true "up" appears tilted toward SAM's forward axis... away from it: up_z becomes negative
    up_sam = R @ np.array([0.0, -1.0, 0.0])
    assert up_sam[2] < 0 and np.degrees(np.arcsin(-up_sam[2])) == pytest.approx(np.degrees(np.arctan(214.5 / 1384)), abs=0.5)


# --- 5. metric scale -----------------------------------------------------------


def test_metric_scale():
    assert g.metric_scale(1.7, np.array([0.8, 0.85, 0.9, 5.0, 0.85])) == pytest.approx(2.0)


# --- 6. smoothing ----------------------------------------------------------------


def nonuniform_times():
    uniform = np.arange(0, 6, 1 / 3)
    burst = np.arange(3.0, 4.0, 0.1)  # ~10 fps burst around an error
    return np.unique(np.round(np.concatenate([uniform, burst]), 6))


def test_gaussian_smooth_constant_unchanged():
    t = nonuniform_times()
    X = np.ones((len(t), 5, 3)) * [1.0, 2.0, 3.0]
    np.testing.assert_allclose(g.gaussian_smooth(t, X, 0.3), X)


def test_gaussian_smooth_preserves_shape_and_limits():
    t = nonuniform_times()
    X = rng.normal(size=(len(t), 4, 3))
    assert g.gaussian_smooth(t, X, 0.3).shape == X.shape
    np.testing.assert_allclose(g.gaussian_smooth(t, X, 1e-6), X, atol=1e-9)  # tiny σ -> identity
    np.testing.assert_allclose(g.gaussian_smooth(t, X, 1e6), np.broadcast_to(X.mean(0), X.shape), atol=1e-6)


def test_gaussian_smooth_linear_interior_uniform():
    t = np.arange(0, 10, 0.1)
    X = 2.0 * t + 1.0
    out = g.gaussian_smooth(t, X, 0.3)
    interior = (t > 2) & (t < 8)  # away from edge effects
    np.testing.assert_allclose(out[interior], X[interior], atol=1e-6)


def test_gaussian_smooth_reduces_noise():
    t = nonuniform_times()
    X = rng.normal(scale=0.01, size=(len(t), 50, 3))
    assert g.gaussian_smooth(t, X, 0.3).std() < 0.8 * X.std()


# --- 7. center of mass -----------------------------------------------------------


def test_com_translated_box():
    box = trimesh.creation.box(extents=[0.3, 1.0, 0.2])
    box.apply_translation([1.0, 2.0, 3.0])
    np.testing.assert_allclose(g.center_of_mass(box.vertices, box.faces), [1.0, 2.0, 3.0], atol=1e-9)


def test_com_sphere():
    s = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    s.apply_translation([-0.2, 0.9, 2.5])
    np.testing.assert_allclose(g.center_of_mass(s.vertices, s.faces), [-0.2, 0.9, 2.5], atol=1e-9)


def test_com_two_boxes_weighted_by_volume():
    # unit cube at x=0 and a 2x1x1 box at x=3: COM x = (1·0 + 2·3)/3 = 2
    a = trimesh.creation.box(extents=[1, 1, 1])
    b = trimesh.creation.box(extents=[2, 1, 1])
    b.apply_translation([3.0, 0.0, 0.0])
    m = trimesh.util.concatenate([a, b])
    np.testing.assert_allclose(g.center_of_mass(m.vertices, m.faces), [2.0, 0.0, 0.0], atol=1e-9)


def test_com_matches_trimesh_on_irregular_mesh():
    m = trimesh.convex.convex_hull(rng.normal(size=(60, 3)) * [0.2, 0.9, 0.15] + [0.1, 1.0, 3.0])
    np.testing.assert_allclose(g.center_of_mass(m.vertices, m.faces), m.center_mass, atol=1e-9)


def test_is_watertight():
    box = trimesh.creation.box()
    assert g.is_watertight(box.faces)
    assert not g.is_watertight(box.faces[1:])


# --- 8. base of support ------------------------------------------------------------


def test_contact_points():
    V = np.array([[0.1, 0.0, 0.2], [0.3, 0.01, 0.4], [0.5, 0.02, 0.6], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(g.contact_points(V, floor_tol=0.015), [[0.1, 0.2], [0.3, 0.4]])


def test_support_hull_square_with_interior_points():
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.vstack([corners, rng.uniform(0.1, 0.9, size=(30, 2))])
    hull = g.support_hull(pts)
    assert len(hull) == 4
    assert {tuple(p) for p in hull} == {tuple(p) for p in corners}
    assert shoelace(hull) == pytest.approx(1.0)  # positive => counter-clockwise in (x, z)


# --- 9. stability margin -------------------------------------------------------------

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.mark.parametrize(
    "p, expected",
    [
        ([0.5, 0.5], 0.5),
        ([0.2, 0.5], 0.2),
        ([0.5, 0.9], 0.1),
        ([1.0, 0.5], 0.0),
        ([1.5, 0.5], -0.5),
        ([2.0, 2.0], -np.sqrt(2.0)),  # nearest point on the boundary is the corner
    ],
)
def test_stability_margin_square(p, expected):
    assert g.stability_margin(np.array(p), SQUARE) == pytest.approx(expected)


def test_stability_margin_orientation_independent():
    assert g.stability_margin(np.array([0.2, 0.5]), SQUARE[::-1]) == pytest.approx(0.2)


def test_margin_components_rectangle():
    rect = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    side, fwd = g.margin_components(np.array([0.5, 0.3]), rect)
    assert side == pytest.approx(0.5)
    assert fwd == pytest.approx(0.3)
    side, fwd = g.margin_components(np.array([1.8, 0.95]), rect)
    assert side == pytest.approx(0.2)
    assert fwd == pytest.approx(0.05)


def test_margin_components_outside_is_negative_offset_to_nearest_point():
    rect = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    side, fwd = g.margin_components(np.array([2.5, 0.5]), rect)  # nearest point (2, 0.5)
    assert side == pytest.approx(-0.5) and fwd == pytest.approx(0.0)
    side, fwd = g.margin_components(np.array([3.0, 2.0]), rect)  # nearest point: corner (2, 1)
    assert side == pytest.approx(-1.0) and fwd == pytest.approx(-1.0)


# --- 10. sway heatmap -------------------------------------------------------------------


def test_sway_heatmap():
    a = 0.03
    V = np.zeros((2, 3, 3))
    V[:, 1, 0] = [a, -a]  # oscillates ±a in x
    V[:, 2] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    V[:, 0] = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]  # static, off origin
    np.testing.assert_allclose(g.sway_heatmap(V), [0.0, a, 0.0], atol=1e-12)


# --- 11. noise floor --------------------------------------------------------------------


def test_noise_floor_recovers_per_axis_std():
    sigma = np.array([0.002, 0.005, 0.02])
    base = rng.normal(size=(1, 40, 3))
    foot = base + rng.normal(size=(4000, 40, 3)) * sigma
    nf = g.noise_floor(foot)
    assert set(nf) == {"x", "y", "z"}
    np.testing.assert_allclose([nf["x"], nf["y"], nf["z"]], sigma, rtol=0.05)


def test_noise_floor_static_is_zero():
    foot = np.broadcast_to(rng.normal(size=(1, 10, 3)), (20, 10, 3))
    nf = g.noise_floor(foot)
    for a in "xyz":
        assert nf[a] == pytest.approx(0.0, abs=1e-12)


# --- 12. extrapolated COM -----------------------------------------------------------------


def test_extrapolated_com():
    # ℓ = g/4 => ω₀ = 2
    np.testing.assert_allclose(g.extrapolated_com(np.array([0.1]), np.array([1.0]), 9.81 / 4), [0.6])
    np.testing.assert_allclose(g.extrapolated_com(np.array([0.1]), np.array([0.0]), 1.0), [0.1])
