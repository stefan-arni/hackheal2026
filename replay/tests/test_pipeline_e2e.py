"""End-to-end: SYNTHETIC run -> pipeline (geometry.py) -> compare with ground_truth.json.

Fails with NotImplementedError until geometry.py is implemented.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from service import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_synthetic_run import generate  # noqa: E402

FPS = 3.0
SEED = int(os.environ.get("REPLAY_E2E_SEED", 1))
FRAME_TOL_S = 1 / FPS + 0.05  # one frame period


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    gt = generate(out, seed=SEED, fps=FPS, render_images=False)
    run = pipeline.load_run(out)
    res = pipeline.process(run, events=gt["events"], patient_height_m=gt["patient_height_m"])
    return gt, run, res


def hull_area(h):
    h = np.asarray(h)
    if len(h) < 3:
        return 0.0
    x, z = h[:, 0], h[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


# --- floor -------------------------------------------------------------------------


def test_floor_normal(synthetic):
    gt, _, res = synthetic
    n_true = np.asarray(gt["floor_plane_camera"]["normal"])
    n = np.asarray(res["floor_normal_cam"])
    angle = np.degrees(np.arccos(np.clip(n @ n_true / np.linalg.norm(n), -1, 1)))
    assert angle < 1.0, f"floor normal off by {angle:.2f}°"


def test_floor_origin(synthetic):
    # Absolute depth is unobservable: focal normalization puts the scene at tz·f̄/f_true,
    # a constant shift along camera z (harmless in the floor frame). Compare after it.
    gt, run, res = synthetic
    f_bar = np.median(run.focal)
    tz_norm = np.median(run.cam_t[:, 2] * f_bar / run.focal)
    dz = tz_norm * (1 - gt["focal_true"] / f_bar)
    expected = np.asarray(gt["floor_plane_camera"]["point"]) + [0.0, 0.0, dz]
    err = np.linalg.norm(np.asarray(res["origin_cam"]) - expected)
    assert err < 0.02, f"origin off by {err * 100:.1f} cm (after the {dz * 100:.1f} cm global depth shift)"


def test_floor_frame_axes_match_ground_truth(synthetic):
    # x = patient's left (image right), y = up, z = toward camera
    gt, _, res = synthetic
    R_true = np.asarray(gt["floor_frame"]["R_cam_to_floor"])
    np.testing.assert_allclose(res["R_cam_to_floor"], R_true, atol=0.03)


def test_metric_scale_near_one(synthetic):
    _, _, res = synthetic
    assert res["scale"] == pytest.approx(1.0, abs=0.02)


def test_feet_on_floor(synthetic):
    gt, _, res = synthetic
    left = res["verts"][:, gt["foot_vertices"]["left"], 1]  # front foot never lifts
    assert abs(np.percentile(left, 1)) < 0.01


# --- COM ---------------------------------------------------------------------------


def test_com_height(synthetic):
    gt, _, res = synthetic
    com_true = np.asarray(gt["com_floor"])
    assert res["com"][:, 1].mean() == pytest.approx(com_true[:, 1].mean(), abs=0.015)


def test_com_horizontal_position(synthetic):
    gt, _, res = synthetic
    com_true = np.asarray(gt["com_floor"])
    err = np.abs(res["com"][:, [0, 2]].mean(0) - com_true[:, [0, 2]].mean(0))
    assert (err < 0.015).all(), f"mean COM (x, z) off by {err * 100} cm"


@pytest.mark.parametrize("axis, name", [(0, "side-to-side (x)"), (2, "forward-back (z)")])
def test_com_sway_track(synthetic, axis, name):
    gt, _, res = synthetic
    true = np.asarray(gt["com_floor"])[:, axis]
    est = res["com"][:, axis]
    true, est = true - true.mean(), est - est.mean()
    rms = np.sqrt(np.mean((est - true) ** 2))
    corr = np.corrcoef(est, true)[0, 1]
    assert rms < 0.004, f"{name}: sway RMS error {rms * 1000:.1f} mm"
    assert corr > 0.9, f"{name}: correlation {corr:.2f}"


def test_watertight_and_topology(synthetic):
    gt, run, res = synthetic
    assert res["quality"]["watertight"]
    assert res["verts"].shape == (len(gt["t_ms"]), gt["vertex_count"], 3)


# --- events (timing) ------------------------------------------------------------------


def test_foot_lift_timing_from_bos(synthetic):
    """The BOS shrinks to one foot while the rear foot is up."""
    gt, _, res = synthetic
    t = res["t_s"]
    area = np.array([hull_area(h) for h in res["bos"]])
    lifted = area < 0.65 * np.median(area)
    assert lifted.any(), "no frame with a shrunken BOS"
    t_lift = t[np.argmax(lifted)]
    t_down = t[len(lifted) - np.argmax(lifted[::-1])]  # first frame after the last lifted one
    ev = {e["kind"]: e["t"] / 1000 for e in gt["events"]}
    assert abs(t_lift - ev["foot_lift"]) <= FRAME_TOL_S, f"lift detected at {t_lift:.2f}s, truth {ev['foot_lift']}s"
    assert abs(t_down - ev["foot_down"]) <= FRAME_TOL_S, f"touchdown detected at {t_down:.2f}s, truth {ev['foot_down']}s"


def test_side_margin_drops_before_touchdown(synthetic):
    """Money shot: the COM leans toward the side edge of the BOS before the foot comes down."""
    gt, _, res = synthetic
    t = res["t_s"]
    ev = {e["kind"]: e["t"] / 1000 for e in gt["events"]}
    window = (t > ev["foot_lift"] - 1.5) & (t < ev["foot_down"])
    calm = t < ev["foot_lift"] - 3.0
    assert (res["margin"][calm] > 0).all(), "COM outside BOS during calm stance"
    # nanmin: side-to-side margin may be undefined while the COM is outside a one-foot BOS
    assert np.nanmin(res["margin_ml"][window]) < np.median(res["margin_ml"][calm]) - 0.005


# --- noise floor ------------------------------------------------------------------------


def test_noise_floor(synthetic):
    gt, _, res = synthetic
    nf = res["noise_floor"]
    sigma = gt["noise"]["vertex_noise_m"]
    for a in "xyz":
        assert 0.5 * sigma < nf[a] < 2 * sigma, f"{a} jitter {nf[a] * 1000:.1f} mm vs injected {sigma * 1000:.1f} mm"
    assert res["quality"]["ap_real"]  # coupled depth error must be removed by focal normalization


def test_gravity_override_and_check(synthetic):
    """Phone gravity path: the true 'up' as up_cam reproduces the floor; up_check only reports."""
    gt, run, res = synthetic
    up = np.asarray(gt["floor_plane_camera"]["normal"])
    over = pipeline.process(run, gt["events"], gt["patient_height_m"], up_cam=up, up_source="phone_gravity")
    angle = np.degrees(np.arccos(np.clip(np.asarray(over["floor_normal_cam"]) @ up, -1, 1)))
    assert angle < 0.01 and over["quality"]["floor_source"] == "phone_gravity"
    assert over["com"][:, 1].mean() == pytest.approx(np.asarray(gt["com_floor"])[:, 1].mean(), abs=0.015)
    chk = pipeline.process(run, gt["events"], gt["patient_height_m"], up_check=up)
    assert chk["quality"]["gravity_vs_feet_deg"] < 1.0
    np.testing.assert_allclose(chk["com"], res["com"])  # a check never changes the result


# --- analytics ----------------------------------------------------------------------------

def test_analytics_on_synthetic(synthetic):
    from service import analytics
    gt, _, res = synthetic
    a = analytics.compute(res, gt["events"], expected_stance="tandem")
    ev = {e["kind"]: e["t"] / 1000 for e in gt["events"]}
    t0 = res["t_s"][0]
    # minimum margin happens during the pre-error lean / lift, not in calm stance
    assert ev["foot_lift"] - 1.5 <= a["margin"]["min_at_ms"] / 1000 - t0 <= ev["foot_down"] + 0.5
    assert a["margin"]["time_outside_bos_s"] >= 0
    # side-to-side sway RMS matches the true COM within 3 mm
    true_x = np.asarray(gt["com_floor"])[:, 0]
    assert a["ml_sway"]["rms_cm"] == pytest.approx(true_x.std() * 100, abs=0.3)
    # stance: tandem most of the time, a single-leg (right lifted -> standing on left) segment near the lift
    st = a["stance"]
    assert st["seconds"].get("double_tandem", 0) > 15
    singles = [s for s in st["segments"] if s["stance"] == "single_left"]
    assert singles and abs(singles[0]["start_ms"] / 1000 - ev["foot_lift"]) < 0.6
    assert st["expected"]["stance"] == "tandem" and st["expected"]["fraction_matching"] > 0.85
    assert a["trunk_lean_deg"]["ml_range"] < 10 and a["quality"]["label"] in ("good", "fair", "poor")
    assert a["noise_floor_cm"]["x"] < 1.0
