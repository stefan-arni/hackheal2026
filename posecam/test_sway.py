"""Sway tests (quiet / tandem / Romberg) and depth frames, on synthetic data.

    pytest -q test_sway.py
"""

import base64
import json
import math

import numpy as np
import pytest

from depth import decode_depth, rotate_intrinsics, torso_point
from pose_pipeline import PosePipeline
from sway import SwayConfig, SwaySession, romberg_ratios, sway_messages, sway_metrics
from ws_server import Recorder, decode_message

W, H = 640, 960
FPS = 30
# trunk: 200 px in the image, 0.5 m in world landmarks -> 0.25 cm per pixel
CM_PER_PX = 0.25


def make_pose(dx=0.0, lean_px=0.0, ankle_dx=0.0, depth_xyz=None, vis=0.95, swap_ankles=False):
    """Front view. `dx` shifts the whole body (px), `lean_px` moves the shoulders
    sideways relative to the hips, `ankle_dx` moves the left ankle (a step)."""
    lm = [{"x": 0.5, "y": 0.2, "z": 0.0, "visibility": vis} for _ in range(33)]
    world = [{"x": 0.0, "y": -0.6, "z": 0.0, "visibility": vis} for _ in range(33)]

    def put(i, x, y, wx, wy):
        lm[i] = {"x": x / W, "y": y / H, "z": 0.0, "visibility": vis}
        world[i] = {"x": wx, "y": wy, "z": 0.0, "visibility": vis}

    cx = 320 + dx
    put(11, cx - 40 + lean_px, 300, 0.18, -0.5)   # shoulders
    put(12, cx + 40 + lean_px, 300, -0.18, -0.5)
    put(23, cx - 30, 500, 0.1, 0.0)               # hips
    put(24, cx + 30, 500, -0.1, 0.0)
    la, ra = (27, 28) if not swap_ankles else (28, 27)   # model mixing up left/right
    put(la, cx - 30 + ankle_dx, 850, 0.1, 0.85)   # ankles
    put(ra, cx + 30, 850, -0.1, 0.85)
    pose = {"detected": True, "landmarks": lm, "world_landmarks": world}
    if depth_xyz is not None:
        pose["depth"] = {"torso_m": list(depth_xyz)}
    return pose


def run(sess, poses, t0=0.0):
    """Feed poses at 30 fps; return all events and the last status."""
    events, st = [], None
    for i, p in enumerate(poses):
        st = sess.update(t0 + i / FPS, p, W, H)
        events += st["events"]
    return events, st, t0 + len(poses) / FPS


def still(n, **kw):
    return [make_pose(**kw) for _ in range(n)]


def by_kind(events, kind):
    return [e for e in events if e["kind"] == kind]


# ------------------------------- metrics ------------------------------------ #

def test_metrics_sine_side_to_side():
    t = np.arange(0, 20, 1 / FPS)
    a, f = 0.02, 0.25                                  # 2 cm, 0.25 Hz
    ml = a * np.sin(2 * math.pi * f * t)
    m = sway_metrics(t, ml, None, smooth_s=0.2)
    assert m["rms_ml_cm"] == pytest.approx(2 / math.sqrt(2), rel=0.03)
    assert m["mean_velocity_cm_s"] == pytest.approx(4 * 2 * f, rel=0.05)   # 4 A f
    assert m["area_95_cm2"] is None and m["rms_ap_cm"] is None
    assert m["directional"]["dominant"] is None


def test_metrics_circle_area_and_no_dominant_direction():
    t = np.arange(0, 20, 1 / FPS)
    r, f = 0.01, 0.2
    ml, ap = r * np.cos(2 * math.pi * f * t), r * np.sin(2 * math.pi * f * t)
    m = sway_metrics(t, ml, ap, smooth_s=0.0)
    # circle radius r cm: covariance r^2/2 on each axis
    assert m["area_95_cm2"] == pytest.approx(math.pi * 5.991 * 1.0 ** 2 / 2, rel=0.03)
    assert m["path_length_cm"] == pytest.approx(2 * math.pi * 1.0 * f * 20, rel=0.03)
    assert m["directional"]["dominant"] == "no dominant direction"


def test_metrics_front_back_dominant():
    t = np.arange(0, 10, 1 / FPS)
    ap = 0.03 * np.sin(2 * math.pi * 0.3 * t)
    ml = 0.005 * np.sin(2 * math.pi * 0.2 * t)
    m = sway_metrics(t, ml, ap)
    assert m["directional"]["dominant"] == "front-back"
    assert m["main_axis_deg"] > 80
    assert m["directional"]["forward_cm"] == pytest.approx(3, abs=0.2)


def test_smoothing_removes_jitter_velocity():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 1 / FPS)
    ml = rng.normal(0, 0.003, len(t))                  # 3 mm landmark jitter, no real sway
    raw = sway_metrics(t, ml, None, smooth_s=0.0)["mean_velocity_cm_s"]
    smooth = sway_metrics(t, ml, None, smooth_s=0.2)["mean_velocity_cm_s"]
    assert smooth < raw / 3


def test_romberg_ratios():
    eo = {"mode": "depth", "lost_balance": False,
          "metrics": {"mean_velocity_cm_s": 1.0, "path_length_cm": 30.0, "area_95_cm2": 2.0,
                      "rms_ml_cm": 0.5, "rms_ap_cm": 0.6}}
    ec = {"mode": "depth", "lost_balance": True,
          "metrics": {"mean_velocity_cm_s": 2.5, "path_length_cm": 60.0, "area_95_cm2": 8.0,
                      "rms_ml_cm": 1.0, "rms_ap_cm": 1.2}}
    r = romberg_ratios(eo, ec)
    assert r["velocity_ratio"] == 2.5 and r["area_ratio"] == 4.0 and r["path_ratio"] == 2.0
    assert r["positive"] and r["mode"] == "depth"
    assert romberg_ratios(eo, None) is None


# ------------------------------- session ------------------------------------ #

def sway_poses(seconds, amp_px=8.0, f=0.3, **kw):
    n = int(seconds * FPS)
    return [make_pose(dx=amp_px * math.sin(2 * math.pi * f * i / FPS), **kw) for i in range(n)]


def test_quiet_stance_2d_end_to_end():
    sess = SwaySession(SwayConfig(duration_s=10, countdown_s=2))
    sess.start("quiet")
    ev, st, t = run(sess, still(2 * FPS + 1))
    assert by_kind(ev, "sway_started") and by_kind(ev, "sway_running")
    assert by_kind(ev, "sway_running")[0]["mode"] == "2d"
    ev, st, t = run(sess, sway_poses(10.2), t)
    done = by_kind(ev, "sway_done")[0]
    assert done["mode"] == "2d" and not done["lost_balance"]
    m = done["metrics"]
    assert m["rms_ml_cm"] == pytest.approx(8 * CM_PER_PX / math.sqrt(2), rel=0.1)
    assert m["area_95_cm2"] is None
    assert st["phase"] == "idle"
    assert sess.summary()["results"]["quiet"]["metrics"]["rms_ml_cm"] == m["rms_ml_cm"]


def test_depth_mode_measures_front_back():
    sess = SwaySession(SwayConfig(duration_s=8, countdown_s=2))
    sess.start("romberg_eo")
    _, _, t = run(sess, still(2 * FPS + 1, depth_xyz=(0.0, 0.0, 2.5)))
    poses = []
    for i in range(int(8.5 * FPS)):
        s = math.sin(2 * math.pi * 0.25 * i / FPS)
        c = math.cos(2 * math.pi * 0.25 * i / FPS)
        poses.append(make_pose(depth_xyz=(0.005 * c, 0.0, 2.5 - 0.02 * s)))   # 2 cm toward camera
    ev, _, _ = run(sess, poses, t)
    done = by_kind(ev, "sway_done")[0]
    assert done["mode"] == "depth"
    m = done["metrics"]
    assert m["rms_ap_cm"] == pytest.approx(2 / math.sqrt(2), rel=0.1)
    assert m["area_95_cm2"] > 0
    assert m["directional"]["dominant"] == "front-back"
    assert m["directional"]["forward_cm"] == pytest.approx(2, abs=0.3)


def started(duration=20, countdown=2, test="tandem", **kw):
    sess = SwaySession(SwayConfig(duration_s=duration, countdown_s=countdown))
    sess.start(test, **kw)
    _, _, t = run(sess, still(countdown * FPS + 1))
    return sess, t


def errors_of(ev):
    return [e for e in by_kind(ev, "sway_error")]


def test_step_counts_an_error_and_trial_continues():
    sess, t = started(duration=8)
    ev, st, t = run(sess, sway_poses(3) + still(FPS, ankle_dx=150), t)   # 150 px = 0.43 legs
    errs = errors_of(ev)
    assert len(errs) == 1 and errs[0]["error"] == "step" and errs[0]["counted"]
    assert errs[0]["t"] == pytest.approx(3.0, abs=0.1)
    assert st["phase"] == "running" and st["errors"] == 1 and "step" in st["active"]
    ev, _, _ = run(sess, still(5 * FPS, ankle_dx=150), t)                # stays there to the end
    done = by_kind(ev, "sway_done")[0]
    assert done["errors"] == 1 and done["by_type"]["step"] == 1
    assert done["lost_balance"] and done["lost_balance_at_s"] == pytest.approx(3.0, abs=0.1)
    assert done["duration_s"] == pytest.approx(8, abs=0.1)
    assert done["metrics"]["duration_s"] == pytest.approx(8, abs=0.2)


def test_second_step_after_settling_counts_again():
    sess, t = started()
    ev, _, t = run(sess, still(FPS, ankle_dx=150) + still(FPS, ankle_dx=150), t)   # step, settle
    ev2, _, _ = run(sess, still(FPS, ankle_dx=300), t)                              # another step
    assert len(errors_of(ev)) == 1
    assert len(errors_of(ev2)) == 1 and errors_of(ev2)[0]["counted"]


def test_returning_to_the_stance_is_not_another_step():
    sess, t = started()
    ev, _, t = run(sess, still(2 * FPS, ankle_dx=150), t)        # foot off, held (settles)
    ev2, st, t = run(sess, still(2 * FPS), t)                    # foot back where it started
    assert len(errors_of(ev)) == 1 and not errors_of(ev2)
    assert "step" not in st["active"]
    ev3, _, _ = run(sess, still(FPS, ankle_dx=150), t)           # a new step later counts
    assert len(errors_of(ev3)) == 1


def test_ankle_label_swap_is_not_a_step():
    sess, t = started()
    ev, st, _ = run(sess, still(2 * FPS, swap_ankles=True), t)
    assert not errors_of(ev) and st["errors"] == 0


def test_drift_past_limit_is_an_error():
    sess, t = started(test="quiet", drift_limit_cm=10)
    shift = 12 / CM_PER_PX                                   # 12 cm sideways (2D mode)
    ev, st, _ = run(sess, still(FPS, dx=shift), t)
    errs = errors_of(ev)
    assert [e["error"] for e in errs] == ["drift"]
    assert st["drift_cm"] == pytest.approx(12, abs=0.2)


def test_simultaneous_errors_count_once():
    sess, t = started(test="quiet", drift_limit_cm=10)
    # a big sideways stagger that also moves a foot: drift + step together
    ev, st, _ = run(sess, still(FPS, dx=12 / CM_PER_PX, ankle_dx=150), t)
    errs = errors_of(ev)
    assert {e["error"] for e in errs} == {"drift", "step"}
    assert sum(e["counted"] for e in errs) == 1 and st["errors"] == 1
    assert [e for e in errs if not e["counted"]][0]["not_counted_reason"] == "simultaneous"


def test_out_of_position_over_5s_adds_a_point():
    sess, t = started(duration=10, test="quiet", drift_limit_cm=10)
    ev, _, _ = run(sess, still(6 * FPS, dx=12 / CM_PER_PX), t)
    kinds = [e["error"] for e in errors_of(ev) if e["counted"]]
    assert kinds == ["drift", "out_of_position"]
    assert errors_of(ev)[-1]["t"] == pytest.approx(5.2, abs=0.15)   # 5 s after the drift counted


def test_staying_out_of_stance_after_a_step_adds_a_point():
    sess, t = started(duration=10)
    ev, st, _ = run(sess, still(7 * FPS, ankle_dx=150), t)     # step and stay there
    kinds = [e["error"] for e in errors_of(ev) if e["counted"]]
    assert kinds == ["step", "out_of_position"]
    assert "step" in st["active"]


def test_errors_cap_at_max():
    sess, t = started(duration=60, test="quiet", drift_limit_cm=10)
    poses = []
    for _ in range(14):                                       # 14 separate drifts
        poses += still(FPS // 2, dx=12 / CM_PER_PX) + still(FPS // 2)
    ev, st, _ = run(sess, poses, t)
    assert st["errors"] == 10
    assert errors_of(ev)[-1]["not_counted_reason"] == "max errors"


def test_session_scores_table():
    sess, t = started(duration=3, test="quiet")
    run(sess, still(FPS, ankle_dx=150) + still(3 * FPS, ankle_dx=150), t)
    s = sess.summary()
    assert s["scores"]["quiet"] == 1 and s["scores"]["tandem"] is None
    assert s["total_errors"] == 1 and not s["complete"]
    r = s["results"]["quiet"]
    assert r["errors"] == 1 and r["log"][0]["error"] == "step"
    assert r["trail"] and len(r["trail"][0]) == 3


def test_lean_alert_past_limit():
    sess = SwaySession(SwayConfig(duration_s=6, countdown_s=2))
    sess.start("quiet", lean_limit_deg=10)
    _, _, t = run(sess, still(2 * FPS + 1))
    lean_px = 200 * math.tan(math.radians(15))          # 15 deg toward image-right
    ev, _, _ = run(sess, still(FPS) + still(2 * FPS, lean_px=lean_px) + still(3 * FPS + 5), t)
    alerts = by_kind(ev, "sway_lean")
    assert [e["error"] for e in errors_of(ev)] == ["lean"]
    assert alerts[0]["event"] == "start" and alerts[0]["side"] == "left"
    assert alerts[0]["lean_deg"] == pytest.approx(15, abs=0.5)
    assert alerts[1]["event"] == "end"
    lean = by_kind(ev, "sway_done")[0]["trunk_lean"]
    assert lean["times_over_limit"] == 1 and lean["left_deg"] == pytest.approx(15, abs=0.5)
    assert lean["time_over_limit_s"] == pytest.approx(1.7, abs=0.2)
    msgs = sway_messages({"sway": {"events": alerts}})
    assert msgs[0]["type"] == "alert"


def test_body_lost_fails_trial():
    sess = SwaySession(SwayConfig(duration_s=10, countdown_s=1))
    sess.start("quiet")
    _, _, t = run(sess, still(FPS + 1))
    ev, st, _ = run(sess, [{"detected": False}] * (3 * FPS), t)
    assert by_kind(ev, "sway_failed") and st["phase"] == "idle"


def test_commands():
    sess = SwaySession()
    assert sess.handle_command({"type": "sway_start", "test": "nope"})[0]["type"] == "error"
    assert sess.handle_command({"type": "sway_start", "test": "quiet", "duration": 12})[0]["type"] == "ack"
    assert sess.duration_s == 12
    assert sess.handle_command({"type": "sway_start", "test": "quiet"})[0]["type"] == "error"
    assert sess.handle_command({"type": "sway_cancel"})[0]["type"] == "ack"
    assert sess.phase == "idle"


# ------------------------------- pipeline ----------------------------------- #

class FakePose:
    def __init__(self):
        self.pose = make_pose()

    def process_bgr(self, frame):
        return dict(self.pose)

    def close(self):
        pass


def test_pipeline_runs_sway_and_pauses_single_leg():
    from balance import BalanceConfig
    from bess import BessConfig
    pipe = PosePipeline(FakePose(), BalanceConfig(), None, BessConfig(), sway=SwayConfig(countdown_s=1))
    frame = np.zeros((H, W, 3), np.uint8)
    assert pipe.handle_command({"type": "sway_start", "test": "quiet"})[0]["type"] == "ack"
    out = pipe.process_bgr(frame, t=0.0)
    assert out["sway"]["phase"] == "countdown"
    assert out["balance"]["state"] == "paused" and out["balance"]["paused_for"] == "sway"
    # no BESS while a sway test runs
    assert pipe.handle_command({"type": "bess_start", "stance": "double"})[0]["type"] == "error"


def test_pipeline_uses_capture_timestamp():
    pipe = PosePipeline(FakePose(), None, None, None, sway=SwayConfig(countdown_s=2))
    frame = np.zeros((H, W, 3), np.uint8)
    pipe.handle_command({"type": "sway_start", "test": "quiet"})
    pipe.process_bgr(frame, meta={"timestamp_ms": 1000.0})
    out = pipe.process_bgr(frame, meta={"timestamp_ms": 2500.0})
    assert out["sway"]["countdown_left"] == pytest.approx(0.5)


# ------------------------------- depth -------------------------------------- #

def unproject(u, v, z, K):
    fx, fy, cx, cy = K
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_rotated_intrinsics_keep_3d_distances(rot):
    import cv2
    w, h = 256, 192
    K = (210.0, 211.0, 127.0, 96.5)
    pts = [(10, 20), (200, 150), (128, 96)]
    img = np.zeros((h, w), np.uint16)
    for i, (u, v) in enumerate(pts):
        img[v, u] = i + 1
    rots = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    r = cv2.rotate(img, rots[rot]) if rot else img
    K2 = rotate_intrinsics(K, w, h, rot)
    a = [unproject(u, v, 2.0, K) for u, v in pts]
    b = []
    for i in range(len(pts)):
        v2, u2 = np.argwhere(r == i + 1)[0]
        b.append(unproject(u2, v2, 2.0, K2))
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.linalg.norm(a[i] - a[j]) == pytest.approx(np.linalg.norm(b[i] - b[j]), rel=1e-3)
    # the optical centre stays the optical centre
    assert np.linalg.norm(b[2][:2]) == pytest.approx(np.linalg.norm(a[2][:2]), abs=0.01)


def depth_message(depth_mm: np.ndarray, rot=0, fmt="uint16_mm"):
    import cv2
    h, w = depth_mm.shape
    ok, jpg = cv2.imencode(".jpg", np.zeros((h * 2, w * 2, 3), np.uint8))
    data = depth_mm.astype("<u2") if fmt == "uint16_mm" else (depth_mm / 1000).astype("<f2")
    return json.dumps({"type": "frame", "image": base64.b64encode(jpg.tobytes()).decode(),
                       "rotate": rot, "depth": base64.b64encode(data.tobytes()).decode(),
                       "depth_format": fmt, "depth_size": [w, h],
                       "intrinsics": [200.0, 200.0, (w - 1) / 2, (h - 1) / 2], "camera": "lidar"})


@pytest.mark.parametrize("fmt", ["uint16_mm", "float16_m"])
def test_decode_depth_frame_rotated(fmt):
    d = np.full((192, 256), 2000, np.uint16)
    d[0, 0] = 1234
    frame, meta = decode_message(depth_message(d, rot=90, fmt=fmt))
    assert frame.shape[:2] == (512, 384)                    # image rotated to portrait
    assert meta["_depth"].shape == (256, 192)               # depth rotated the same way
    assert meta["_depth"][0, 191] == pytest.approx(1.234, abs=1e-3)   # top-left -> top-right
    assert "depth" not in meta


def test_torso_point_reads_depth_inside_torso():
    pose = make_pose()
    depth = np.full((240, 160), 3.0, np.float32)            # background 3 m
    # torso region (in depth pixels: image coords / 4) at 2.2 m, with a few holes
    depth[75:125, 65:95] = 2.2
    depth[90:93, 70:90] = 0.0
    K = (150.0, 150.0, 80.0, 120.0)
    tp = torso_point(pose["landmarks"], depth, K)
    assert tp["torso_m"][2] == pytest.approx(2.2)
    com_u = 0.65 * 80 + 0.35 * 80                           # x of hip-mid / shoulder-mid
    assert tp["torso_m"][0] == pytest.approx((com_u - 80) * 2.2 / 150, abs=1e-3)


def test_recorder_round_trip(tmp_path):
    rec = Recorder(str(tmp_path), "s1")
    d = np.full((12, 16), 1500, np.uint16)
    rec.write(depth_message(d))
    rec.write(json.dumps({"type": "sway_start", "test": "quiet"}))
    rec.write(b"\xff\xd8not-a-real-jpeg")
    rec.close()
    lines = open(rec.path).read().splitlines()
    assert len(lines) == 3
    frame, meta = decode_message(lines[0])
    assert meta["_depth"][0, 0] == pytest.approx(1.5)
    assert json.loads(lines[1])["type"] == "sway_start"
    assert json.loads(lines[2])["type"] == "frame"


def test_custom_countdown_to_walk_back():
    sess = SwaySession(SwayConfig(countdown_s=5))
    sess.handle_command({"type": "sway_start", "test": "quiet", "countdown": 15})
    ev, st, t = run(sess, still(14 * FPS))
    assert by_kind(ev, "sway_started")[0]["countdown_s"] == 15
    assert st["phase"] == "countdown" and st["countdown_left"] == pytest.approx(1.0, abs=0.05)
    ev, st, _ = run(sess, still(FPS + 2), t)
    assert by_kind(ev, "sway_running") and st["phase"] == "running"


def test_sway_tracker_measures_alongside():
    from sway import SwayTracker
    tr = SwayTracker(SwayConfig())
    for i in range(10 * FPS):
        tr.update(i / FPS, make_pose(dx=8 * math.sin(2 * math.pi * 0.3 * i / FPS)), W, H)
    m = tr.metrics()
    assert m["mode"] == "2d" and m["rms_ml_cm"] == pytest.approx(8 * CM_PER_PX / math.sqrt(2), rel=0.15)
    assert tr.live_cm() is not None


def test_bess_summary_has_per_stance_breakdown():
    from bess import BessConfig, BessSession
    sess = BessSession(BessConfig(track_eyes=True))
    s = sess.session_summary()
    assert s["by_stance"] == {"double": None, "tandem": None, "single": None}
    assert "eyes_open" in s["error_types"]


def test_ankle_depths_reads_each_foot():
    from depth import ankle_depths
    pose = make_pose()
    depth = np.full((240, 160), 3.0, np.float32)
    lm = pose["landmarks"]
    for i, z in ((27, 2.6), (28, 2.3)):          # left ankle further than right
        u, v = int(round(lm[i]["x"] * 160)), int(round(lm[i]["y"] * 240))
        depth[v - 3:v + 4, u - 3:u + 4] = z
    d = ankle_depths(lm, depth)
    assert d["left"] == pytest.approx(2.6) and d["right"] == pytest.approx(2.3)
