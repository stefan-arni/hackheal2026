"""BESS tests on synthetic bodies (no MediaPipe model needed).

    pytest -q
"""

import asyncio
import json
import math

import cv2
import numpy as np
import pytest
import websockets

import ws_server
from bess import BessConfig, BessSession, hip_angles
from pose_analyzer import PoseResult, spine_angles_2d, spine_angles_3d
from pose_pipeline import PosePipeline, pose_events

W, H = 720, 1280
PX = 600            # pixels per meter (orthographic projection)
FPS = 15
DT = 1 / FPS


def body(stance="double", nondominant="left", hands="hips", step=None, fall=0.0,
         flex=None, abd=None, heel_lift=None, toe_lift=None, touch=False, visible=True,
         lean=0.0, side_bend=0.0, sway=0.0, scale=1.0, z_noise=0.0, rng=None):
    """A person in a BESS stance, in MediaPipe world coords (meters, origin at hips,
    y down, z toward camera negative). Returns (image landmarks, world landmarks).

    step: (side, dx_m) moves that foot sideways. flex/abd: {side: degrees}.
    heel_lift/toe_lift: {side: meters}. touch: single-leg raised foot on the floor.
    lean: bend forward at the waist toward the camera (deg). side_bend: bend sideways (deg).
    sway: whole body tips forward from the ankles, staying straight (deg).
    scale: apparent size (0.9 = stepped back from the camera). z_noise: depth noise (m)
    added to the 3D landmarks, like MediaPipe's.
    """
    dom = "right" if nondominant == "left" else "left"
    flex, abd = dict(flex or {}), dict(abd or {})
    heel_lift, toe_lift = heel_lift or {}, toe_lift or {}
    sx = {"left": 1, "right": -1}          # left is +x
    world = [None] * 33

    def put(i, x, y, z=0.0):
        world[i] = [x, y + fall, z]

    put(0, 0, -0.75, -0.08)                     # nose
    put(7, 0.07, -0.72, 0.0)                    # ears
    put(8, -0.07, -0.72, 0.0)
    for i in range(1, 11):
        if world[i] is None:
            put(i, 0, -0.74, -0.05)
    put(11, 0.18, -0.5)
    put(12, -0.18, -0.5)                        # shoulders
    for i in (13, 14):
        put(i, 0.25 * (1 if i == 13 else -1), -0.25)   # elbows out
    for side, (wr, hip) in {"left": (15, 23), "right": (16, 24)}.items():
        if hands == "hips" or (hands != "down" and hands != side):
            put(wr, sx[side] * 0.16, -0.05)
        else:
            put(wr, sx[side] * 0.22, 0.25)      # hand dropped to the side
    for i in range(17, 23):
        put(i, 0, 0)
    put(23, 0.1, 0.0)
    put(24, -0.1, 0.0)                          # hips

    for side, (hip, knee, ank, heel, toe) in {"left": (23, 25, 27, 29, 31),
                                              "right": (24, 26, 28, 30, 32)}.items():
        hx = sx[side] * 0.1
        foot_x = sx[side] * 0.05 if stance == "double" else (0.0 if stance == "tandem" else hx)
        foot_z = 0.0
        if stance == "tandem":
            foot_z = 0.15 if side == nondominant else -0.15        # non-dominant in back
        if step and step[0] == side:
            foot_x += step[1]
        f = math.radians(flex.get(side, 20 if (stance == "single" and side == dom) else 0))
        a = math.radians(abd.get(side, 0))
        knee_x, knee_y, knee_z = hx + sx[side] * 0.42 * math.sin(a), 0.42 * math.cos(f) * math.cos(a), -0.42 * math.sin(f)
        put(knee, knee_x, knee_y, knee_z)
        raised = stance == "single" and side == dom and not touch
        base_y = 0.80 - (0.25 if raised else 0.0)
        put(ank, foot_x, base_y, foot_z)
        put(heel, foot_x, base_y + 0.05 - heel_lift.get(side, 0.0), foot_z + 0.03)
        put(toe, foot_x, base_y + 0.03 - toe_lift.get(side, 0.0), foot_z - 0.12)

    def rot_x(p, deg, cy, cz):           # tip toward the camera (-z) about a horizontal axis
        a = math.radians(deg)
        y, z = p[1] - cy, p[2] - cz
        return [p[0], cy + y * math.cos(a) + z * math.sin(a), cz - y * math.sin(a) + z * math.cos(a)]

    def rot_z(p, deg, cx, cy):           # bend sideways
        a = math.radians(deg)
        x, y = p[0] - cx, p[1] - cy
        return [cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a), p[2]]

    upper = [i for i in range(23)]                  # everything above the hips
    hip_y = fall
    if lean:
        for i in upper:
            world[i] = rot_x(world[i], -lean, hip_y, 0.0)
    if side_bend:
        for i in upper:
            world[i] = rot_z(world[i], side_bend, 0.0, hip_y)
    if sway:
        for i in range(33):
            world[i] = rot_x(world[i], -sway, 0.85 + fall, 0.0)

    lms = [{"x": 0.5 + scale * p[0] * PX / W, "y": 0.45 + scale * p[1] * PX / H, "z": p[2],
            "visibility": 0.95 if visible else 0.1} for p in world]
    if z_noise:
        rng = rng or np.random.default_rng(0)
        world = [[p[0], p[1], p[2] + rng.normal(0, z_noise)] for p in world]
    wl = [{"x": p[0], "y": p[1], "z": p[2], "visibility": 0.95} for p in world]
    return lms, wl


def pose(**kw):
    lms, wl = body(**kw)
    return {"detected": True, "landmarks": lms, "world_landmarks": wl}


def session(**cfg):
    return BessSession(BessConfig(**{"countdown_s": 2.0, "baseline_s": 1.0, **cfg}))


def run(s, frames, t0=0.0, eyes=None):
    """frames: list of pose dicts. eyes: list of bools/None (default closed)."""
    out = []
    for i, p in enumerate(frames):
        e = eyes[i] if eyes is not None else False
        out.append(s.update(t0 + i * DT, p, W, H, e))
    return out, t0 + len(frames) * DT


def secs(x):
    return int(round(x * FPS))


def do_test(s, stance, segments, nondominant="left", eyes=None, t0=0.0):
    """Start a stance, hold the clean position through the countdown, then play
    `segments` = [(seconds, pose kwargs)] for the 20 s trial. Returns the done event."""
    s.start(stance, nondominant, t=t0)
    clean = pose(stance=stance, nondominant=nondominant)
    res, t = run(s, [clean] * secs(2.0) + [clean], t0)
    assert res[-1]["phase"] == "running", res[-1]
    frames = []
    for dur, kw in segments:
        frames += [pose(stance=stance, nondominant=nondominant, **kw)] * secs(dur)
    remaining = secs(20.5) - len(frames)
    frames += [clean] * max(remaining, 0)
    res, t = run(s, frames, t, eyes)
    done = [e for r in res for e in r["events"] if e["kind"] == "bess_done"]
    assert len(done) == 1
    return done[0], res


# ------------------------------- geometry ----------------------------------- #

def test_hip_angles():
    _, wl = body(stance="single", nondominant="left")
    fl, ab = hip_angles(wl, "right")         # dominant leg raised at 20 deg flexion
    assert fl == pytest.approx(20, abs=0.5) and ab == pytest.approx(0, abs=0.5)
    _, wl = body(abd={"left": 35})
    fl, ab = hip_angles(wl, "left")
    assert ab == pytest.approx(35, abs=0.5) and fl == pytest.approx(0, abs=0.5)
    _, wl = body()
    assert max(hip_angles(wl, "right")) < 1


# ------------------------------- error rules -------------------------------- #

def test_clean_trial_scores_zero():
    for stance in ("double", "tandem", "single"):
        s = session()
        done, _ = do_test(s, stance, [])
        assert done["errors"] == 0, (stance, done["log"])
        assert done["warnings"] == []


def test_hands_off_hips_counts_each_distinct_event():
    s = session()
    done, _ = do_test(s, "double", [(3, {}), (1, {"hands": "left"}), (3, {}), (1, {"hands": "down"})])
    assert done["by_type"]["hands_off_hips"] == 2 and done["errors"] == 2


def test_brief_blip_is_ignored():
    s = session()
    done, _ = do_test(s, "double", [(3, {}), (0.1, {"hands": "down"})])
    assert done["errors"] == 0


def test_out_of_position_over_5s_adds_a_point():
    s = session()
    done, _ = do_test(s, "double", [(2, {}), (7, {"hands": "down"})])
    assert done["by_type"]["hands_off_hips"] == 1
    assert done["by_type"]["out_of_position"] == 1
    assert done["errors"] == 2


def test_eyes_open_auto_and_manual():
    s = session(track_eyes=True)
    eyes = [False] * secs(4) + [True] * secs(1) + [False] * secs(17)
    done, _ = do_test(s, "double", [], eyes=eyes)
    assert done["by_type"]["eyes_open"] == 1
    # manual mark from the app
    s = session(track_eyes=True)
    s.start("double", t=0)
    clean = pose()
    _, t = run(s, [clean] * secs(2) + [clean])
    s.mark("eyes_open")
    res, _ = run(s, [clean] * secs(21), t)
    done = [e for r in res for e in r["events"] if e["kind"] == "bess_done"][0]
    assert done["by_type"]["eyes_open"] == 1


def test_unknown_eyes_dont_count():
    s = session()
    done, _ = do_test(s, "double", [], eyes=[None] * secs(22))
    assert done["errors"] == 0


def test_step_and_simultaneous_errors_count_once():
    s = session()
    # stepping out: foot moves AND the heel comes up at the same moment -> one point
    done, _ = do_test(s, "tandem", [(3, {}), (1, {"step": ("left", 0.2), "heel_lift": {"left": 0.06}})])
    assert done["errors"] == 1
    reasons = [e.get("not_counted_reason") for e in done["log"] if not e["counted"]]
    assert reasons == ["simultaneous"]


def test_hip_abduction_over_30():
    s = session()
    done, _ = do_test(s, "single", [(4, {}), (1, {"abd": {"right": 40}})])
    assert done["by_type"]["hip_angle"] == 1
    s = session()
    done, _ = do_test(s, "single", [(4, {}), (1, {"flex": {"right": 25}})])   # under 30
    assert done["errors"] == 0


def test_heel_or_forefoot_lift():
    s = session()
    done, _ = do_test(s, "double", [(4, {}), (1, {"toe_lift": {"right": 0.06}})])
    assert done["by_type"]["foot_lift"] == 1
    s = session()
    done, _ = do_test(s, "single", [(4, {}), (1, {"heel_lift": {"left": 0.06}})])   # stance foot
    assert done["by_type"]["foot_lift"] == 1


def test_single_leg_raised_foot_touchdown_is_a_step():
    s = session()
    done, _ = do_test(s, "single", [(5, {}), (1, {"touch": True})])
    assert done["by_type"]["step_stumble_fall"] >= 1


def test_fall():
    s = session()
    done, _ = do_test(s, "double", [(5, {}), (2, {"fall": 0.3})])
    assert done["by_type"]["step_stumble_fall"] == 1


def test_max_10_errors():
    s = session()
    segs = [(0.6, {"hands": "down"}), (0.6, {})] * 16
    done, _ = do_test(s, "double", segs)
    assert done["errors"] == 10
    assert any(e.get("not_counted_reason") == "max_reached" for e in done["log"])


def test_hidden_body_holds_state():
    s = session()
    done, _ = do_test(s, "double", [(3, {}), (3, {"visible": False})])
    assert done["errors"] == 0


# ------------------------------- session ------------------------------------ #

def test_full_session_breakdown():
    s = session()
    d1, _ = do_test(s, "double", [(3, {}), (1, {"hands": "left"})])
    d2, _ = do_test(s, "tandem", [(3, {}), (1, {"hands": "left"}), (3, {}), (1, {"step": ("right", 0.2)})])
    d3, _ = do_test(s, "single", [(3, {}), (1, {"abd": {"right": 40}}), (3, {}), (1, {"touch": True}),
                                  (3, {}), (1, {"hands": "down"})])
    summ = d3["session"]
    assert summ["scores"] == {"double": 1, "tandem": 2, "single": 3}
    assert summ["total"] == 6 and summ["complete"]
    # re-running a stance replaces its score
    d1b, _ = do_test(s, "double", [])
    assert d1b["session"]["scores"]["double"] == 0 and d1b["session"]["total"] == 5


def test_countdown_waits_then_fails_without_view():
    s = session(max_wait_s=2.0)
    s.start("double", t=0)
    res, _ = run(s, [{"detected": False}] * secs(5))
    failed = [e for r in res for e in r["events"] if e["kind"] == "bess_failed"]
    assert len(failed) == 1 and res[-1]["phase"] == "idle"


def test_countdown_waits_for_body_then_starts():
    s = session(max_wait_s=5.0)
    s.start("double", t=0)
    res, t = run(s, [{"detected": False}] * secs(3))
    assert res[-1]["phase"] == "countdown" and res[-1]["waiting_for_view"]
    res, _ = run(s, [pose()] * secs(1.5), t)
    assert res[-1]["phase"] == "running"


def test_setup_warnings():
    s = session()
    s.start("single", "left", t=0)
    # raised the wrong (non-dominant) foot: i.e. this is a single stance on the right leg
    res, _ = run(s, [pose(stance="single", nondominant="right")] * secs(2.2))
    running = [e for r in res for e in r["events"] if e["kind"] == "bess_running"][0]
    assert any("dominant" in w for w in running["warnings"])

    s = session()
    s.start("tandem", "left", t=0)
    res, _ = run(s, [pose(stance="tandem", nondominant="right")] * secs(2.2))
    running = [e for r in res for e in r["events"] if e["kind"] == "bess_running"][0]
    assert any("in back" in w for w in running["warnings"])


def test_commands():
    s = session()
    assert s.handle_command({"type": "bess_start", "stance": "sideways"})[0]["type"] == "error"
    assert s.handle_command({"type": "bess_mark", "error": "eyes_open"})[0]["type"] == "error"
    r = s.handle_command({"type": "bess_start", "stance": "tandem", "nondominant": "right"})[0]
    assert r["type"] == "ack" and s.phase == "countdown" and s.nondominant == "right"
    assert s.handle_command({"type": "bess_cancel"})[0]["type"] == "ack" and s.phase == "idle"
    assert s.handle_command({"type": "nope"})[0]["type"] == "error"
    r = s.handle_command({"type": "bess_status"})[0]
    assert r["scores"] == {"double": None, "tandem": None, "single": None} and r["total"] == 0


# ------------------------------- server ------------------------------------- #

class ScriptedBody:
    def __init__(self):
        self.kw = {}

    def process_bgr(self, frame):
        lms, wl = body(**self.kw)
        spine = {**spine_angles_2d(lms, W, H), **spine_angles_3d(wl),
                 "trunk_visibility": 0.95, "reliable": True}
        return PoseResult(True, lms, wl, spine, 1.0)

    def close(self):
        pass


class NoEyes:
    def is_open(self, frame, lms):
        return False

    def close(self):
        pass


def test_server_bess_flow():
    clock = {"t": 0.0}
    scripted = ScriptedBody()

    class Clocked(PosePipeline):
        def process_bgr(self, frame, t=None):
            clock["t"] += DT
            return super().process_bgr(frame, clock["t"])

    ok, buf = cv2.imencode(".jpg", np.zeros((64, 36, 3), np.uint8))
    jpg = buf.tobytes()

    async def go():
        handler = ws_server.make_handler(
            lambda: Clocked(scripted, bess=BessConfig(countdown_s=2.0), eye_closure=NoEyes()),
            "pose", pose_events)
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "bess_start", "stance": "double",
                                          "nondominant": "left"}))
                ack = json.loads(await asyncio.wait_for(ws.recv(), 5))
                for i in range(secs(23)):
                    scripted.kw = {"hands": "down"} if secs(8) <= i < secs(9) else {}
                    await ws.send(jpg)
                    while True:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        msgs.append(m)
                        if m["type"] == "pose":
                            break
                try:
                    while True:
                        msgs.append(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
                except asyncio.TimeoutError:
                    pass
            return ack, msgs

    ack, msgs = asyncio.run(go())
    assert ack["type"] == "ack" and ack["command"] == "bess_start"
    errs = [m for m in msgs if m.get("kind") == "bess_error"]
    results = [m for m in msgs if m["type"] == "result"]
    assert len(errs) == 1 and errs[0]["error"] == "hands_off_hips" and errs[0]["counted"]
    assert len(results) == 1 and results[0]["kind"] == "bess_done" and results[0]["errors"] == 1
    assert results[0]["session"]["scores"]["double"] == 1
    running = [m for m in msgs if m["type"] == "pose" and m["bess"]["phase"] == "running"]
    assert running and "time_left" in running[0]["bess"]


# ---------------------- eye tracking off (default) ------------------------- #

def test_eyes_ignored_by_default():
    s = session()
    eyes = [True] * secs(23)                     # eyes open the whole time
    done, res = do_test(s, "double", [], eyes=eyes)
    assert done["errors"] == 0
    assert "eyes_open" not in done["by_type"]
    running = [r for r in res if r["phase"] == "running"]
    assert running and all("eyes" not in r for r in running)
    s.start("double", t=100.0)
    run(s, [pose()] * secs(2.2), 100.0)
    with pytest.raises(ValueError, match="eye tracking is turned off"):
        s.mark("eyes_open")


def test_pipeline_eyes_off_loads_no_face_model():
    p = PosePipeline(ScriptedBody(), bess=BessConfig())
    assert p.eye_closure is None and not p.bess.cfg.track_eyes
    with pytest.raises(ValueError):
        PosePipeline(ScriptedBody(), bess=BessConfig(), bess_eyes="sometimes")


# ------------- single-leg balance tracker paused for double/tandem ---------- #

def _pipeline_with_balance():
    from balance import BalanceConfig
    scripted = ScriptedBody()
    return scripted, PosePipeline(scripted, balance=BalanceConfig(), bess=BessConfig(countdown_s=2.0))


def _frames(p, scripted, n, t0, **kw):
    scripted.kw = kw
    frame = np.zeros((H, W, 3), np.uint8)
    out = []
    for i in range(n):
        out.append(p.process_bgr(frame, t0 + i * DT))
    return out, t0 + n * DT


def test_balance_tracker_paused_during_double_and_tandem():
    for stance in ("double", "tandem"):
        scripted, p = _pipeline_with_balance()
        p.handle_command({"type": "bess_start", "stance": stance, "nondominant": "left"})
        # during the test, even a clearly raised foot doesn't start/stop single-leg tracking
        res, t = _frames(p, scripted, secs(22.5), 0.0, stance=stance, heel_lift={"right": 0.3})
        during = [r for r in res if r["bess"]["phase"] in ("countdown", "running")]
        assert during and all(r["balance"]["state"] == "paused" for r in during)
        assert all(r["balance"]["paused_for"] == stance for r in during)
        assert not any(r["balance"]["event"] for r in res)
        # after the test it's back on
        assert res[-1]["bess"]["phase"] == "idle"
        res, _ = _frames(p, scripted, 3, t)
        assert res[-1]["balance"]["state"] != "paused"


def test_balance_tracker_runs_during_single_leg():
    scripted, p = _pipeline_with_balance()
    p.handle_command({"type": "bess_start", "stance": "single", "nondominant": "left"})
    res, _ = _frames(p, scripted, secs(6), 0.0, stance="single")
    running = [r for r in res if r["bess"]["phase"] == "running"]
    assert running and all(r["balance"]["state"] != "paused" for r in running)
    assert any(r["balance"]["event"] == "balance_start" for r in res)


# --------------------- hip angle: 2D-based geometry ------------------------- #

def _hip_session_running(stance="double", **kw):
    s = session()
    s.start(stance, "left", t=0)
    _, t = run(s, [pose(stance=stance)] * secs(2) + [pose(stance=stance)])
    assert s.phase == "running"
    return s, t


def _max_hip(s, t, frames):
    res, t = run(s, frames, t)
    return max(max(r.get("hip_angles", {}).values(), default=0) for r in res), res, t


def test_forward_bend_at_waist_detected_in_feet_together():
    for deg, expect in ((15, False), (40, True)):
        s, t = _hip_session_running()
        peak, res, _ = _max_hip(s, t, [pose(lean=deg)] * secs(2))
        assert (peak > 30) == expect, (deg, peak)
        assert abs(peak - deg) < 6, (deg, peak)          # reads close to the true angle


def test_side_bend_detected():
    s, t = _hip_session_running()
    peak, _, _ = _max_hip(s, t, [pose(side_bend=38)] * secs(2))
    assert peak > 30


def test_whole_body_sway_is_not_hip_flexion():
    s, t = _hip_session_running()
    peak, _, _ = _max_hip(s, t, [pose(sway=10)] * secs(2))
    assert peak < 8


def test_stepping_back_from_camera_is_not_hip_flexion():
    s, t = _hip_session_running()
    peak, _, _ = _max_hip(s, t, [pose(scale=0.9)] * secs(2))
    assert peak < 8


def test_depth_noise_no_false_hip_errors_and_bend_still_caught():
    rng = np.random.default_rng(1)
    noisy = lambda **kw: (lambda lw: {"detected": True, "landmarks": lw[0], "world_landmarks": lw[1]})(
        body(z_noise=0.06, rng=rng, **kw))
    s = session()
    s.start("double", t=0)
    _, t = run(s, [noisy()] * secs(2) + [noisy()])
    res, t = run(s, [noisy()] * secs(8), t)
    assert not any(e["kind"] == "bess_error" for r in res for e in r["events"])
    assert max(max(r["hip_angles"].values()) for r in res if r.get("hip_angles")) < 12
    res, _ = run(s, [noisy(lean=40)] * secs(2), t)
    errs = [e for r in res for e in r["events"] if e["kind"] == "bess_error"]
    assert [e["error"] for e in errs] == ["hip_angle"]


def test_single_leg_raised_thigh_uses_absolute_angle():
    s, t = _hip_session_running("single")
    res, _ = run(s, [pose(stance="single")] * secs(1), t)
    assert 15 < res[-1]["hip_angles"]["right"] < 25          # starts at ~20 deg
    s, t = _hip_session_running("single")
    peak, _, _ = _max_hip(s, t, [pose(stance="single", flex={"right": 40})] * secs(1))
    assert peak > 30
