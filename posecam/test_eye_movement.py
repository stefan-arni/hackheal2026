"""Eye-movement tests: saccade speed and smoothness on synthetic eye traces.

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
from eye_movement import EyeMovementTest, GazeTracker, MovementConfig, iris_to_degrees
from eye_server import eye_events

CFG = MovementConfig()
K = CFG.eye_radius_mm / CFG.eye_width_mm      # degrees -> eye-widths via sin


def deg_to_h(deg):
    return 0.5 + math.sin(math.radians(deg)) * K


def main_sequence(amp):
    """Real saccades: duration grows with size (~2.2 ms/deg + 21 ms)."""
    return (2.2 * abs(amp) + 21) / 1000


def min_jerk(t, t0, a0, a1):
    d = main_sequence(a1 - a0)
    s = np.clip((t - t0) / d, 0, 1)
    return a0 + (a1 - a0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def true_peak(amp):
    return 1.875 * abs(amp) / main_sequence(amp)


def trace(fn, seconds, fps, noise=0.003, seed=0, gaps=()):
    """Samples (t, eyes) from a gaze-angle function fn(t) -> (x_deg, y_deg)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        if any(a <= t < b for a, b in gaps):
            out.append((t, None))
            continue
        x, y = fn(t)
        e = {"h": deg_to_h(x) + rng.normal(0, noise), "v": math.sin(math.radians(y)) * K + rng.normal(0, noise)}
        out.append((t, {"right": e, "left": dict(e)}))
    return out


def feed(tracker, samples):
    return [tracker.update(t, e) for t, e in samples]


def saccades_in(results):
    return [r["saccade"] for r in results if r["saccade"]]


# ------------------------------- gaze + speed -------------------------------- #

def test_degrees_conversion():
    assert iris_to_degrees(0.0, CFG) == 0
    assert iris_to_degrees(0.1, CFG) == pytest.approx(14.48, abs=0.05)
    assert iris_to_degrees(-0.1, CFG) == pytest.approx(-14.48, abs=0.05)


@pytest.mark.parametrize("fps,min_pct", [(60, 75), (120, 85), (240, 90)])
def test_single_saccade_size_duration_speed(fps, min_pct):
    res = feed(GazeTracker(), trace(lambda t: (min_jerk(t, 1.0, -10, 10), 0), 2.0, fps))
    s = saccades_in(res)
    assert len(s) == 1
    s = s[0]
    assert s["amplitude_deg"] == pytest.approx(20, abs=2)
    assert s["direction"] == "right"
    assert abs(s["duration_ms"] - main_sequence(20) * 1000) < 2000 / fps + 10
    pct = 100 * s["peak_velocity_dps"] / true_peak(20)
    assert min_pct <= pct <= 115, pct
    assert s["mean_velocity_dps"] < s["peak_velocity_dps"]


def test_30fps_reads_speed_low_but_finds_saccade():
    s = saccades_in(feed(GazeTracker(), trace(lambda t: (min_jerk(t, 1.0, 10, -10), 0), 2.0, 30)))
    assert len(s) == 1 and s[0]["direction"] == "left"
    assert s[0]["peak_velocity_dps"] < true_peak(20)


@pytest.mark.parametrize("fps", [30, 60, 120, 240])
def test_fixation_noise_makes_no_saccades(fps):
    res = feed(GazeTracker(), trace(lambda t: (0, 0), 5.0, fps, noise=0.004, seed=fps))
    assert saccades_in(res) == []


def test_vertical_saccade():
    s = saccades_in(feed(GazeTracker(), trace(lambda t: (0, min_jerk(t, 1.0, 0, 12)), 2.0, 120)))
    assert len(s) == 1 and s[0]["direction"] == "down"


def test_blink_gap_does_not_create_a_saccade():
    # eyes move slowly to the side during a blink: the jump across the gap isn't a saccade
    res = feed(GazeTracker(), trace(lambda t: (0 if t < 1.0 else 15, 0), 2.0, 60, gaps=[(0.9, 1.1)]))
    assert saccades_in(res) == []


def test_small_jumps_ignored():
    s = saccades_in(feed(GazeTracker(), trace(lambda t: (min_jerk(t, 1.0, 0, 1.2), 0), 2.0, 120)))
    assert s == []


# ------------------------------- tests --------------------------------------- #

def run_test(mode, fn, seconds=10.0, fps=120, yaw=lambda t: 0.0, **kw):
    g, test = GazeTracker(), EyeMovementTest()
    test.start(mode, seconds)
    done = None
    for t, e in trace(fn, seconds + 0.2, fps, **kw):
        r = test.update(t, g.update(t, e) if e else g.update(t, None), yaw(t))
        for ev in r["events"]:
            if ev["kind"] == "eye_test_done":
                done = ev
    return done


def saccade_task(corrections=()):
    """Back and forth between -10 and +10 deg every 0.8 s; jumps listed in
    `corrections` undershoot to 70% and then correct 150 ms later."""
    jumps = [(0.5 + 0.8 * i, (-10, 10) if i % 2 == 0 else (10, -10)) for i in range(11)]

    def fn(t):
        x = -10.0
        for i, (t0, (a, b)) in enumerate(jumps):
            if t < t0:
                break
            if i in corrections:
                mid = a + 0.7 * (b - a)
                x = min_jerk(t, t0, a, mid) if t < t0 + 0.15 else min_jerk(t, t0 + 0.15, mid, b)
            else:
                x = min_jerk(t, t0, a, b)
        return x, 0.0
    return fn


def test_saccade_test_clean():
    res = run_test("saccades", saccade_task())
    sp, sm = res["speed"], res["smoothness"]
    assert sp["count"] == 11
    assert sp["amplitude_deg"]["median"] == pytest.approx(20, abs=2)
    assert 0.85 * true_peak(20) < sp["peak_velocity_dps"]["median"] < 1.15 * true_peak(20)
    assert sm["primary_saccades"] == 11 and sm["corrective_saccades"] == 0 and sm["score"] == 100
    assert res["quality"]["warnings"] == []


def test_saccade_test_counts_corrective_saccades():
    res = run_test("saccades", saccade_task(corrections=(2, 5, 8)))
    sm = res["smoothness"]
    assert sm["primary_saccades"] == 11 and sm["corrective_saccades"] == 3
    assert sm["score"] == round(100 * 8 / 11)


def test_pursuit_smooth():
    # follow a target moving +-10 deg at 0.4 Hz (peak ~25 deg/s): no catch-up saccades
    res = run_test("pursuit", lambda t: (10 * math.sin(2 * math.pi * 0.4 * t), 0))
    sm = res["smoothness"]
    assert sm["catch_up_saccades"] == 0 and sm["score"] >= 95
    assert 10 < sm["pursuit_speed_dps"]["median"] < 25


def test_pursuit_with_catch_up_saccades_scores_lower():
    # the eye lags the target, then jumps to catch up every 0.5 s ("cogwheel" pursuit)
    def fn(t):
        target = 20 * math.sin(2 * math.pi * 0.25 * t)
        k = math.floor(t / 0.5) * 0.5
        lagged = 20 * math.sin(2 * math.pi * 0.25 * (k - 0.15))
        stuck = 20 * math.sin(2 * math.pi * 0.25 * k)
        return (min_jerk(t, k, lagged, stuck) if t - k < 0.08 else stuck + 0.2 * (target - stuck), 0)
    smooth = run_test("pursuit", lambda t: (20 * math.sin(2 * math.pi * 0.25 * t), 0))
    jerky = run_test("pursuit", fn)
    assert jerky["smoothness"]["catch_up_saccades"] >= 8
    assert jerky["smoothness"]["score"] < smooth["smoothness"]["score"] - 30


def test_low_fps_and_head_turn_warnings():
    res = run_test("saccades", saccade_task(), fps=30, yaw=lambda t: 0.2 * (t > 5))
    w = " ".join(res["quality"]["warnings"])
    assert "frames/s" in w and "head turned" in w


def test_commands():
    test = EyeMovementTest()
    assert test.handle_command({"type": "eye_test_start", "mode": "hop"})[0]["type"] == "error"
    assert test.handle_command({"type": "eye_test_start", "mode": "saccades", "duration": 500})[0]["type"] == "error"
    r = test.handle_command({"type": "eye_test_start", "mode": "pursuit", "duration": 5})[0]
    assert r["type"] == "ack" and r["running"] and r["mode"] == "pursuit"
    r = test.handle_command({"type": "eye_test_cancel"})[0]
    assert not r["running"]
    assert test.handle_command({"type": "nope"})[0]["type"] == "error"


# ----------------------- through the eye analyzer + server ------------------- #

def _face_analyzer():
    from test_eyes import analyzer
    return analyzer()


def test_analyzer_runs_saccade_test_on_face_landmarks():
    from test_eyes import make_face
    a = _face_analyzer()
    a.handle_command({"type": "eye_test_start", "mode": "saccades", "duration": 3})
    fn = saccade_task()
    done = None
    for i in range(int(3.2 * 120)):
        t = i / 120
        h = deg_to_h(fn(t)[0])
        r = a.analyze_points(make_face(r_h=h, l_h=h), t)
        for ev in r["eye_test"]["events"]:
            if ev["kind"] == "eye_test_done":
                done = ev
    assert done and done["speed"]["count"] == 4
    assert done["speed"]["amplitude_deg"]["median"] == pytest.approx(20, abs=2)


class TimedEyes:
    """Server-side analyzer that uses the frame's capture timestamp (meta)."""

    def __init__(self):
        from test_eyes import make_face
        self.a, self.make_face = _face_analyzer(), make_face
        self.fn = saccade_task()

    def process_bgr(self, frame, t=None, meta=None):
        t = meta["timestamp_ms"] / 1000
        h = deg_to_h(self.fn(t - 1000.0)[0])
        return self.a.analyze_points(self.make_face(r_h=h, l_h=h), t)

    def handle_command(self, msg):
        return self.a.handle_command(msg)

    def close(self):
        pass


def test_server_uses_capture_timestamps_and_sends_results():
    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    b64 = __import__("base64").b64encode(buf.tobytes()).decode()

    async def go():
        async with websockets.serve(ws_server.make_handler(TimedEyes, "eyes", eye_events),
                                    "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "eye_test_start", "mode": "saccades", "duration": 3}))
                msgs.append(json.loads(await asyncio.wait_for(ws.recv(), 5)))
                # frames captured at 120 fps (timestamps), sent as fast as the server answers
                for i in range(int(3.3 * 120)):
                    ts = 1_000_000 + i * 1000 / 120
                    await ws.send(json.dumps({"type": "frame", "image": b64, "timestamp_ms": ts}))
                    while True:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        msgs.append(m)
                        if m["type"] == "eyes":
                            break
                try:
                    while True:
                        msgs.append(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
                except asyncio.TimeoutError:
                    pass
            return msgs

    msgs = asyncio.run(go())
    assert msgs[0]["type"] == "ack" and msgs[0]["running"]
    sacc = [m for m in msgs if m.get("kind") == "saccade"]
    done = [m for m in msgs if m["type"] == "result" and m["kind"] == "eye_test_done"]
    assert len(sacc) == 4 and len(done) == 1
    # speeds come from the 120 fps capture clock, not the server's processing speed
    assert done[0]["quality"]["effective_fps"] == pytest.approx(120, abs=1)
    assert done[0]["speed"]["peak_velocity_dps"]["median"] > 0.85 * true_peak(20)
