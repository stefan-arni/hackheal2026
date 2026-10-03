"""Duck detection + quack tests on synthetic poses (no MediaPipe model needed).

    pytest -q
"""

import asyncio
import json

import cv2
import numpy as np
import pytest
import websockets

import ws_server
from duck import DuckConfig, DuckDetector
from pose_analyzer import PoseResult, spine_angles_2d, spine_angles_3d
from pose_pipeline import PosePipeline, duck_events, pose_events

W, H = 640, 960
PX_PER_M = 400


def make_body(head_drop=0.0, feet_visible=True, sw_m=0.4):
    """Standing person, 1.7 m tall. head_drop (m) lowers nose + shoulders (a duck)."""
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    world = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.95} for _ in range(33)]

    def put(i, x, y_world):           # world: origin at hips, y down
        world[i] = {"x": x, "y": y_world, "z": 0.0, "visibility": 0.95}
        lms[i] = {"x": 0.5 + x * PX_PER_M / W, "y": 0.45 + y_world * PX_PER_M / H, "z": 0.0,
                  "visibility": 0.95}

    put(0, 0.0, -0.75 + head_drop)                     # nose
    put(11, sw_m / 2, -0.5 + head_drop)                # shoulders
    put(12, -sw_m / 2, -0.5 + head_drop)
    put(23, 0.1, 0.0)
    put(24, -0.1, 0.0)                                 # hips
    for i, y in ((27, 0.80), (28, 0.80), (29, 0.85), (30, 0.85), (31, 0.83), (32, 0.83)):
        put(i, 0.1 if i % 2 else -0.1, y)
        if not feet_visible:
            lms[i]["visibility"] = 0.1
    return lms, world


def run(det, drops, feet_visible=True, t0=0.0, dt=1 / 15):
    out = []
    for i, d in enumerate(drops):
        lms, world = make_body(d, feet_visible)
        mode, val = det.measure(lms, world, W, H)
        out.append(det.update(t0 + i * dt, mode, val))
    return out, t0 + len(drops) * dt


def quacks(res):
    return [r for r in res if r["event"] == "duck"]


# ------------------------------- detector ---------------------------------- #

def test_standing_still_never_quacks():
    res, _ = run(DuckDetector(), [0.0] * 60)
    assert not quacks(res)
    assert res[-1]["mode"] == "world" and res[-1]["drop"] == pytest.approx(0, abs=1e-9)


def test_quick_duck_quacks_once():
    # 1.6 m head height; drop 0.5 m (31%) over ~0.3 s, hold, come back up
    seq = [0.0] * 20 + [0.1, 0.25, 0.4, 0.5] + [0.5] * 15 + [0.0] * 15
    res, _ = run(DuckDetector(), seq)
    q = quacks(res)
    assert len(q) == 1 and q[0]["count"] == 1 and q[0]["mode"] == "world"
    assert not res[-1]["ducking"]


def test_small_bob_does_not_quack():
    res, _ = run(DuckDetector(), [0.0] * 20 + [0.15] * 10 + [0.0] * 10)    # ~9% drop
    assert not quacks(res)


def test_slow_sink_does_not_quack():
    # sinking 0.5 m over 10 s: never 20% below the last 2 s
    seq = [0.0] * 15 + list(np.linspace(0, 0.5, 150)) + [0.5] * 15
    res, _ = run(DuckDetector(), seq)
    assert not quacks(res)


def test_two_ducks_two_quacks():
    one = [0.0] * 15 + [0.5] * 10
    res, _ = run(DuckDetector(), one + one + [0.0] * 15)
    assert len(quacks(res)) == 2 and res[-1]["count"] == 2


def test_cooldown_blocks_rapid_repeats():
    det = DuckDetector(DuckConfig(cooldown_s=2.0))
    seq = [0.0] * 15 + [0.5] * 4 + [0.0] * 4 + [0.5] * 4 + [0.0] * 10   # second duck 0.5 s later
    res, _ = run(det, seq)
    assert len(quacks(res)) == 1


def test_single_glitch_frame_does_not_quack():
    res, _ = run(DuckDetector(), [0.0] * 20 + [0.6] + [0.0] * 20)
    assert not quacks(res)


def test_upper_body_only_uses_image_mode():
    seq = [0.0] * 20 + [0.45] * 10 + [0.0] * 10       # 0.45 m = 1.1 shoulder-widths
    res, _ = run(DuckDetector(), seq, feet_visible=False)
    assert res[0]["mode"] == "image"
    assert len(quacks(res)) == 1


def test_head_not_visible():
    det = DuckDetector()
    lms, world = make_body()
    lms[0]["visibility"] = 0.1
    mode, val = det.measure(lms, world, W, H)
    r = det.update(0.0, mode, val)
    assert r["skip_reason"] == "head_not_visible" and r["event"] is None


def test_reset():
    det = DuckDetector()
    run(det, [0.0] * 15 + [0.5] * 5)
    assert det.count == 1
    det.reset()
    assert det.count == 0 and not det.ducking


# ------------------------------- quack sound -------------------------------- #

def test_quack_synth_and_wav(tmp_path):
    from quack import RATE, ensure_quack_wav, synth_quack
    y = synth_quack()
    assert 0.2 < len(y) / RATE < 0.4
    assert np.isfinite(y).all() and 0.5 < np.abs(y).max() <= 0.81
    import wave
    p = ensure_quack_wav(tmp_path / "q.wav")
    with wave.open(str(p)) as wf:
        assert wf.getframerate() == RATE and wf.getnframes() == len(y)


# ------------------------------- server ------------------------------------- #

class ScriptedBody:
    def __init__(self, drops):
        self.drops, self.i = drops, 0

    def process_bgr(self, frame):
        lms, world = make_body(self.drops[min(self.i, len(self.drops) - 1)])
        self.i += 1
        spine = {**spine_angles_2d(lms, W, H), **spine_angles_3d(world),
                 "trunk_visibility": 0.95, "reliable": True}
        return PoseResult(True, lms, world, spine, 1.0)

    def close(self):
        pass


def test_server_sends_duck_event():
    drops = [0.0] * 20 + [0.5] * 10 + [0.0] * 10
    clock = {"t": 0.0}

    class Clocked(PosePipeline):
        def process_bgr(self, frame, t=None):
            clock["t"] += 1 / 15
            return super().process_bgr(frame, clock["t"])

    ok, buf = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
    jpg = buf.tobytes()

    async def go():
        from balance import BalanceConfig
        handler = ws_server.make_handler(
            lambda: Clocked(ScriptedBody(drops), BalanceConfig(), DuckConfig()), "pose", pose_events)
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for _ in drops:
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
            return msgs

    msgs = asyncio.run(go())
    ducks = [m for m in msgs if m.get("kind") == "duck"]
    assert len(ducks) == 1 and ducks[0]["type"] == "event" and ducks[0]["count"] == 1
    poses = [m for m in msgs if m["type"] == "pose"]
    assert all("duck" in m and "balance" in m for m in poses)
    # a duck must not be mistaken for a foot lifting
    assert not any(m.get("kind") in ("balance_started", "foot_touchdown") for m in msgs)


def test_duck_events_shape():
    assert duck_events({"duck": {"event": None}}) == []
    msg = duck_events({"duck": {"event": "duck", "count": 2, "drop": 0.3, "mode": "world"}}, 7)[0]
    assert msg == {"type": "event", "kind": "duck", "frame_id": 7, "count": 2, "drop": 0.3,
                   "mode": "world"}


def test_bundled_quack_is_used(monkeypatch):
    import subprocess
    import wave

    import quack
    assert quack.QUACK_PATH.exists(), "sounds/quack.wav should ship with the project"
    with wave.open(str(quack.QUACK_PATH)) as wf:
        dur = wf.getnframes() / wf.getframerate()
        assert wf.getsampwidth() == 2 and 0.1 < dur < 0.5     # short, plays instantly
    played = []
    monkeypatch.setattr(quack.sys, "platform", "linux")
    monkeypatch.setattr(quack.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: played.append(cmd))
    quack.play_quack()
    assert played and played[0][-1] == str(quack.QUACK_PATH)
