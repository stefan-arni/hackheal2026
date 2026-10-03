"""Single-leg balance tests on synthetic poses (no MediaPipe model needed).

    pytest -q
"""

import asyncio
import json

import cv2
import numpy as np
import pytest
import websockets

import ws_server
from balance import BalanceConfig, BalanceMonitor, balance_events, foot_heights
from pose_pipeline import PosePipeline
from pose_analyzer import PoseResult, spine_angles_2d, spine_angles_3d

W, H = 640, 960
# image: pixels-per-meter so world and image agree for an upright person
PX_PER_M = 400


def make_pose(left_lift=0.0, right_lift=0.0, foot_vis=0.95, left_back=0.0):
    """Standing person. Lifts in meters. `left_back` moves the left foot further
    from the camera, which raises it in the image (perspective) but not in 3D."""
    lms = [{"x": 0.5, "y": 0.3, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    world = [{"x": 0.0, "y": -0.5, "z": 0.0, "visibility": 0.95} for _ in range(33)]
    # world: origin at hips, y down, meters. ankle 0.80, heel 0.85, toe 0.83
    for side, xoff, idx, lift in (("left", 0.1, (23, 27, 29, 31), left_lift),
                                  ("right", -0.1, (24, 28, 30, 32), right_lift)):
        hip, ankle, heel, toe = idx
        for i, y in ((hip, 0.0), (ankle, 0.80 - lift), (heel, 0.85 - lift), (toe, 0.83 - lift)):
            world[i] = {"x": xoff, "y": y, "z": 0.0, "visibility": 0.95}
            img_y = 0.4 + y * PX_PER_M / H
            if side == "left" and i != hip:
                img_y -= left_back * PX_PER_M / H
            lms[i] = {"x": 0.5 + xoff * PX_PER_M / W, "y": img_y, "z": 0.0,
                      "visibility": foot_vis if i != hip else 0.95}
    return lms, world


def heights(**kw):
    lms, world = make_pose(**kw)
    return foot_heights(lms, world, W, H)


def feed(mon, seq, t0=0.0, dt=1 / 15):
    """seq: list of foot_heights() dicts (or None) at 15 fps."""
    out = []
    for i, h in enumerate(seq):
        out.append(mon.update(t0 + i * dt, h))
    return out, t0 + len(seq) * dt


# ------------------------------- measurement -------------------------------- #

def test_both_feet_down():
    h = heights()
    assert h["world"]["left"] == pytest.approx(0) and h["world"]["right"] == pytest.approx(0)
    assert h["image"]["left"] == pytest.approx(0) and h["image"]["right"] == pytest.approx(0)


def test_lifted_foot_height_world_and_image():
    h = heights(right_lift=0.15)
    assert h["world"]["right"] == pytest.approx(0.15)
    assert h["world"]["left"] == pytest.approx(0)
    # image: 0.15 m lift / 0.80 m leg (hip->ankle) = 0.1875 leg-lengths
    assert h["image"]["right"] == pytest.approx(0.15 / 0.80, rel=1e-6)


def test_perspective_fools_image_but_not_world():
    h = heights(left_back=0.2)       # foot placed back: looks higher in the image only
    assert h["image"]["left"] > 0.2
    assert h["world"]["left"] == pytest.approx(0)


def test_hidden_feet_return_none():
    assert heights(foot_vis=0.2) is None


# ------------------------------- state machine ------------------------------ #

def test_balance_then_touchdown():
    mon = BalanceMonitor()
    res, t = feed(mon, [heights()] * 10 + [heights(right_lift=0.2)] * 30)
    starts = [r for r in res if r["event"] == "balance_start"]
    assert len(starts) == 1
    assert starts[0]["lifted_foot"] == "right" and starts[0]["standing_foot"] == "left"
    assert res[-1]["state"] == "balancing" and res[-1]["balance_time_s"] > 1.5

    res, _ = feed(mon, [heights()] * 10, t)
    downs = [r for r in res if r["event"] == "touchdown"]
    assert len(downs) == 1
    d = downs[0]
    assert d["touched_foot"] == "right" and d["touch_count"] == 1
    assert d["held_s"] == pytest.approx(2.0, abs=0.15)
    assert res[-1]["state"] == "idle"


def test_short_lift_does_not_start_balance():
    mon = BalanceMonitor()
    res, _ = feed(mon, [heights(left_lift=0.2)] * 3 + [heights()] * 10)   # 0.2 s
    assert not any(r["event"] for r in res)


def test_hovering_low_is_not_a_touchdown():
    mon = BalanceMonitor()
    _, t = feed(mon, [heights(right_lift=0.2)] * 15)
    res, _ = feed(mon, [heights(right_lift=0.05)] * 20, t)   # between touch (0.03) and lift (0.08)
    assert not any(r["event"] == "touchdown" for r in res)
    assert res[-1]["state"] == "balancing"


def test_single_noisy_frame_does_not_trigger():
    mon = BalanceMonitor()
    _, t = feed(mon, [heights(right_lift=0.2)] * 15)
    res, _ = feed(mon, [heights(right_lift=0.2)] * 3 + [heights()] + [heights(right_lift=0.2)] * 5, t)
    assert not any(r["event"] == "touchdown" for r in res)


def test_feet_hidden_mid_balance_keeps_state():
    mon = BalanceMonitor()
    _, t = feed(mon, [heights(right_lift=0.2)] * 15)
    res, t = feed(mon, [None] * 10, t)
    assert all(r["state"] == "balancing" and r["skip_reason"] == "feet_not_visible" for r in res)
    res, _ = feed(mon, [heights()] * 10, t)
    assert any(r["event"] == "touchdown" for r in res)


def test_multiple_attempts_track_best():
    mon = BalanceMonitor()
    _, t = feed(mon, [heights(right_lift=0.2)] * 30 + [heights()] * 10)         # ~2 s
    res, _ = feed(mon, [heights(left_lift=0.2)] * 60 + [heights()] * 10, t)    # ~4 s, other leg
    d = [r for r in res if r["event"] == "touchdown"][0]
    assert d["touched_foot"] == "left" and d["touch_count"] == 2
    assert d["best_hold_s"] == pytest.approx(4.0, abs=0.15)


def test_image_mode():
    mon = BalanceMonitor(BalanceConfig(mode="image"))
    res, t = feed(mon, [heights(right_lift=0.2)] * 15)
    assert res[-1]["state"] == "balancing"
    res, _ = feed(mon, [heights()] * 10, t)
    assert any(r["event"] == "touchdown" for r in res)


def test_reset_and_config_validation():
    mon = BalanceMonitor()
    feed(mon, [heights(right_lift=0.2)] * 15 + [heights()] * 10)
    assert mon.touch_count == 1
    mon.reset()
    assert mon.touch_count == 0 and mon.state == "idle"
    with pytest.raises(ValueError):
        BalanceConfig(lift_threshold=0.03, touch_threshold=0.05)
    with pytest.raises(ValueError):
        BalanceConfig(mode="sideways")


# --------------------------- analyzer + server ------------------------------ #

class ScriptedPose:
    """Stands in for PoseAnalyzer, returning synthetic poses from a script."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def process_bgr(self, frame):
        kw = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        lms, world = make_pose(**kw)
        spine = {**spine_angles_2d(lms, W, H), **spine_angles_3d(world),
                 "trunk_visibility": 0.95, "reliable": True}
        return PoseResult(True, lms, world, spine, 1.0)

    def close(self):
        pass


def test_server_sends_balance_messages():
    script = [{}] * 5 + [{"right_lift": 0.2}] * 20 + [{}] * 12
    clock = {"t": 0.0}

    class Clocked(PosePipeline):     # fake 15 fps clock so timing is deterministic
        def process_bgr(self, frame, t=None):
            clock["t"] += 1 / 15
            return super().process_bgr(frame, clock["t"])

    ok, buf = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))
    jpg = buf.tobytes()

    async def go():
        handler = ws_server.make_handler(lambda: Clocked(ScriptedPose(script), BalanceConfig()), "pose",
                                         balance_events)
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for _ in script:
                    await ws.send(jpg)
                    while True:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        msgs.append(m)
                        if m["type"] == "pose":
                            break
                await ws.send(json.dumps({"type": "recalibrate"}))
                await ws.send(jpg)
                while True:
                    last = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    if last["type"] == "pose":
                        break
                try:
                    while True:
                        msgs.append(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
                except asyncio.TimeoutError:
                    pass
            return msgs, last

    msgs, last = asyncio.run(go())
    started = [m for m in msgs if m.get("kind") == "balance_started"]
    downs = [m for m in msgs if m.get("kind") == "foot_touchdown"]
    assert len(started) == 1 and started[0]["standing_foot"] == "left"
    assert len(downs) == 1 and downs[0]["type"] == "alert" and downs[0]["foot"] == "right"
    assert downs[0]["held_s"] == pytest.approx(20 / 15, abs=0.15)
    poses = [m for m in msgs if m["type"] == "pose"]
    assert all("balance" in m and "spine" in m for m in poses)
    assert last["balance"]["touch_count"] == 0      # recalibrate reset the session


# ----------------------------- floor calibration ----------------------------- #

def hd(world=None, img_y=None, img_x=None, leg_px=320.0):
    """A foot_heights()-style dict from per-foot world heights (m above floor) and
    image y of each foot's lowest point (pixels, y down)."""
    world = world or {"left": 0.0, "right": 0.0}
    img_y = img_y or {"left": 820.0, "right": 820.0}
    img_x = img_x or {"left": 360.0, "right": 280.0}
    gw = min(world.values())
    gi = max(img_y.values())
    return {"visibility": 0.95, "leg_px": leg_px,
            "lowest_px": {s: [img_x[s], img_y[s]] for s in ("left", "right")},
            "world": {s: world[s] - gw for s in world},
            "image": {s: (gi - img_y[s]) / leg_px for s in img_y}}


def test_floor_calibration_removes_per_foot_offset():
    # this camera/model reads the left foot 2.5 cm "higher" even when it's down
    biased = lambda l, r: hd(world={"left": 0.025 + l, "right": r})
    mon = BalanceMonitor()
    res, t = feed(mon, [biased(0, 0)] * 20)                  # 1.3 s both feet down
    assert res[-1]["floor_calibrated"]
    assert res[-1]["foot_heights"]["left"] == pytest.approx(0, abs=1e-6)
    # lift the left foot 6 cm: true height 6 cm (uncalibrated would read 8.5 cm)
    res, _ = feed(mon, [biased(0.06, 0)] * 5, t)
    assert res[-1]["foot_heights"]["left"] == pytest.approx(0.06, abs=1e-6)
    assert res[-1]["state"] == "idle"      # 6 cm is below the 8 cm lift threshold


def test_uncalibrated_falls_back_to_relative():
    mon = BalanceMonitor()
    res, _ = feed(mon, [hd(world={"left": 0.1, "right": 0.0})] * 3)
    assert not res[-1]["floor_calibrated"]
    assert res[-1]["foot_heights"]["left"] == pytest.approx(0.1)


def test_image_floor_line_ignores_standing_foot_jitter():
    cfg = BalanceConfig(mode="image")
    mon = BalanceMonitor(cfg)
    _, t = feed(mon, [hd()] * 20)                        # learn floor line at y=820
    # balance on the right foot; left raised 64 px (0.2 legs). The standing foot's
    # landmark jitters up by 15 px, which used to read as the raised foot moving down.
    seq = [hd(img_y={"left": 756.0, "right": 820.0})] * 10
    seq += [hd(img_y={"left": 756.0, "right": 805.0})] * 10
    res, _ = feed(mon, seq, t)
    heights_left = [r["foot_heights"]["left"] for r in res[3:]]     # after the median filter fills
    assert all(h == pytest.approx(0.2, abs=1e-6) for h in heights_left)
    # without calibration the same jitter would read 0.153
    raw = hd(img_y={"left": 756.0, "right": 805.0})["image"]["left"]
    assert raw == pytest.approx(49 / 320)


def test_lifting_during_calibration_restarts_it():
    mon = BalanceMonitor()
    _, t = feed(mon, [hd()] * 8 + [hd(world={"left": 0.1, "right": 0})] * 3)
    assert mon.floor is None
    res, _ = feed(mon, [hd()] * 20, t)
    assert res[-1]["floor_calibrated"]
