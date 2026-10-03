"""Eye-alignment tests on synthetic faces (no MediaPipe model needed).

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
from eye_server import eye_events
from eye_tracker import (EYES, FACE_EDGE_A, FACE_EDGE_B, NOSE_TIP, AlignmentMonitor,
                         EyeAnalyzer, MonitorConfig, eye_geometry, head_yaw_ratio)

EYE_W = 60.0


def make_face(r_h=0.5, l_h=0.5, r_v=0.0, l_v=0.0, openness=0.3, yaw=0.0, roll_deg=0.0):
    """478 pixel points with eyes/irises/nose placed exactly.

    Unmirrored camera: subject's RIGHT eye appears on the image LEFT.
    h is the iris position along the eye (0..1, image left->right), v in eye-widths.
    """
    pts = np.zeros((478, 2))

    def place_eye(spec, x0, h, v):
        outer_inner = spec["corners"]
        left_px, right_px = np.array([x0, 200.0]), np.array([x0 + EYE_W, 200.0])
        # assign whichever corner index sits on the image left
        a_idx, b_idx = outer_inner if spec is EYES["right"] else (outer_inner[0], outer_inner[1])
        pts[a_idx], pts[b_idx] = left_px, right_px
        mid = (left_px + right_px) / 2
        pts[spec["upper"]] = mid + [0, -openness * EYE_W / 2]
        pts[spec["lower"]] = mid + [0, openness * EYE_W / 2]
        c = left_px + [h * EYE_W, v * EYE_W]
        r = 7.0
        idx = spec["iris"]
        pts[idx[0]] = c
        for i, (dx, dy) in zip(idx[1:], [(r, 0), (0, -r), (-r, 0), (0, r)]):
            pts[i] = c + [dx, dy]

    place_eye(EYES["right"], 100.0, r_h, r_v)   # image left
    place_eye(EYES["left"], 220.0, l_h, l_v)    # image right

    # nose + face edges for yaw: yaw shifts the nose toward one edge
    pts[FACE_EDGE_A] = [60.0, 260.0]
    pts[FACE_EDGE_B] = [320.0, 260.0]
    pts[NOSE_TIP] = [190.0 + yaw * 130.0, 260.0]

    if roll_deg:
        th = math.radians(roll_deg)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pts = (pts - [190, 230]) @ R.T + [190, 230]
    return pts


def analyzer(**cfg):
    """EyeAnalyzer without loading the MediaPipe model."""
    a = EyeAnalyzer.__new__(EyeAnalyzer)
    a.cfg = MonitorConfig(**cfg)
    a.monitor = AlignmentMonitor(a.cfg)
    return a


def run(a, faces, t0=0.0, dt=0.1):
    """Feed a list of faces at 10 fps, return list of results."""
    out = []
    for i, f in enumerate(faces):
        out.append(a.analyze_points(f, t0 + i * dt))
    return out, t0 + len(faces) * dt


def calibrate(a, face=None):
    res, t = run(a, [face if face is not None else make_face()] * 25)
    assert any(r["event"] == "calibrated" for r in res)
    return t


# ------------------------------- geometry ---------------------------------- #

def test_geometry_centered_iris():
    g = eye_geometry(make_face(), EYES["right"])
    assert g["h"] == pytest.approx(0.5)
    assert g["v"] == pytest.approx(0.0, abs=1e-9)
    assert g["openness"] == pytest.approx(0.3)


def test_geometry_offsets():
    g = eye_geometry(make_face(l_h=0.7, l_v=0.1), EYES["left"])
    assert g["h"] == pytest.approx(0.7)
    assert g["v"] == pytest.approx(0.1)


def test_geometry_ignores_head_roll():
    flat = eye_geometry(make_face(r_h=0.62, r_v=0.05), EYES["right"])
    tilted = eye_geometry(make_face(r_h=0.62, r_v=0.05, roll_deg=20), EYES["right"])
    assert tilted["h"] == pytest.approx(flat["h"], abs=1e-6)
    assert tilted["v"] == pytest.approx(flat["v"], abs=1e-6)


def test_yaw_ratio():
    assert head_yaw_ratio(make_face()) == pytest.approx(0, abs=1e-9)
    assert head_yaw_ratio(make_face(yaw=0.5)) > 0.35


# ------------------------------- alert logic -------------------------------- #

def test_calibration_then_monitoring():
    a = analyzer()
    res, _ = run(a, [make_face()] * 25)
    assert res[0]["state"] == "calibrating"
    assert res[-1]["state"] == "monitoring"
    assert sum(r["event"] == "calibrated" for r in res) == 1


def test_both_eyes_looking_sideways_is_not_an_alert():
    a = analyzer()
    t = calibrate(a)
    res, _ = run(a, [make_face(r_h=0.68, l_h=0.68)] * 20, t)
    assert not any(r["alerting"] for r in res)
    assert abs(res[-1]["deviation_h"]) < 1e-6


def test_one_eye_drifting_triggers_alert():
    a = analyzer()
    t = calibrate(a)
    res, _ = run(a, [make_face(l_h=0.68)] * 15, t)    # left eye turns out, right stays
    starts = [r for r in res if r["event"] == "start"]
    assert len(starts) == 1
    assert starts[0]["drifting_eye"] == "left"
    assert starts[0]["direction"] == "horizontal"
    assert res[-1]["alerting"]


def test_vertical_drift_and_which_eye():
    a = analyzer()
    t = calibrate(a)
    res, _ = run(a, [make_face(r_v=0.12)] * 15, t)
    start = next(r for r in res if r["event"] == "start")
    assert start["drifting_eye"] == "right" and start["direction"] == "vertical"


def test_brief_blip_does_not_alert():
    a = analyzer(hold_s=0.5)
    t = calibrate(a)
    faces = [make_face(l_h=0.7)] * 3 + [make_face()] * 10   # 0.3 s blip
    res, _ = run(a, faces, t)
    assert not any(r["alerting"] for r in res)


def test_alert_clears_when_eyes_realign():
    a = analyzer()
    t = calibrate(a)
    res, t = run(a, [make_face(l_h=0.7)] * 15, t)
    assert res[-1]["alerting"]
    res, _ = run(a, [make_face()] * 20, t)
    ends = [r for r in res if r["event"] == "end"]
    assert len(ends) == 1 and ends[0]["alert_duration_s"] > 0
    assert not res[-1]["alerting"]


def test_baseline_absorbs_natural_asymmetry():
    # this person's eyes naturally sit slightly differently: should not alert
    a = analyzer()
    t = calibrate(a, make_face(r_h=0.45, l_h=0.58))
    res, _ = run(a, [make_face(r_h=0.45, l_h=0.58)] * 20, t)
    assert not any(r["alerting"] for r in res)


def test_blinks_and_head_turns_are_skipped():
    a = analyzer()
    t = calibrate(a)
    res, _ = run(a, [make_face(l_h=0.8, openness=0.05)] * 15, t)
    assert all(r["skip_reason"] == "blink" for r in res)
    assert not any(r["alerting"] for r in res)
    res, _ = run(a, [make_face(l_h=0.8, yaw=0.6)] * 15, t + 2)
    assert all(r["skip_reason"] == "head_turned" for r in res)
    assert not any(r["alerting"] for r in res)


def test_calibration_waits_for_usable_frames():
    a = analyzer()
    res, t = run(a, [make_face(openness=0.05)] * 30)     # eyes closed the whole time
    assert all(r["state"] == "calibrating" for r in res)
    calibrate(a)


def test_recalibrate_resets():
    a = analyzer()
    t = calibrate(a)
    run(a, [make_face(l_h=0.7)] * 15, t)
    a.recalibrate()
    assert a.monitor.baseline is None and not a.monitor.alerting


# ----------------------- server: alert + recalibrate ------------------------ #

class ScriptedEyes:
    """Server-side analyzer that replays synthetic faces on a fake 10 fps clock."""

    def __init__(self, script):
        self.a = analyzer()
        self.script = script
        self.i = 0

    def process_bgr(self, frame):
        face = self.script[min(self.i, len(self.script) - 1)]
        r = self.a.analyze_points(face, self.i * 0.1)
        self.i += 1
        return r

    def recalibrate(self):
        self.a.recalibrate()

    def close(self):
        pass


def test_server_sends_alert_and_status_messages():
    script = [make_face()] * 25 + [make_face(l_h=0.7)] * 15 + [make_face()] * 20
    ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
    jpg = buf.tobytes()

    async def go():
        async with websockets.serve(ws_server.make_handler(lambda: ScriptedEyes(script), "eyes", eye_events),
                                    "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for _ in script:
                    await ws.send(jpg)
                    while True:   # wait for this frame's result (+ any alert after it)
                        m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        msgs.append(m)
                        if m["type"] == "eyes":
                            break
                # recalibrate -> next frame is calibrating again
                await ws.send(json.dumps({"type": "recalibrate"}))
                await ws.send(jpg)
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    if m["type"] == "eyes":
                        break
                # drain anything left
                try:
                    while True:
                        msgs.append(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
                except asyncio.TimeoutError:
                    pass
            return msgs, m

    msgs, after_recal = asyncio.run(go())
    kinds = [(m["type"], m.get("kind"), m.get("event")) for m in msgs if m["type"] != "eyes"]
    assert ("status", "eye_calibrated", None) in kinds
    starts = [m for m in msgs if m["type"] == "alert" and m["event"] == "start"]
    ends = [m for m in msgs if m["type"] == "alert" and m["event"] == "end"]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0]["drifting_eye"] == "left" and starts[0]["kind"] == "eye_misalignment"
    assert after_recal["state"] == "calibrating"


def test_pose_and_eye_servers_are_independent():
    """The pose side never imports the iris tracker or the eye server."""
    import re
    for f in ("server.py", "pose_pipeline.py", "bess.py", "balance.py", "duck.py", "pose_analyzer.py"):
        src = open(f).read()
        assert not re.search(r"^\s*(from|import)\s+(eye_tracker|eye_server)\b", src, re.M), f
