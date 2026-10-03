"""Tests that run without the MediaPipe model file.

    pytest -q
"""

import asyncio
import base64
import json

import cv2
import numpy as np
import pytest
import websockets

from pose_analyzer import (L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, PoseResult,
                           spine_angles_2d, spine_angles_3d, trunk_visibility)
import ws_server


def make_landmarks(hip, shoulder, half_width=0.1, vis=0.9):
    """33 landmarks with only hips/shoulders placed; hip/shoulder are mid-points."""
    lms = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0} for _ in range(33)]
    for idx, (base, dx) in {L_HIP: (hip, -half_width), R_HIP: (hip, half_width),
                            L_SHOULDER: (shoulder, -half_width), R_SHOULDER: (shoulder, half_width)}.items():
        lms[idx] = {"x": base[0] + dx, "y": base[1], "z": base[2] if len(base) > 2 else 0.0,
                    "visibility": vis}
    return lms


# ------------------------------- 2D angle --------------------------------- #

def test_2d_upright_is_zero():
    lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.5, 0.3))
    assert spine_angles_2d(lms, 640, 480)["trunk_angle_deg"] == pytest.approx(0, abs=1e-6)


def test_2d_lean_right_45():
    # 100 px right, 100 px up in a square image -> +45 deg
    lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.6, 0.6))
    assert spine_angles_2d(lms, 1000, 1000)["trunk_angle_deg"] == pytest.approx(45, abs=0.01)


def test_2d_respects_aspect_ratio():
    # same normalized offsets, but image is twice as wide -> steeper lean
    lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.6, 0.6))
    assert spine_angles_2d(lms, 2000, 1000)["trunk_angle_deg"] == pytest.approx(63.43, abs=0.01)


def test_2d_lean_left_negative():
    lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.4, 0.6))
    assert spine_angles_2d(lms, 1000, 1000)["trunk_angle_deg"] == pytest.approx(-45, abs=0.01)


# ------------------------------- 3D angle --------------------------------- #

def test_3d_upright():
    # world coords: y is DOWN, so shoulders are at negative y
    s = spine_angles_3d(make_landmarks(hip=(0, 0, 0), shoulder=(0, -0.5, 0)))
    assert s["inclination_deg"] == pytest.approx(0, abs=1e-6)
    assert s["flexion_deg"] == pytest.approx(0, abs=1e-6)
    assert s["lateral_deg"] == pytest.approx(0, abs=1e-6)
    assert s["trunk_length_m"] == pytest.approx(0.5)


def test_3d_forward_bend_30():
    # lean toward camera (negative z) by 30 deg
    a = np.radians(30)
    s = spine_angles_3d(make_landmarks(hip=(0, 0, 0), shoulder=(0, -0.5 * np.cos(a), -0.5 * np.sin(a))))
    assert s["inclination_deg"] == pytest.approx(30, abs=0.01)
    assert s["flexion_deg"] == pytest.approx(30, abs=0.01)
    assert s["lateral_deg"] == pytest.approx(0, abs=0.01)


def test_3d_backward_and_lateral():
    s = spine_angles_3d(make_landmarks(hip=(0, 0, 0), shoulder=(0.2, -0.4, 0.2)))
    assert s["flexion_deg"] < 0          # leaning away from camera
    assert s["lateral_deg"] == pytest.approx(26.57, abs=0.01)
    assert s["inclination_deg"] == pytest.approx(35.26, abs=0.01)


def test_trunk_visibility_is_minimum():
    lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.5, 0.3), vis=0.9)
    lms[R_HIP]["visibility"] = 0.2
    assert trunk_visibility(lms) == pytest.approx(0.2)


# ----------------------------- WebSocket server ----------------------------- #

class FakeAnalyzer:
    def __init__(self):
        self.shapes = []

    def process_bgr(self, frame):
        self.shapes.append(frame.shape)
        h, w = frame.shape[:2]
        lms = make_landmarks(hip=(0.5, 0.7), shoulder=(0.55, 0.3))
        world = make_landmarks(hip=(0, 0, 0), shoulder=(0.05, -0.5, -0.1))
        spine = {**spine_angles_2d(lms, w, h), **spine_angles_3d(world),
                 "trunk_visibility": 0.9, "reliable": True}
        return PoseResult(True, landmarks=lms, world_landmarks=world,
                          spine=spine, inference_ms=1.0)

    def close(self):
        pass


def jpeg(w=64, h=32):
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    return buf.tobytes()


def test_server_protocol():
    fake = FakeAnalyzer()

    async def run():
        async with websockets.serve(ws_server.make_handler(lambda: fake, "pose"), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                # binary JPEG
                await ws.send(jpeg())
                r = json.loads(await ws.recv())
                assert r["type"] == "pose" and r["detected"] and r["image_size"] == [64, 32]

                # JSON + base64 with rotation and metadata echo
                await ws.send(json.dumps({"type": "frame", "frame_id": 7, "timestamp_ms": 123,
                                          "rotate": 90,
                                          "image": base64.b64encode(jpeg()).decode()}))
                r = json.loads(await ws.recv())
                assert r["frame_id"] == 7 and r["client_timestamp_ms"] == 123
                assert r["image_size"] == [32, 64]   # rotated

                # ping
                await ws.send(json.dumps({"type": "ping"}))
                assert json.loads(await ws.recv()) == {"type": "pong"}

                # garbage doesn't kill the connection
                await ws.send(b"not a jpeg")
                assert json.loads(await ws.recv())["type"] == "error"
                await ws.send(jpeg())
                assert json.loads(await ws.recv())["type"] == "pose"

    asyncio.run(run())


class CrashingAnalyzer(FakeAnalyzer):
    def process_bgr(self, frame):
        raise RuntimeError("boom")


def test_server_reports_processing_errors_instead_of_hanging():
    async def run():
        async with websockets.serve(ws_server.make_handler(CrashingAnalyzer, "pose"), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for _ in range(2):  # connection survives repeated failures
                    await ws.send(jpeg())
                    r = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    assert r["type"] == "error" and "boom" in r["error"]

    asyncio.run(run())


def test_server_reports_model_load_failure():
    def bad_factory():
        raise FileNotFoundError("no model")

    async def run():
        async with websockets.serve(ws_server.make_handler(bad_factory, "pose"), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                r = json.loads(await asyncio.wait_for(ws.recv(), 5))
                assert r["type"] == "error" and "no model" in r["error"]

    asyncio.run(run())
