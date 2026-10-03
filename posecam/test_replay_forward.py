"""Replay forwarding: sampling, bursts, events, /end, non-blocking. No network, no models."""

import asyncio
import base64
import json
import time

import cv2
import numpy as np
import websockets

import ws_server
from replay_forward import ReplayForwarder

JPEG = cv2.imencode(".jpg", np.zeros((64, 36, 3), np.uint8))[1].tobytes()


class Recorder:
    def __init__(self, delay=0.0):
        self.calls, self.delay = [], delay

    def __call__(self, url, body, ctype):
        time.sleep(self.delay)
        self.calls.append((url, body, ctype))

    def frames(self):
        return [c for c in self.calls if c[0].endswith("/frame")]

    def field(self, call, name):
        body = call[1]
        i = body.index(f'name="{name}"'.encode())
        return body[i:].split(b"\r\n\r\n", 1)[1].split(b"\r\n", 1)[0].decode()


def feed(fw, t0, t1, fps=30.0):
    t = t0
    while t < t1:
        fw.offer_frame(JPEG, t, (1080, 1920))
        t += 1000 / fps


def test_nothing_sent_outside_a_test():
    rec = Recorder()
    fw = ReplayForwarder("http://replay:8017", sender=rec)
    feed(fw, 0, 3000)
    fw.flush()
    assert rec.calls == []


def test_uniform_rate_during_a_test():
    rec = Recorder()
    fw = ReplayForwarder("http://replay:8017/", sender=rec)
    fw.on_messages([{"kind": "bess_started", "stance": "tandem"}], 1000)
    feed(fw, 1000, 5000)  # 4 s at 30 fps
    fw.flush()
    frames = rec.frames()
    assert 5 <= len(frames) <= 7  # 1.5 fps
    assert frames[0][0] == "http://replay:8017/replay/bess-tandem-1000/frame"
    assert rec.field(frames[0], "kind") == "uniform"
    assert json.loads(rec.field(frames[0], "crop")) == [0, 0, 1080, 1920]  # full frame, never cropped
    assert JPEG in frames[0][1]


def test_burst_includes_the_half_second_before_the_error():
    rec = Recorder()
    fw = ReplayForwarder("http://replay:8017", sender=rec)
    fw.on_messages([{"kind": "bess_started", "stance": "single"}], 0)
    feed(fw, 0, 2000)
    fw.on_messages([{"kind": "bess_error", "error": "step_stumble_fall", "counted": True}], 2000)
    feed(fw, 2000 + 1000 / 30, 3000)
    fw.flush()
    burst_t = sorted(float(rec.field(c, "t")) for c in rec.frames() if rec.field(c, "kind") == "burst")
    assert burst_t[0] <= 1550 and burst_t[-1] >= 2400  # -0.5 s .. +0.5 s
    assert all(b - a >= 99 for a, b in zip(burst_t, burst_t[1:]))  # <= 10 fps
    all_t = [float(rec.field(c, "t")) for c in rec.frames()]
    assert len(all_t) == len(set(all_t))  # each frame sent once


def test_end_posts_events_on_bess_done():
    rec = Recorder()
    fw = ReplayForwarder("http://replay:8017", patient_height_cm=185, sender=rec)
    fw.on_messages([{"kind": "bess_started", "stance": "single"}], 100)
    fw.on_messages([{"kind": "foot_touchdown", "foot": "right"}], 4200)
    fw.on_messages([{"kind": "bess_error", "error": "hands_off_hips", "label": "Hands off hips"}], 5300)
    fw.on_messages([{"kind": "bess_done", "errors": 1}], 20100)
    fw.flush()
    end = [c for c in rec.calls if c[0].endswith("/end")]
    assert len(end) == 1 and end[0][0] == "http://replay:8017/replay/bess-single-100/end"
    payload = json.loads(end[0][1])
    assert payload["patient_height_cm"] == 185
    assert [(e["t"], e["kind"], e["side"]) for e in payload["events"]] == [
        (4200, "foot_down", "right"), (5300, "hands_off_hips", None)]
    assert fw.trial is None


def test_never_blocks_and_drops_when_full():
    rec = Recorder(delay=0.2)
    fw = ReplayForwarder("http://replay:8017", sender=rec, max_queue=2)
    fw.on_messages([{"kind": "bess_started", "stance": "double"}], 0)
    t0 = time.perf_counter()
    for k in range(20):
        fw.on_messages([{"kind": "bess_error", "error": "foot_lift"}], 1000 * k)
        feed(fw, 1000 * k, 1000 * k + 600)
    assert time.perf_counter() - t0 < 0.5  # the server thread never waits on the network
    assert fw.dropped > 0


def test_sender_errors_are_swallowed():
    def boom(*a):
        raise OSError("replay down")
    fw = ReplayForwarder("http://replay:8017", sender=boom)
    fw.on_messages([{"kind": "bess_started", "stance": "double"}], 0)
    feed(fw, 0, 1000)
    fw.flush()  # no exception escapes


class ScriptedPipeline:
    """Fake analyzer: emits bess_started / bess_error / bess_done on chosen frames."""

    def __init__(self):
        self.n = 0

    def process_bgr(self, frame):
        self.n += 1
        ev = {1: [{"kind": "bess_started", "stance": "tandem"}],
              4: [{"kind": "bess_error", "error": "foot_lift", "counted": True}],
              8: [{"kind": "bess_done", "errors": 1}]}.get(self.n, [])
        return {"detected": True, "bess": {"events": ev}}

    def close(self):
        pass


def bess_events(result, frame_id):
    return [{"type": "event", "frame_id": frame_id, **e} for e in result.get("bess", {}).get("events", [])]


def test_server_stamps_events_and_forwards():
    rec = Recorder()
    fw_holder = {}

    def factory():
        fw_holder["fw"] = ReplayForwarder("http://replay:8017", sender=rec)
        return fw_holder["fw"]

    async def go():
        handler = ws_server.make_handler(ScriptedPipeline, "pose", bess_events, forwarder_factory=factory)
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            events = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for k in range(8):
                    await ws.send(json.dumps({"type": "frame", "image": base64.b64encode(JPEG).decode(),
                                              "frame_id": k, "timestamp_ms": 1_000_000 + 100 * k}))
                    while True:  # wait for this frame's result (+ any events)
                        m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                        if m["type"] == "event":
                            events.append(m)
                        if m["type"] == "pose" and m["frame_id"] == k:
                            break
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                m = json.loads(await asyncio.wait_for(ws.recv(), 5))  # last frame's events
                events.append(m)
            return events

    events = asyncio.run(go())
    assert events and all(e.get("client_timestamp_ms") is not None for e in events)
    fw_holder["fw"].flush()
    urls = [c[0] for c in rec.calls]
    assert any(u.endswith("/frame") for u in urls)
    assert sum(u.endswith("/end") for u in urls) == 1
