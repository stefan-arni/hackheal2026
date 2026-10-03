"""Live dashboard publishing: throttle, events, payload, trial IDs, drop policy, non-blocking."""

import asyncio
import base64
import json
import time

import cv2
import numpy as np
import websockets

import ws_server
from replay_forward import ReplayForwarder
from replay_publish import LivePublisher

JPEG = cv2.imencode(".jpg", np.zeros((64, 36, 3), np.uint8))[1].tobytes()


class Sink:
    def __init__(self, delay=0.0):
        self.batches, self.delay = [], delay

    def __call__(self, url, payload):
        time.sleep(self.delay)
        self.batches.append((url, payload))

    def items(self, typ=None):
        return [i for _, p in self.batches for i in p["items"] if typ is None or i["type"] == typ]


RESULT = {
    "detected": True,
    "landmarks": [{"x": 0.5, "y": 0.1 + k / 40, "visibility": 0.9} for k in range(33)],
    "balance": {"state": "balancing", "lifted_foot": "right", "standing_foot": "left", "balance_time_s": 3.2,
                "foot_heights": {"left": 0.0, "right": 0.12}},
    "spine": {"trunk_angle_deg": 12.5, "lateral_deg": 10.0, "flexion_deg": 4.0, "reliable": True},
    "bess": {"phase": "running", "stance": "single", "time_left": 6.4, "errors": 2,
             "by_type": {"step_stumble_fall": 1}, "active": [], "session": {"scores": {"single": None}, "total": 0}},
}


def test_summaries_are_throttled_to_10_hz_and_events_always_go_through():
    sink = Sink()
    pub = LivePublisher("http://replay:8017/", "sess-1", sender=sink)
    for k in range(60):  # 2 s at 30 fps
        msgs = [{"type": "alert", "kind": "foot_touchdown", "foot": "right"}] if k == 31 else []
        pub.on_result(RESULT, msgs, 1_000_000 + 33.3 * k, None, (720, 1280))
    pub.flush()
    assert sink.batches[0][0] == "http://replay:8017/live/sess-1/push"
    assert 18 <= len(sink.items("summary")) <= 21
    ev = sink.items("event")
    assert len(ev) == 1 and ev[0]["kind"] == "foot_touchdown" and ev[0]["t"] == 1_000_000 + 33.3 * 31


def test_summary_payload():
    sink = Sink()
    pub = LivePublisher("http://replay:8017", "sess-1", sender=sink)
    pub.on_result(RESULT, [], 5_000, None, (720, 1280))
    pub.flush()
    s = sink.items("summary")[0]
    assert s["session"] == "sess-1" and s["image_size"] == [720, 1280] and len(s["landmarks"]) == 33
    assert s["balance"]["foot_heights"]["right"] == 0.12 and s["spine"]["trunk_angle_deg"] == 12.5
    assert s["bess"]["stance"] == "single" and s["bess"]["time_left"] == 6.4 and s["bess"]["errors"] == 2


def test_trial_id_follows_bess_and_matches_the_forwarder():
    sink = Sink()
    fw = ReplayForwarder("http://replay:8017", session_id="sess-1", sender=lambda *a: None)
    pub = LivePublisher("http://replay:8017", "sess-1", sender=sink, forwarder=fw)
    start = [{"type": "status", "kind": "bess_started", "stance": "tandem"}]
    fw.on_messages(start, 1_791_000_000_000, recv_ms=10)
    pub.on_result(RESULT, start, 1_791_000_000_000, 10, (720, 1280))
    pub.flush()
    assert sink.items("event")[0]["trial"] == fw.trial == "bess-tandem-1791000000000"
    alone = LivePublisher("http://replay:8017", "sess-1", sender=Sink())
    alone.on_result(RESULT, start, None, 4_242, (720, 1280))  # no forwarder, no phone clock
    assert alone.trial == "bess-tandem-4242"


def test_events_survive_a_full_queue():
    sink = Sink(delay=0.3)
    pub = LivePublisher("http://replay:8017", "s", sender=sink, max_queue=4, hz=1000)
    t0 = time.perf_counter()
    for k in range(50):
        pub.on_result(RESULT, [], 1000 + k, None, (720, 1280))
    pub.on_result(RESULT, [{"type": "event", "kind": "bess_error", "error": "foot_lift"}], 2000, None, (720, 1280))
    assert time.perf_counter() - t0 < 0.2  # never blocks the server
    pub.flush()
    assert pub.dropped > 0
    assert any(i["kind"] == "bess_error" for i in sink.items("event"))


def test_send_errors_are_swallowed():
    def boom(*a):
        raise OSError("dashboard down")
    pub = LivePublisher("http://replay:8017", "s", sender=boom)
    pub.on_result(RESULT, [], 1000, None, (720, 1280))
    pub.flush()


class ScriptedPipeline:
    def __init__(self):
        self.n = 0

    def process_bgr(self, frame):
        self.n += 1
        ev = {1: [{"kind": "bess_started", "stance": "single"}], 5: [{"kind": "bess_error", "error": "foot_lift"}],
              9: [{"kind": "bess_done", "stance": "single", "errors": 1, "by_type": {"foot_lift": 1}}]}.get(self.n, [])
        return {**RESULT, "bess": {**RESULT["bess"], "events": ev}}

    def close(self):
        pass


def bess_events(result, frame_id):
    return [{"type": "event", "frame_id": frame_id, **e} for e in result.get("bess", {}).get("events", [])]


def test_server_publishes_with_the_forwarders_trial_and_session():
    sink, frames = Sink(), []
    fw_holder = {}

    def fw_factory():
        fw_holder["fw"] = ReplayForwarder("http://replay:8017", session_id="sess-9", sender=lambda u, b, c: frames.append((u, b)))
        return fw_holder["fw"]

    def pub_factory(fw):
        fw_holder["pub"] = LivePublisher("http://replay:8017", "sess-9", sender=sink, forwarder=fw)
        return fw_holder["pub"]

    async def go():
        handler = ws_server.make_handler(ScriptedPipeline, "pose", bess_events,
                                         forwarder_factory=fw_factory, publisher_factory=pub_factory)
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                for k in range(10):
                    await ws.send(json.dumps({"type": "frame", "image": base64.b64encode(JPEG).decode(),
                                              "frame_id": k, "timestamp_ms": 1_000_000 + 120 * k}))
                    while json.loads(await asyncio.wait_for(ws.recv(), 5))["type"] != "pose":
                        pass
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.1)

    asyncio.run(go())
    fw_holder["pub"].flush()
    fw_holder["fw"].flush()
    ev = sink.items("event")
    assert [e["kind"] for e in ev] == ["bess_started", "bess_error", "bess_done"]
    assert all(e["trial"] == "bess-single-1000000" and e["session"] == "sess-9" for e in ev[:2])
    assert len(sink.items("summary")) >= 5
    end = [json.loads(b) for u, b in frames if u.endswith("/end")]
    assert end and end[0]["session_id"] == "sess-9" and end[0]["bess"]["errors"] == 1
    assert any(b"sess-9" in b for u, b in frames if u.endswith("/frame"))
