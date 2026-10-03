"""Optional, non-blocking forwarding of frames and events to the replay service.

    python server.py --replay-url http://<laptop>:8017            # off unless given

During a BESS test (bess_started -> bess_done/failed/cancelled) this sends:
  - full frames (never cropped: SAM 3D Body assumes the image centre is the optical axis)
    at 1.5 fps, plus a 10 fps burst covering 0.5 s before and after every error /
    touchdown (the 0.5 s before comes from a 1 s ring buffer)
    -> POST {url}/replay/{trialId}/frame   multipart: jpeg, t, crop, frame_size, kind
  - on bess_done (or failed/cancelled): POST {url}/replay/{trialId}/end with the trial's
    events as {t, kind, side} (t = the frame's client timestamp, ms) and patient height.

Never blocks the pose server: network calls run on one background thread behind a bounded
queue; when the queue is full, items are dropped and counted. Standard library only.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
import uuid
from collections import deque
from typing import Callable

log = logging.getLogger("replay-forward")

UNIFORM_FPS = 1.5
BURST_FPS = 10.0
BURST_HALF_S = 0.5
RING_S = 1.0
END_KINDS = ("bess_done", "bess_failed", "bess_cancelled")


def _multipart(fields: dict, jpeg: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="jpeg"; filename="frame.jpg"\r\n'
                 f'Content-Type: image/jpeg\r\n\r\n'.encode() + jpeg + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def http_send(url: str, body: bytes, content_type: str, timeout: float = 10.0) -> None:
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


class ReplayForwarder:
    """One per client connection. offer_frame() for every received frame, on_messages() for
    every processed frame's event messages. Both return immediately."""

    def __init__(self, url: str, *, patient_height_cm: float | None = None,
                 sender: Callable[[str, bytes, str], None] = http_send, max_queue: int = 64,
                 clock: Callable[[], float] = time.time):
        self.url = url.rstrip("/")
        self.height = patient_height_cm
        self.sender = sender
        self.clock = clock
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.ring: deque = deque()  # (t_ms, jpeg, (w, h))
        self.trial: str | None = None
        self.events: list[dict] = []
        self.last_uniform = -1e18
        self.last_burst = -1e18
        self.burst_until = -1e18
        self.sent_ms: set[float] = set()
        self.dropped = 0
        self.sent = 0
        self._worker = threading.Thread(target=self._drain, name="replay-forward", daemon=True)
        self._worker.start()

    # ---- inputs (called on the server's event loop; must be quick)

    def offer_frame(self, jpeg: bytes, t_ms: float | None, size: tuple[int, int]) -> None:
        t = float(t_ms) if t_ms is not None else self.clock() * 1000
        self.ring.append((t, jpeg, size))
        while self.ring and t - self.ring[0][0] > RING_S * 1000:
            self.ring.popleft()
        if self.trial is None:
            return
        if t <= self.burst_until and t - self.last_burst >= 1000 / BURST_FPS:
            self._send_frame(t, jpeg, size, "burst")
            self.last_burst = t
        elif t - self.last_uniform >= 1000 / UNIFORM_FPS:
            self._send_frame(t, jpeg, size, "uniform")
            self.last_uniform = t

    def on_messages(self, msgs: list[dict], t_ms: float | None) -> None:
        t = float(t_ms) if t_ms is not None else self.clock() * 1000
        for m in msgs:
            kind = m.get("kind")
            if kind == "bess_started":
                self.trial = f"bess-{m.get('stance', 'test')}-{int(t)}"
                self.events, self.sent_ms = [], set()
                self.last_uniform = self.last_burst = self.burst_until = -1e18
                log.info("replay trial %s started", self.trial)
            elif self.trial is None:
                continue
            elif kind in ("bess_error", "foot_touchdown"):
                self.events.append({"t": t, "kind": "foot_down" if kind == "foot_touchdown" else m.get("error", kind),
                                    "side": m.get("foot"), "label": m.get("label"), "counted": m.get("counted", True)})
                self._burst(t)
            elif kind in END_KINDS:
                body = json.dumps({"events": self.events, "patient_height_cm": self.height}).encode()
                self._put((f"{self.url}/replay/{self.trial}/end", body, "application/json"))
                log.info("replay trial %s ended (%s): %d events, %d frames sent, %d dropped",
                         self.trial, kind, len(self.events), self.sent, self.dropped)
                self.trial = None

    # ---- internals

    def _burst(self, t: float) -> None:
        """10 fps from 0.5 s before the event (ring buffer) to 0.5 s after (live)."""
        last = -1e18
        for ft, jpeg, size in list(self.ring):
            if t - BURST_HALF_S * 1000 <= ft <= t and ft - last >= 1000 / BURST_FPS:
                self._send_frame(ft, jpeg, size, "burst")
                last = ft
        self.last_burst = last
        self.burst_until = max(self.burst_until, t + BURST_HALF_S * 1000)

    def _send_frame(self, t: float, jpeg: bytes, size: tuple[int, int], kind: str) -> None:
        if t in self.sent_ms:  # a frame can be both uniform and burst; send it once
            return
        self.sent_ms.add(t)
        w, h = size
        body, ctype = _multipart({"t": f"{t:.1f}", "crop": json.dumps([0, 0, w, h]),
                                  "frame_size": json.dumps([w, h]), "kind": kind}, jpeg)
        self._put((f"{self.url}/replay/{self.trial}/frame", body, ctype))

    def _put(self, item) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> None:
        while True:
            url, body, ctype = self.q.get()
            try:
                self.sender(url, body, ctype)
                self.sent += 1
            except Exception as e:  # the pose server must keep running whatever replay does
                log.warning("replay forward failed (%s): %s", url.rsplit("/", 1)[-1], e)
            finally:
                self.q.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for queued sends (tests / shutdown)."""
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)
