"""Optional, non-blocking live publishing to the replay service's doctor dashboard.

    python server.py --publish-url http://<laptop>:8017 --session-id <id>     # off unless given

Pushes ~10 Hz summaries of each processed frame (2D landmarks, balance state, foot heights,
trunk angle, BESS stance/timer/errors/scores, trial ID, session ID) and every event message,
as JSON batches to POST {url}/live/{sessionId}/push.

Never blocks the pose server: one background thread drains a bounded queue and sends whatever
is queued as one batch. When the queue is full a summary is dropped (never an event).
Standard library only. The trial ID comes from the replay forwarder when it is running (same
ID on both channels); otherwise it is derived the same way (bess-<stance>-<ms at bess_start>).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from typing import Callable

log = logging.getLogger("replay-publish")


def http_send_json(url: str, payload: dict, timeout: float = 5.0) -> None:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


class LivePublisher:
    def __init__(self, url: str, session_id: str, *, hz: float = 10.0,
                 sender: Callable[[str, dict], None] = http_send_json, max_queue: int = 64,
                 clock: Callable[[], float] = time.time, forwarder=None):
        self.url = f"{url.rstrip('/')}/live/{session_id}/push"
        self.session = session_id
        self.period_ms = 1000.0 / hz
        self.sender = sender
        self.clock = clock
        self.forwarder = forwarder
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.last_summary = -1e18
        self._trial: str | None = None
        self.dropped = 0
        self.sent_batches = 0
        threading.Thread(target=self._drain, name="replay-publish", daemon=True).start()

    @property
    def trial(self) -> str | None:
        return self.forwarder.trial if self.forwarder is not None else self._trial

    # ---- input (server event loop; quick)

    def on_result(self, result: dict, msgs: list[dict], t_ms: float | None, recv_ms: float | None,
                  size: tuple[int, int]) -> None:
        now = float(recv_ms) if recv_ms is not None else self.clock() * 1000
        t = float(t_ms) if t_ms is not None else now
        for m in msgs:
            if m.get("kind") == "bess_started" and self.forwarder is None:
                self._trial = f"bess-{m.get('stance', 'test')}-{int(t)}"
            self._put({"type": "event", "t": t, "trial": self.trial, "session": self.session,
                       **{k: v for k, v in m.items() if k != "type"}, "msg_type": m.get("type")}, event=True)
            if m.get("kind") in ("bess_done", "bess_failed", "bess_cancelled") and self.forwarder is None:
                self._trial = None
        if t - self.last_summary < 0.9 * self.period_ms:  # 10% slack: 30 fps -> every 3rd frame = 10 Hz
            return
        self.last_summary = t
        self._put(self.summary(result, t, size), event=False)

    def summary(self, result: dict, t: float, size: tuple[int, int]) -> dict:
        lms = result.get("landmarks") or []
        b = result.get("balance") or {}
        sp = result.get("spine") or {}
        bs = result.get("bess") or {}
        sess = bs.get("session") or {}
        return {
            "type": "summary", "t": t, "trial": self.trial, "session": self.session,
            "detected": bool(result.get("detected")), "image_size": list(size),
            "landmarks": [[_r(l.get("x")), _r(l.get("y")), _r(l.get("visibility"), 2)] for l in lms],
            "balance": {"state": b.get("state"), "lifted_foot": b.get("lifted_foot"), "standing_foot": b.get("standing_foot"),
                        "balance_time_s": b.get("balance_time_s"), "foot_heights": b.get("foot_heights"),
                        "skip_reason": b.get("skip_reason")},
            "spine": {"trunk_angle_deg": sp.get("trunk_angle_deg"), "lateral_deg": sp.get("lateral_deg"),
                      "flexion_deg": sp.get("flexion_deg"), "reliable": sp.get("reliable")},
            "bess": {"phase": bs.get("phase"), "stance": bs.get("stance"), "time_left": bs.get("time_left"),
                     "countdown_left": bs.get("countdown_left"), "errors": bs.get("errors"), "by_type": bs.get("by_type"),
                     "active": bs.get("active"), "scores": sess.get("scores"), "total": sess.get("total")},
        }

    # ---- internals

    def _put(self, item: dict, event: bool) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            if not event:
                self.dropped += 1
                return
            try:  # make room for the event by dropping the oldest queued item
                self.q.get_nowait()
                self.q.task_done()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def _drain(self) -> None:
        while True:
            batch = [self.q.get()]
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            try:
                self.sender(self.url, {"items": batch})
                self.sent_batches += 1
            except Exception as e:  # the pose server must keep running whatever the dashboard does
                log.warning("live publish failed: %s", e)
            finally:
                for _ in batch:
                    self.q.task_done()

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.01)
