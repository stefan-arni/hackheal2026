"""Sessions and the doctor dashboard's live feed.

posecam's publisher POSTs batches to /live/{session}/push: ~10 Hz summaries (landmarks, balance,
trunk angle, BESS timer/errors) and every event. The dashboard subscribes to
/live/{session}/stream (server-sent events) and gets a snapshot first, then each item.
The service itself also posts replay-progress items (frames done, stage published) here.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    name: str = ""
    trials: list[str] = field(default_factory=list)  # in start order
    summary: dict | None = None  # latest posecam summary
    feed: deque = field(default_factory=lambda: deque(maxlen=300))  # events + replay progress
    subscribers: set = field(default_factory=set)

    def snapshot(self) -> dict:
        return {"type": "snapshot", "session": self.id, "name": self.name, "trials": self.trials,
                "summary": self.summary, "feed": list(self.feed)}

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "created": self.created, "trials": self.trials}


class LiveHub:
    def __init__(self, store: Path):
        self.store = store
        self.store.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        for p in self.store.glob("*.json"):  # survive a service restart
            try:
                d = json.loads(p.read_text())
                self.sessions[d["id"]] = Session(d["id"], d.get("created", time.time()), d.get("name", ""), d.get("trials", []))
            except (ValueError, KeyError):
                pass

    def create(self, name: str = "") -> Session:
        sid = time.strftime("s-%Y%m%d-%H%M%S")
        while sid in self.sessions:
            sid += "x"
        s = Session(sid, name=name)
        self.sessions[sid] = s
        self.save(s)
        return s

    def get(self, sid: str, create: bool = False) -> Session | None:
        if sid not in self.sessions and create:
            self.sessions[sid] = Session(sid)
            self.save(self.sessions[sid])
        return self.sessions.get(sid)

    def save(self, s: Session) -> None:
        (self.store / f"{s.id}.json").write_text(json.dumps(s.to_json(), indent=1))

    def add_trial(self, s: Session, trial_id: str) -> None:
        if trial_id not in s.trials:
            s.trials.append(trial_id)
            self.save(s)
            self.publish(s, {"type": "trial", "trial": trial_id, "t_wall": time.time()})

    def publish(self, s: Session, item: dict[str, Any]) -> None:
        if item.get("type") == "summary":
            s.summary = item
        else:
            s.feed.append(item)
        for q in list(s.subscribers):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:  # a slow dashboard drops items rather than stalling the service
                pass

    async def stream(self, s: Session):
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        s.subscribers.add(q)
        try:
            yield f"data: {json.dumps(s.snapshot())}\n\n"
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(item)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            s.subscribers.discard(q)
