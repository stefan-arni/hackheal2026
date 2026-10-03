"""Session-wide frame queue for fal: one priority order across all trials of a session.

Workers pull the best job each time a slot frees up (priorities are re-evaluated at pick time,
because deadlines appear only when a trial ends):
  1. error-burst frames, plus each trial's first ANCHOR_FRAMES coarse uniform frames (the error
     replay needs a few frames with both feet down to fit the floor)
  2. uniform frames, earliest trial deadline first, coarse-to-fine within a trial
     (every 4th uniform frame, then every 2nd, then the rest), then by capture time
  3. anything whose trial already passed its deadline (stragglers: still processed, last)
Worker count should match fal's real parallelism (measured ~3), so our order is the order
fal works in; extra submissions would only queue at fal in FIFO order.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

EXPECTED_TRIAL_S = 30.0  # deadline guess for a trial that is still recording
ANCHOR_FRAMES = 4  # first coarse uniform frames of a trial that run with the bursts


def coarse_level(k: int) -> int:
    """Coarse-to-fine order of the k-th uniform frame of a trial: 0 (every 4th), 1 (every 2nd), 2."""
    return 0 if k % 4 == 0 else (1 if k % 2 == 0 else 2)


@dataclass
class Job:
    trial: Any  # app.Trial (needs .deadline_at, .created_at)
    stem: str
    kind: str  # "burst" | "uniform"
    level: int
    t_ms: float
    payload: dict = field(default_factory=dict)
    seq: int = 0
    anchor: bool = False  # uniform frame needed for the early error replay

    def key(self, now: float, deadline_s: float) -> tuple:
        dl = self.trial.deadline_at or (self.trial.created_at + EXPECTED_TRIAL_S + deadline_s)
        cls = 0 if self.kind == "burst" or self.anchor else 1
        if now > dl:
            cls += 2  # past its trial's deadline: straggler, processed last
        return (cls, dl, self.level if self.kind == "uniform" else 0, self.t_ms, self.seq)


class FrameScheduler:
    def __init__(self, run: Callable[[Job], Awaitable[None]], workers: int = 3, deadline_s: float = 45.0):
        self.run, self.workers, self.deadline_s = run, workers, deadline_s
        self.jobs: list[Job] = []
        self.running: dict[int, Job] = {}
        self.cond: asyncio.Condition | None = None
        self.done_times: deque = deque(maxlen=200)
        self._seq = itertools.count()
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        if self._tasks:
            return
        self.cond = asyncio.Condition()
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.workers)]

    async def submit(self, job: Job) -> None:
        self.start()
        job.seq = next(self._seq)
        async with self.cond:
            self.jobs.append(job)
            self.cond.notify()

    async def poke(self) -> None:
        """Re-evaluate priorities (e.g. a trial just got its deadline)."""
        if self.cond is not None:
            async with self.cond:
                self.cond.notify_all()

    def pending(self, trial=None) -> int:
        return sum(1 for j in self.jobs if trial is None or j.trial is trial)

    def in_flight(self, trial=None) -> int:
        return sum(1 for j in self.running.values() if trial is None or j.trial is trial)

    def ahead_of(self, trial) -> int:
        """Jobs that will run before this trial's last job (incl. its own)."""
        now = time.time()
        mine = [j.key(now, self.deadline_s) for j in self.jobs if j.trial is trial]
        if not mine:
            return 0
        worst = max(mine)
        return sum(1 for j in self.jobs if j.key(now, self.deadline_s) <= worst)

    def rate(self) -> float:
        """Recent completions per second (falls back to workers / 5 s)."""
        now = time.time()
        recent = [t for t in self.done_times if now - t < 60]
        if len(recent) >= 3:
            return len(recent) / max(now - recent[0], 1.0)
        return self.workers / 5.0

    def eta_s(self, trial) -> float | None:
        n = self.ahead_of(trial) + self.in_flight(trial)
        return None if n == 0 else math.ceil(n / self.rate())

    async def _worker(self, i: int) -> None:
        while True:
            async with self.cond:
                while not self.jobs:
                    await self.cond.wait()
                now = time.time()
                job = min(self.jobs, key=lambda j: j.key(now, self.deadline_s))
                self.jobs.remove(job)
                self.running[i] = job
            try:
                await self.run(job)
            finally:
                self.running.pop(i, None)
                self.done_times.append(time.time())
