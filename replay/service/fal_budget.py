"""fal spend guard: every billed call is logged, counted and capped.

    data/fal_spend.jsonl   one line per event (submitted / submit_failed / done / recovered)
    data/fal_price.json    the per-call price you confirmed from the fal dashboard

Rules:
  - hard stop at TOTAL_CAP calls ever and RUN_CAP per run, unless override=True
  - until the price is confirmed, only PROBE_CALLS calls are allowed in total
    (the 3-frame probe); confirm with `tools/fal_budget.py confirm-price 0.015`
  - a batch prints "about to send N new frames ≈ $X, total so far $Y" and asks
    for confirmation unless yes=True

Counting is conservative: a submit that died on a network error *may* have been
billed, so it counts.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPLAY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEND_LOG = REPLAY_ROOT / "data/fal_spend.jsonl"
DEFAULT_PRICE_FILE = REPLAY_ROOT / "data/fal_price.json"

EST_USD_PER_CALL = 0.015
TOTAL_CAP = 500
RUN_CAP = 80
PROBE_CALLS = 3


class BudgetError(RuntimeError):
    pass


class PriceNotConfirmed(BudgetError):
    pass


class Budget:
    def __init__(
        self,
        log_path: Path = DEFAULT_SPEND_LOG,
        price_path: Path = DEFAULT_PRICE_FILE,
        *,
        total_cap: int = TOTAL_CAP,
        run_cap: int = RUN_CAP,
        override: bool = False,
        quiet: bool = False,
    ):
        self.log_path, self.price_path = Path(log_path), Path(price_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.total_cap, self.run_cap, self.override, self.quiet = total_cap, run_cap, override, quiet
        self._inflight: dict[str, int] = {}
        self._lock = asyncio.Lock()

    # --- log ---------------------------------------------------------------

    def entries(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]

    def log(self, event: str, **fields: Any) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if entry.get("billed") and not self.quiet:
            print(f"[fal $] {self.status_line(entry.get('run'))}", file=sys.stderr)

    @staticmethod
    def _is_billed(e: dict) -> bool:
        return e.get("billed") is True or e.get("billed") == "maybe"

    def billed(self, run: str | None = None) -> int:
        return sum(self._is_billed(e) and (run is None or e.get("run") == run) for e in self.entries())

    def price(self) -> tuple[float, bool]:
        if self.price_path.exists():
            return float(json.loads(self.price_path.read_text())["usd_per_call"]), True
        return EST_USD_PER_CALL, False

    def confirm_price(self, usd_per_call: float, note: str = "") -> None:
        self.price_path.write_text(json.dumps({
            "usd_per_call": usd_per_call, "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note,
        }, indent=1))

    def status_line(self, run: str | None = None) -> str:
        usd, confirmed = self.price()
        total = self.billed()
        s = f"total {total}/{self.total_cap} calls ≈ ${total * usd:.3f}"
        if run is not None:
            s = f"run {run!r} {self.billed(run)}/{self.run_cap}, " + s
        return s + ("" if confirmed else f"  (price UNCONFIRMED, est ${usd}/call; probe allows {PROBE_CALLS})")

    def pending_request_id(self, key: str) -> str | None:
        """A request_id submitted for this key whose result never made it into the cache."""
        rid = None
        for e in self.entries():
            if e.get("key") != key:
                continue
            if e["event"] == "submitted":
                rid = e.get("request_id")
            elif e["event"] in ("done", "recovered", "forgotten"):
                rid = None
        return rid

    # --- checks --------------------------------------------------------------

    def check(self, n_new: int, run: str) -> None:
        """Raise unless n_new more calls fit (counting calls already in flight)."""
        inflight_total = sum(self._inflight.values())
        total = self.billed() + inflight_total + n_new
        in_run = self.billed(run) + self._inflight.get(run, 0) + n_new
        _, confirmed = self.price()
        if not confirmed and total > PROBE_CALLS:
            raise PriceNotConfirmed(
                f"fal price not confirmed: only the {PROBE_CALLS}-frame probe is allowed "
                f"({self.billed()} billed so far, {n_new} more requested). Check the fal dashboard "
                "(usage & billing) for the real per-call cost, then run:\n"
                "    uv run python tools/fal_budget.py confirm-price <usd_per_call>"
            )
        if self.override:
            return
        if total > self.total_cap:
            raise BudgetError(f"total cap: {total} calls > {self.total_cap} (pass the override flag to exceed)")
        if in_run > self.run_cap:
            raise BudgetError(f"run cap: {in_run} calls in run {run!r} > {self.run_cap} (pass the override flag to exceed)")

    @contextlib.asynccontextmanager
    async def reserve(self, run: str):
        """Hold one call's worth of budget while a submit is in flight."""
        async with self._lock:
            self.check(1, run)
            self._inflight[run] = self._inflight.get(run, 0) + 1
        try:
            yield
        finally:
            async with self._lock:
                self._inflight[run] -= 1

    def confirm_batch(self, n_new: int, run: str, *, yes: bool = False) -> None:
        """Print the cost of the batch and ask; raises BudgetError if declined or over a cap."""
        usd, confirmed = self.price()
        print(
            f"about to send {n_new} new frames ≈ ${n_new * usd:.3f}, "
            f"total so far ${self.billed() * usd:.3f} ({self.billed()} calls)"
            + ("" if confirmed else f"  [price unconfirmed, est ${usd}/call]")
        )
        self.check(n_new, run)
        if yes or n_new == 0:
            return
        if not sys.stdin.isatty():
            raise BudgetError("refusing to send without confirmation (no terminal); pass --yes")
        if input("send? [y/N] ").strip().lower() not in ("y", "yes"):
            raise BudgetError("cancelled")
