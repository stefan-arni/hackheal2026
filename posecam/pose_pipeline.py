"""Pose analyzer + the add-ons that run on its landmarks (balance, duck, BESS).

The pose server and the --local test client both use this, so they behave
the same. Each add-on can be turned off independently.
"""

from __future__ import annotations

import logging
import time

from balance import BalanceConfig, BalanceMonitor, balance_events, foot_heights
from bess import BessConfig, BessSession, bess_messages
from duck import DuckConfig, DuckDetector

log = logging.getLogger("pose-server")


class PosePipeline:
    def __init__(self, pose_analyzer, balance: BalanceConfig | None = None,
                 duck: DuckConfig | None = None, bess: BessConfig | None = None,
                 bess_eyes: str = "off", eye_closure=None):
        """bess_eyes: "off" (default) = eyes not tracked or scored, "auto" = check
        eyes with the face model, "manual" = only via bess_mark commands from the
        app. `eye_closure` can be injected (tests)."""
        if bess_eyes not in ("off", "auto", "manual"):
            raise ValueError("bess_eyes must be 'off', 'auto' or 'manual'")
        self.pose = pose_analyzer
        self.balance = BalanceMonitor(balance) if balance is not None else None
        self.duck = DuckDetector(duck) if duck is not None else None
        self.bess = None
        self.eye_closure = None
        if bess is not None:
            bess.track_eyes = bess_eyes != "off"
            self.bess = BessSession(bess)
            if bess_eyes == "auto":
                if eye_closure is None:
                    from bess import EyeClosure
                    eye_closure = EyeClosure(bess.eyes_open_below)
                self.eye_closure = eye_closure

    def process_bgr(self, frame, t: float | None = None) -> dict:
        res = self.pose.process_bgr(frame)
        out = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        t = time.monotonic() if t is None else t
        h, w = frame.shape[:2]
        detected = bool(out.get("detected"))

        if self.balance is not None:
            paused_for = self._balance_paused_for()
            if paused_for:
                # Feet together / tandem: both feet stay on the floor, so the
                # single-leg "raised foot touched down" tracker is paused.
                out["balance"] = self.balance.paused_status(paused_for)
            else:
                heights = (foot_heights(out["landmarks"], out.get("world_landmarks"), w, h,
                                        self.balance.cfg.min_visibility) if detected else None)
                out["balance"] = self.balance.update(t, heights)
                if heights:
                    out["balance"]["foot_points_px"] = heights["lowest_px"]

        if self.duck is not None:
            mode, value = (self.duck.measure(out["landmarks"], out.get("world_landmarks"), w, h)
                           if detected else (None, None))
            out["duck"] = self.duck.update(t, mode, value)

        if self.bess is not None:
            eyes_open = None
            if (self.eye_closure is not None and detected
                    and self.bess.phase in ("countdown", "running")):
                eyes_open = self.eye_closure.is_open(frame, out["landmarks"])
            out["bess"] = self.bess.update(t, out, w, h, eyes_open)
        return out

    def _balance_paused_for(self) -> str | None:
        """The BESS stance that pauses single-leg balance tracking, if any."""
        b = self.bess
        if b is not None and b.phase in ("countdown", "running") and b.stance in ("double", "tandem"):
            return b.stance
        return None

    def handle_command(self, msg: dict) -> list[dict]:
        if self.bess is None:
            return [{"type": "error", "error": "BESS is turned off on this server",
                     "command": msg.get("type")}]
        return self.bess.handle_command(msg)

    def recalibrate(self):
        """'recalibrate' from the client resets balance stats and the duck count."""
        if self.balance is not None:
            self.balance.reset()
        if self.duck is not None:
            self.duck.reset()

    def close(self):
        self.pose.close()
        if self.eye_closure is not None:
            self.eye_closure.close()


def duck_events(result: dict, frame_id=None) -> list[dict]:
    d = result.get("duck") or {}
    if d.get("event") == "duck":
        return [{"type": "event", "kind": "duck", "frame_id": frame_id, "count": d["count"],
                 "drop": d["drop"], "mode": d["mode"]}]
    return []


def pose_events(result: dict, frame_id=None) -> list[dict]:
    """All extra messages for the pose server, logged as they go out."""
    msgs = (balance_events(result, frame_id) + duck_events(result, frame_id)
            + bess_messages(result, frame_id))
    for m in msgs:
        if m["kind"] == "balance_started":
            log.info("balance started: standing on %s foot", m["standing_foot"])
        elif m["kind"] == "foot_touchdown":
            log.warning("FOOT TOUCHDOWN: %s foot after %.1fs (touch #%d)",
                        m["foot"], m["held_s"], m["touch_count"])
        elif m["kind"] == "duck":
            log.info("DUCK #%d - quack! (%s, drop %.2f)", m["count"], m["mode"], m["drop"])
        elif m["kind"] == "bess_started":
            log.info("BESS %s: get in position (non-dominant %s)", m["stance"], m["nondominant"])
        elif m["kind"] == "bess_running":
            log.info("BESS %s: 20 s started%s", m["stance"],
                     "".join(f"\n    warning: {w}" for w in m.get("warnings", [])))
        elif m["kind"] == "bess_error":
            log.info("BESS error at %.1fs: %s%s", m["t"], m["label"],
                     "" if m["counted"] else f" (not counted: {m['not_counted_reason']})")
        elif m["kind"] == "bess_done":
            s = m["session"]["scores"]
            log.warning("BESS %s done: %d errors | double %s, tandem %s, single %s | total %d",
                        m["stance"], m["errors"], s["double"], s["tandem"], s["single"],
                        m["session"]["total"])
        elif m["kind"] in ("bess_failed", "bess_cancelled"):
            log.warning("BESS %s %s %s", m["stance"], m["kind"].split("_")[1], m.get("reason", ""))
    return msgs
