"""Pose analyzer + the add-ons that run on its landmarks (balance, duck, BESS, sway).

The pose server and the --local test client both use this, so they behave
the same. Each add-on can be turned off independently.
"""

from __future__ import annotations

import logging
import time

from balance import BalanceConfig, BalanceMonitor, FeetState, balance_events, foot_heights
from bess import BessConfig, BessSession, bess_messages
from duck import DuckConfig, DuckDetector
from sway import SwayConfig, SwaySession, SwayTracker, sway_messages

log = logging.getLogger("pose-server")


class PosePipeline:
    def __init__(self, pose_analyzer, balance: BalanceConfig | None = None,
                 duck: DuckConfig | None = None, bess: BessConfig | None = None,
                 bess_eyes: str = "off", eye_closure=None, sway: SwayConfig | None = None):
        """bess_eyes: "off" (default) = eyes not tracked or scored, "auto" = check
        eyes with the face model, "manual" = only via bess_mark commands from the
        app. `eye_closure` can be injected (tests)."""
        if bess_eyes not in ("off", "auto", "manual"):
            raise ValueError("bess_eyes must be 'off', 'auto' or 'manual'")
        self.pose = pose_analyzer
        self.balance = BalanceMonitor(balance) if balance is not None else None
        self.feet = FeetState()          # each foot up / down, every frame
        self.duck = DuckDetector(duck) if duck is not None else None
        self.bess = None
        self.eye_closure = None
        self.sway = SwaySession(sway) if sway is not None else None
        # sway measured during each BESS stance (velocity, area), per stance
        self._bess_sway = SwayTracker(sway) if bess is not None else None
        self.bess_sway: dict[str, dict | None] = {}
        if bess is not None:
            bess.track_eyes = bess_eyes != "off"
            self.bess = BessSession(bess)
            if bess_eyes == "auto":
                if eye_closure is None:
                    from bess import EyeClosure
                    eye_closure = EyeClosure(bess.eyes_open_below)
                self.eye_closure = eye_closure

    def process_bgr(self, frame, t: float | None = None, meta: dict | None = None) -> dict:
        """`meta` is the frame's metadata from ws_server: its capture time
        (`timestamp_ms`, used as the clock when sent, so network delays don't
        distort speeds) and an optional depth map (`_depth`, `_intrinsics`)."""
        meta = meta or {}
        res = self.pose.process_bgr(frame)
        out = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        if t is None:
            ts = meta.get("timestamp_ms")
            t = ts / 1000.0 if isinstance(ts, (int, float)) else time.monotonic()
        h, w = frame.shape[:2]
        detected = bool(out.get("detected"))

        if meta.get("_depth") is not None:
            from depth import ankle_depths, torso_point
            tp = torso_point(out["landmarks"], meta["_depth"], meta["_intrinsics"]) if detected else None
            out["depth"] = {"camera": meta.get("camera"), **(tp or {"torso_m": None}),
                            "ankles_m": ankle_depths(out["landmarks"], meta["_depth"]) if detected else None}

        out["feet"] = self.feet.update(
            foot_heights(out["landmarks"], out.get("world_landmarks"), w, h, 0.3) if detected else None)

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
            self._track_bess_sway(t, out, w, h)

        if self.sway is not None:
            out["sway"] = self.sway.update(t, out, w, h)
        return out

    def _track_bess_sway(self, t, out, w, h):
        b, tr = out["bess"], self._bess_sway
        if b["phase"] == "countdown":
            tr.reset()
        elif b["phase"] == "running":
            tr.update(t, out, w, h)
            b["sway_cm"] = tr.live_cm()
        for ev in b.get("events", []):
            if ev["kind"] == "bess_done":
                ev["sway"] = self.bess_sway[ev["stance"]] = tr.metrics()
                tr.reset()
                ev["session"]["sway"] = dict(self.bess_sway)
        b["session"]["sway"] = dict(self.bess_sway)

    def _balance_paused_for(self) -> str | None:
        """The BESS stance or sway test that pauses single-leg balance tracking, if any."""
        if self.sway is not None and self.sway.phase in ("countdown", "running"):
            return "sway"
        b = self.bess
        if b is not None and b.phase in ("countdown", "running") and b.stance in ("double", "tandem"):
            return b.stance
        return None

    def handle_command(self, msg: dict) -> list[dict]:
        kind = str(msg.get("type", ""))
        if kind.startswith("sway_"):
            if self.sway is None:
                return [{"type": "error", "error": "sway tests are turned off on this server",
                         "command": kind}]
            if kind == "sway_start" and self.bess is not None and self.bess.phase != "idle":
                return [{"type": "error", "error": "a BESS test is running", "command": kind}]
            return self.sway.handle_command(msg)
        if (kind == "bess_start" and self.sway is not None and self.sway.phase != "idle"):
            return [{"type": "error", "error": "a sway test is running", "command": kind}]
        if self.bess is None:
            return [{"type": "error", "error": "BESS is turned off on this server",
                     "command": msg.get("type")}]
        if kind == "bess_reset":
            self.bess_sway = {}
        replies = self.bess.handle_command(msg)
        for r in replies:
            if r.get("type") == "ack":
                r["sway"] = dict(self.bess_sway)
        return replies

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
            + bess_messages(result, frame_id) + sway_messages(result, frame_id))
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
            log.info("BESS %s: scoring started%s", m["stance"],
                     "".join(f"\n    warning: {w}" for w in m.get("warnings", [])))
        elif m["kind"] == "bess_error":
            log.info("BESS error at %.1fs: %s%s", m["t"], m["label"],
                     "" if m["counted"] else f" (not counted: {m['not_counted_reason']})")
        elif m["kind"] == "bess_done":
            s = m["session"]["scores"]
            log.warning("BESS %s done: %d errors | double %s, tandem %s, single %s | total %d",
                        m["stance"], m["errors"], s["double"], s["tandem"], s["single"],
                        m["session"]["total"])
        elif m["kind"] == "sway_started":
            log.info("SWAY %s: get in position", m["test"])
        elif m["kind"] == "sway_running":
            log.info("SWAY %s: recording (%s)%s", m["test"], m["mode"],
                     "".join(f"\n    warning: {w}" for w in m.get("warnings", [])))
        elif m["kind"] == "sway_done":
            mt = m.get("metrics") or {}
            log.warning("SWAY %s done (%s): %d errors | velocity %s cm/s, path %s cm, area %s cm2, "
                        "rms ML %s / AP %s cm", m["test"], m["mode"], m["errors"],
                        mt.get("mean_velocity_cm_s"), mt.get("path_length_cm"), mt.get("area_95_cm2"),
                        mt.get("rms_ml_cm"), mt.get("rms_ap_cm"))
        elif m["kind"] == "sway_error":
            log.warning("SWAY %s at %.1fs: %s%s (errors: %d)", "+1" if m["counted"] else "  ",
                        m["t"], m["label"],
                        "" if m["counted"] else f" (not counted: {m['not_counted_reason']})",
                        m["errors"])
        elif m["kind"] == "sway_lean":
            if m["event"] == "start":
                log.warning("SWAY lean alert: %.1f deg to the %s (limit %.0f) at %.1fs",
                            m["lean_deg"], m["side"], m["limit_deg"], m["t"])
            else:
                log.info("SWAY lean back within limit at %.1fs", m["t"])
        elif m["kind"] in ("sway_failed", "sway_cancelled"):
            log.warning("SWAY %s %s %s", m["test"], m["kind"].split("_")[1], m.get("reason", ""))
        elif m["kind"] in ("bess_failed", "bess_cancelled"):
            log.warning("BESS %s %s %s", m["stance"], m["kind"].split("_")[1], m.get("reason", ""))
    return msgs
