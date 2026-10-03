"""Single-leg balance: detects when the raised foot touches the ground.

Works on the 33 MediaPipe pose landmarks (from pose_analyzer.py). For each foot
we take its lowest point (ankle, heel or toe). The lower foot is the standing
foot and defines the ground; the other foot's height above it says whether it
is raised.

Height is measured in one of two ways:
  - "world" (default): MediaPipe's 3D world landmarks, in meters. Not fooled by
    perspective (a foot placed further back looks higher in the image).
  - "image": 2D pixels, divided by the standing leg's length so it doesn't
    depend on how far the person is from the camera.

Floor calibration: whenever both feet are down for `calibrate_s`, the floor is
  (re)learned. In 3D, each foot's resting offset is recorded and subtracted
  (MediaPipe often puts one foot's landmarks a centimeter or two higher than the
  other). In 2D, a floor line through both resting feet is stored, and the raised
  foot is measured from that line instead of from the standing foot, so jitter
  in the standing foot no longer shows up as height on the raised one. Until the
  first calibration, heights are measured relative to the other foot.

State machine:
  idle       -> both feet down (floor calibrates here)
  lifting    -> one foot is above the lift threshold, waiting `lift_hold_s`
  balancing  -> on one leg; timer running
  touchdown  -> event fired when the raised foot drops below the touch
                threshold; then back to idle and ready for the next attempt

The lift threshold is higher than the touch threshold (hysteresis) so a foot
hovering near the ground doesn't flicker between states.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

# Pose landmark indices
FEET = {
    "left": {"hip": 23, "ankle": 27, "heel": 29, "toe": 31},
    "right": {"hip": 24, "ankle": 28, "heel": 30, "toe": 32},
}
OTHER = {"left": "right", "right": "left"}


def _get(lm, key):
    return lm[key] if isinstance(lm, dict) else getattr(lm, key)


def foot_heights(landmarks, world_landmarks, width: int, height: int,
                 min_visibility: float = 0.3) -> dict | None:
    """Height of each foot above the lower foot. None if feet aren't visible.

    Returns {"world": {"left": m, "right": m}, "image": {"left": legs, "right": legs},
             "lowest_px": {"left": [x, y], "right": [x, y]}, "visibility": v}
    """
    vis = min(_get(landmarks[i], "visibility") or 0.0
              for f in FEET.values() for i in (f["ankle"], f["heel"], f["toe"]))
    if vis < min_visibility:
        return None

    out = {"visibility": round(vis, 3), "lowest_px": {}}

    # image space: y grows downward, so the lowest point has the largest y
    low_img = {}
    for side, f in FEET.items():
        pts = [(_get(landmarks[i], "x") * width, _get(landmarks[i], "y") * height)
               for i in (f["ankle"], f["heel"], f["toe"])]
        low = max(pts, key=lambda p: p[1])
        low_img[side] = low
        out["lowest_px"][side] = [round(low[0], 1), round(low[1], 1)]
    ground_img = max(p[1] for p in low_img.values())
    standing = max(low_img, key=lambda s: low_img[s][1])
    f = FEET[standing]
    hx, hy = _get(landmarks[f["hip"]], "x") * width, _get(landmarks[f["hip"]], "y") * height
    ax, ay = _get(landmarks[f["ankle"]], "x") * width, _get(landmarks[f["ankle"]], "y") * height
    leg = max(((hx - ax) ** 2 + (hy - ay) ** 2) ** 0.5, 1e-6)
    out["image"] = {s: (ground_img - p[1]) / leg for s, p in low_img.items()}
    out["leg_px"] = leg

    # world space: meters, y also points down
    if world_landmarks is not None:
        low_w = {s: max(_get(world_landmarks[i], "y") for i in (f["ankle"], f["heel"], f["toe"]))
                 for s, f in FEET.items()}
        ground_w = max(low_w.values())
        out["world"] = {s: ground_w - y for s, y in low_w.items()}
    return out


@dataclass
class BalanceConfig:
    mode: str = "world"            # "world" (meters) or "image" (leg-lengths)
    lift_threshold: float | None = None    # foot counts as raised above this
    touch_threshold: float | None = None   # ...and as down again below this
    lift_hold_s: float = 0.3       # raised this long before the balance timer starts
    smooth_frames: int = 3         # median filter: one noisy frame can't trigger a touch
    min_visibility: float = 0.3    # dark clothes/shoes score 0.35-0.5 on clearly visible feet
    calibrate: bool = True         # (2) learn the floor while both feet are down
    calibrate_s: float = 1.0       # ...over this long

    def __post_init__(self):
        if self.mode not in ("world", "image"):
            raise ValueError("mode must be 'world' or 'image'")
        defaults = {"world": (0.08, 0.03), "image": (0.10, 0.04)}[self.mode]
        if self.lift_threshold is None:
            self.lift_threshold = defaults[0]
        if self.touch_threshold is None:
            self.touch_threshold = defaults[1]
        if self.touch_threshold >= self.lift_threshold:
            raise ValueError("touch_threshold must be below lift_threshold")


@dataclass
class BalanceMonitor:
    cfg: BalanceConfig = field(default_factory=BalanceConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        """Clear the current attempt, the session stats and the floor calibration."""
        self.state = "idle"
        self.lifted_foot: str | None = None
        self._lift_since: float | None = None
        self.balance_start: float | None = None
        self._hist = {"left": deque(maxlen=self.cfg.smooth_frames),
                      "right": deque(maxlen=self.cfg.smooth_frames)}
        self._calib: deque = deque()                         # samples while both feet down
        self.floor: dict | None = None                       # learned floor
        self.touch_count = 0
        self.last_hold_s: float | None = None
        self.best_hold_s: float | None = None

    # ---------------- floor calibration ----------------

    def _calibrate(self, t: float, heights: dict):
        """Collect samples while both feet are down; refresh the floor from the
        last `calibrate_s` of them. Lifting a foot clears the samples."""
        cfg = self.cfg
        rel = heights[cfg.mode]
        both_down = max(rel.values()) < cfg.touch_threshold
        if not (cfg.calibrate and self.state == "idle" and both_down):
            self._calib.clear()
            return
        self._calib.append((t, heights))
        while self._calib and t - self._calib[0][0] > cfg.calibrate_s:
            self._calib.popleft()
        if t - self._calib[0][0] < cfg.calibrate_s * 0.9 or len(self._calib) < 5:
            return
        samples = [h for _, h in self._calib]
        floor = {"samples": len(samples)}
        if all("world" in h for h in samples):
            floor["world_offset"] = {s: median(h["world"][s] for h in samples) for s in FEET}
        pts = {s: (median(h["lowest_px"][s][0] for h in samples),
                   median(h["lowest_px"][s][1] for h in samples)) for s in FEET}
        (x1, y1), (x2, y2) = pts["left"], pts["right"]
        if abs(x2 - x1) > 1e-6:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = 0.0
        floor["line"] = {"slope": slope, "x0": x1, "y0": y1}
        self.floor = floor

    def _floor_heights(self, heights: dict) -> dict:
        """Heights above the learned floor (falls back to relative heights)."""
        rel = heights[self.cfg.mode]
        if self.floor is None:
            return dict(rel)
        if self.cfg.mode == "world" and "world_offset" in self.floor:
            off = self.floor["world_offset"]
            return {s: rel[s] - off[s] for s in FEET}
        if self.cfg.mode == "image" and "leg_px" in heights:
            ln = self.floor["line"]
            out = {}
            for s in FEET:
                x, y = heights["lowest_px"][s]
                floor_y = ln["y0"] + ln["slope"] * (x - ln["x0"])
                out[s] = (floor_y - y) / heights["leg_px"]
            return out
        return dict(rel)

    # ---------------- per frame ----------------

    def update(self, t: float, heights: dict | None) -> dict:
        """Feed one frame's foot_heights() (or None). Returns status; `event` is
        "balance_start" or "touchdown" on the frame where that happens."""
        cfg = self.cfg
        event = None
        out_heights = None
        if heights is None:
            skip = "feet_not_visible"
        else:
            skip = None
            self._calibrate(t, heights)
            h = self._floor_heights(heights)
            for s in ("left", "right"):
                self._hist[s].append(h[s])
            sm = {s: median(self._hist[s]) for s in ("left", "right")}
            out_heights = {s: round(v, 4) for s, v in sm.items()}
            raised = max(sm, key=sm.get)
            raised_h = sm[raised]

            if self.state in ("idle", "lifting"):
                if raised_h >= cfg.lift_threshold:
                    if self.state == "idle" or raised != self.lifted_foot:
                        self.state, self.lifted_foot, self._lift_since = "lifting", raised, t
                    if t - self._lift_since >= cfg.lift_hold_s:
                        self.state = "balancing"
                        self.balance_start = self._lift_since
                        event = "balance_start"
                else:
                    # a lift only counts while the foot stays above the lift threshold;
                    # dipping back below it restarts the attempt (no backdated timer)
                    self.state, self.lifted_foot, self._lift_since = "idle", None, None

            elif self.state == "balancing":
                if sm[self.lifted_foot] < cfg.touch_threshold:
                    held = t - self.balance_start
                    self.touch_count += 1
                    self.last_hold_s = round(held, 2)
                    self.best_hold_s = max(self.best_hold_s or 0.0, self.last_hold_s)
                    event = "touchdown"
                    touched = self.lifted_foot
                    self.state, self.lifted_foot, self.balance_start = "idle", None, None
                    self._lift_since = None
                    return self._status(t, event, skip, out_heights, touched_foot=touched)

        return self._status(t, event, skip, out_heights)

    def paused_status(self, reason: str) -> dict:
        """Status while tracking is paused (e.g. during a BESS feet-together or
        tandem test). The current attempt is dropped; session stats are kept."""
        self.state, self.lifted_foot, self.balance_start, self._lift_since = "idle", None, None, None
        for d in self._hist.values():
            d.clear()
        self._calib.clear()
        return {
            "state": "paused",
            "paused_for": reason,
            "event": None,
            "skip_reason": None,
            "mode": self.cfg.mode,
            "foot_heights": None,
            "floor_calibrated": self.floor is not None,
            "lifted_foot": None,
            "standing_foot": None,
            "balance_time_s": None,
            "touch_count": self.touch_count,
            "last_hold_s": self.last_hold_s,
            "best_hold_s": self.best_hold_s,
        }

    def _status(self, t, event, skip, heights, touched_foot=None) -> dict:
        out = {
            "state": self.state,
            "event": event,
            "skip_reason": skip,
            "mode": self.cfg.mode,
            "foot_heights": heights,
            "floor_calibrated": self.floor is not None,
            "lifted_foot": self.lifted_foot,
            "standing_foot": OTHER[self.lifted_foot] if self.lifted_foot else None,
            "balance_time_s": round(t - self.balance_start, 2) if self.balance_start is not None else None,
            "touch_count": self.touch_count,
            "last_hold_s": self.last_hold_s,
            "best_hold_s": self.best_hold_s,
        }
        if touched_foot:
            out["touched_foot"] = touched_foot
            out["held_s"] = self.last_hold_s
        return out


def balance_events(result: dict, frame_id=None) -> list[dict]:
    """Extra WebSocket messages for the pose server."""
    b = result.get("balance") or {}
    ev = b.get("event")
    if ev == "balance_start":
        return [{"type": "status", "kind": "balance_started", "frame_id": frame_id,
                 "lifted_foot": b["lifted_foot"], "standing_foot": b["standing_foot"]}]
    if ev == "touchdown":
        return [{"type": "alert", "kind": "foot_touchdown", "frame_id": frame_id,
                 "foot": b["touched_foot"], "held_s": b["held_s"],
                 "touch_count": b["touch_count"], "best_hold_s": b["best_hold_s"]}]
    return []


# --------------------------------------------------------------------------- #
# Drawing (laptop test client)
# --------------------------------------------------------------------------- #

def draw_balance(frame, b: dict | None):
    import time
    import cv2
    if not b:
        return frame
    h, w = frame.shape[:2]
    pts = b.get("foot_points_px") or {}
    for side, p in pts.items():
        lifted = side == b.get("lifted_foot")
        color = (0, 200, 255) if lifted else (255, 200, 0)
        cv2.circle(frame, (int(p[0]), int(p[1])), 9, color, 2)

    # ground line under the standing foot
    if pts:
        gy = int(max(p[1] for p in pts.values()))
        cv2.line(frame, (0, gy), (w, gy), (120, 120, 120), 1)

    state = b["state"]
    if state == "paused":
        names = {"double": "feet together", "tandem": "tandem", "sway": "sway"}
        msg = f"Single-leg tracking paused during {names.get(b.get('paused_for'), 'BESS')} test"
        cv2.putText(frame, msg, (15, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(frame, msg, (15, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return frame
    if b.get("skip_reason"):
        text, color = "Balance: feet not visible", (0, 0, 255)
    elif state == "balancing":
        text, color = (f"Balancing on {b['standing_foot']} foot: {b['balance_time_s']:.1f}s", (0, 220, 0))
    elif state == "lifting":
        text, color = f"{b['lifted_foot'].capitalize()} foot raised...", (0, 200, 255)
    else:
        text, color = "Both feet down: lift one foot to start", (200, 200, 200)
    stats = f"touches {b['touch_count']}"
    stats += "   floor: " + ("calibrated" if b.get("floor_calibrated") else "stand on both feet")
    if b.get("last_hold_s") is not None:
        stats += f"   last {b['last_hold_s']:.1f}s   best {b['best_hold_s']:.1f}s"
    y0 = h - 70
    for i, (s, c) in enumerate(((text, color), (stats, (255, 255, 255)))):
        cv2.putText(frame, s, (15, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, s, (15, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    # flash for ~1.5 s after a touchdown
    if b.get("event") == "touchdown":
        draw_balance.flash_until = time.time() + 1.5
        draw_balance.flash_text = f"FOOT DOWN ({b['touched_foot']}) after {b['held_s']:.1f}s"
    if time.time() < getattr(draw_balance, "flash_until", 0):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
        txt = draw_balance.flash_text
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (w - tw) // 2
        cv2.rectangle(frame, (x - 10, h // 3 - th - 12), (x + tw + 10, h // 3 + 12), (0, 0, 180), -1)
        cv2.putText(frame, txt, (x, h // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


class FeetState:
    """Is each foot down or up, every frame (never paused, unlike BalanceMonitor).
    Uses the height above the lower foot (3D metres, or 2D leg-lengths without
    world landmarks), the same lift / touch thresholds as BalanceMonitor, and a
    short median so a single noisy frame can't flip it."""

    THRESHOLDS = {"world": (0.08, 0.03), "image": (0.10, 0.04)}   # (up above, down below)

    def __init__(self, smooth_frames: int = 3):
        self._hist = {s: deque(maxlen=smooth_frames) for s in FEET}
        self.up = {s: False for s in FEET}

    def update(self, heights: dict | None) -> dict:
        if heights is None:
            for d in self._hist.values():
                d.clear()
            return {"state": "not_visible", "left": None, "right": None, "heights_cm": None}
        mode = "world" if "world" in heights else "image"
        lift, touch = self.THRESHOLDS[mode]
        for s in FEET:
            self._hist[s].append(heights[mode][s])
            h = median(self._hist[s])
            if self.up[s] and h < touch:
                self.up[s] = False
            elif not self.up[s] and h >= lift:
                self.up[s] = True
        if self.up["left"] and self.up["right"]:      # can't both be above the lower one
            low = min(FEET, key=lambda s: median(self._hist[s]))
            self.up[low] = False
        state = ("left_up" if self.up["left"] else "right_up" if self.up["right"] else "both_down")
        return {"state": state,
                "left": "up" if self.up["left"] else "down",
                "right": "up" if self.up["right"] else "down",
                "heights_cm": ({s: round(median(self._hist[s]) * 100, 1) for s in FEET}
                               if mode == "world" else None)}
