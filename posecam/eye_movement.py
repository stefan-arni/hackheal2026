"""Eye movements: saccades (speed) and smoothness, from iris tracking.

Builds on eye_tracker.py, which gives each iris's position inside its eye
(h = across the eye, v = up/down, in eye-widths, measured between the eye
corners so head movement doesn't count as eye movement).

Gaze angle
  Both eyes are averaged (they move together in saccades and pursuit), then
  converted to degrees with a simple eyeball model: an iris shift of d mm
  means sin(angle) = d / eyeball radius. With a ~30 mm eye opening and ~12 mm
  radius, 0.1 eye-widths is about 14.5 deg. These are approximate (no per-person
  calibration), so amplitudes and speeds are estimates; compare people and
  sessions measured the same way.

Speed
  Velocity is measured over a fixed ~30 ms span rather than frame to frame, so
  landmark jitter doesn't turn into fake speed at high frame rates. Saccades are
  found with a velocity threshold (start above 80 deg/s, end below half that),
  and each one reports amplitude, duration, peak and mean speed, and direction.
  Real saccades peak at 300-500 deg/s and last 20-80 ms: a 30 fps webcam sees
  them in 1-2 frames and reads the peak low. Use 60+ fps for speed, ideally a
  120/240 fps (slow-motion) recording.

Smoothness
  Saccade test (look back and forth between two targets): a clean saccade lands
  in one jump; a "corrective" saccade is a small extra jump in the same direction
  right after (undershoot). Smoothness = % of saccades that landed in one jump.
  Pursuit test (follow a moving target): the eye should glide. Smoothness = % of
  the eye's path covered by smooth movement rather than catch-up saccades, plus
  catch-up saccades per second.

This is a screening/demo aid, not a validated clinical measurement.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import median

import numpy as np

TEST_MODES = {"saccades": "Saccades", "pursuit": "Smooth pursuit"}


@dataclass
class MovementConfig:
    eye_width_mm: float = 30.0         # eye opening, corner to corner (typical adult)
    eye_radius_mm: float = 12.0        # eyeball radius (typical adult)
    velocity_window_s: float = 0.03    # detection: speed measured over at least this span
    peak_window_s: float = 0.012       # peak speed: re-measured over this shorter span
    saccade_velocity: float = 80.0     # deg/s: a saccade starts above this
    end_ratio: float = 0.5             # ...and ends below this fraction of it
    min_amplitude_deg: float = 2.0     # smaller jumps are noise / microsaccades
    min_saccade_s: float = 0.015       # shorter than this is a noise spike (real ones last 20+ ms)
    max_saccade_s: float = 0.25        # longer than this isn't one saccade
    max_gap_s: float = 0.12            # a longer gap (blink, lost face) breaks the trace
    corrective_window_s: float = 0.3   # correction must start this soon after a saccade
    corrective_max_ratio: float = 0.5  # ...be at most this fraction of its size, same direction
    primary_min_deg: float = 5.0       # saccade test: jumps this big are target-to-target
    test_duration_s: float = 10.0
    center_window_s: float = 1.0       # "straight ahead" = median of the first second, then fixed
    max_head_yaw_change: float = 0.12  # warn if the head turns more than this during a test


def iris_to_degrees(offset_eyewidths: float, cfg: MovementConfig) -> float:
    s = offset_eyewidths * cfg.eye_width_mm / cfg.eye_radius_mm
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


@dataclass
class GazeTracker:
    """Per-frame gaze angle, velocity and saccade detection."""
    cfg: MovementConfig = field(default_factory=MovementConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self._center: list = []                # (h, v) samples for the straight-ahead estimate
        self._center_t0: float | None = None
        self.center: tuple | None = None       # fixed once learned
        self._trace: deque = deque()           # (t, x_deg, y_deg) for the speed window
        self._recent: deque = deque()          # (t, x_deg, y_deg) last ~0.5 s, for peak speed
        self._last_t: float | None = None
        self._in_sacc = False
        self._sacc: dict | None = None
        self.last_velocity: float | None = None

    def _center_hv(self, t, h, v):
        """Straight ahead = median of the first second, then FIXED. (A running
        center would jump between the two targets in a back-and-forth task and
        look like a saccade.) Speeds and sizes only use differences, so an
        imperfect center barely matters."""
        if self.center is not None:
            return self.center
        if self._center_t0 is None:
            self._center_t0 = t
        self._center.append((h, v))
        c = (median(p[0] for p in self._center), median(p[1] for p in self._center))
        if t - self._center_t0 >= self.cfg.center_window_s:
            self.center = c
        return c

    def update(self, t: float, eyes: dict | None) -> dict:
        """eyes: {"right": {"h", "v"}, "left": {"h", "v"}} or None (blink / no face).

        Returns {"gaze_deg", "velocity_dps", "in_saccade", "saccade"} where
        "saccade" is the finished saccade on the frame it ends, else None.
        """
        cfg = self.cfg
        out = {"gaze_deg": None, "velocity_dps": None, "in_saccade": self._in_sacc, "saccade": None}
        if eyes is None:
            return out
        if self._last_t is not None and t - self._last_t > cfg.max_gap_s:
            # blink or dropout: start a fresh trace, drop any half-seen saccade
            self._trace.clear()
            self._recent.clear()
            self._in_sacc, self._sacc = False, None
        self._last_t = t

        h = (eyes["right"]["h"] + eyes["left"]["h"]) / 2
        v = (eyes["right"]["v"] + eyes["left"]["v"]) / 2
        h0, v0 = self._center_hv(t, h, v)
        x, y = iris_to_degrees(h - h0, cfg), iris_to_degrees(v - v0, cfg)
        out["gaze_deg"] = [round(x, 2), round(y, 2)]

        self._trace.append((t, x, y))
        self._recent.append((t, x, y))
        while self._recent and t - self._recent[0][0] > 0.5:
            self._recent.popleft()
        while len(self._trace) > 2 and t - self._trace[1][0] >= cfg.velocity_window_s:
            self._trace.popleft()
        t0, x0, y0 = self._trace[0]
        if t - t0 < 1e-6:
            return out
        vel = math.hypot(x - x0, y - y0) / (t - t0)
        out["velocity_dps"] = round(vel, 1)
        self.last_velocity = vel

        if not self._in_sacc:
            if vel >= cfg.saccade_velocity:
                self._in_sacc = True
                self._sacc = {"t_start": t0, "start": (x0, y0), "peak": vel}
        else:
            self._sacc["peak"] = max(self._sacc["peak"], vel)
            if vel < cfg.saccade_velocity * cfg.end_ratio:
                s = self._sacc
                self._in_sacc, self._sacc = False, None
                # the movement ended where the current (quiet) speed window starts
                t_end = self._trace[0][0]
                dur = t_end - s["t_start"] if t_end > s["t_start"] else t - s["t_start"]
                # size: average a few frames at each end so landmark jitter doesn't add to it
                sx, sy = self._mean_pos(s["t_start"] - cfg.velocity_window_s, s["t_start"])
                ex, ey = self._mean_pos(t_end, t)
                dx, dy = ex - sx, ey - sy
                amp = math.hypot(dx, dy)
                if amp >= cfg.min_amplitude_deg and cfg.min_saccade_s <= dur <= cfg.max_saccade_s:
                    out["saccade"] = {
                        "t_start": round(s["t_start"], 4),
                        "t_end": round(t_end, 4),
                        "duration_ms": round(dur * 1000, 1),
                        "amplitude_deg": round(amp, 2),
                        "peak_velocity_dps": round(max(s["peak"], self._peak_speed(s["t_start"], t)), 1),
                        "mean_velocity_dps": round(amp / dur, 1) if dur > 0 else None,
                        "dx_deg": round(dx, 2),
                        "dy_deg": round(dy, 2),
                        "direction": _direction(dx, dy),
                    }
        out["in_saccade"] = self._in_sacc
        return out


    def _mean_pos(self, t_from: float, t_to: float) -> tuple[float, float]:
        pts = [p for p in self._recent if t_from - 1e-9 <= p[0] <= t_to + 1e-9]
        if not pts:
            pts = [self._recent[-1]]
        return (float(np.mean([p[1] for p in pts])), float(np.mean([p[2] for p in pts])))

    def _peak_speed(self, t_from: float, t_to: float) -> float:
        """Peak speed during a saccade over the shortest span the frame rate allows
        (>= peak_window_s, at least one frame). At 30 fps this is one frame, so
        the peak still reads low; at 120+ fps it's close to the true peak."""
        pts = [p for p in self._recent if t_from - self.cfg.velocity_window_s <= p[0] <= t_to]
        best = 0.0
        j = 0
        for i in range(1, len(pts)):
            while j + 1 < i and pts[i][0] - pts[j + 1][0] >= self.cfg.peak_window_s:
                j += 1
            dt = pts[i][0] - pts[j][0]
            if dt > 1e-6:
                best = max(best, math.hypot(pts[i][1] - pts[j][1], pts[i][2] - pts[j][2]) / dt)
        return best


def _direction(dx, dy) -> str:
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"      # image right / left
    return "down" if dy > 0 else "up"


@dataclass
class EyeMovementTest:
    """A timed saccade or pursuit test, built on GazeTracker output."""
    cfg: MovementConfig = field(default_factory=MovementConfig)

    def __post_init__(self):
        self.last_result: dict | None = None
        self._clear()

    def _clear(self):
        self.mode: str | None = None
        self.duration: float = self.cfg.test_duration_s
        self._t0: float | None = None
        self._samples: list[tuple] = []     # (t, x, y, vel, in_saccade)
        self._saccades: list[dict] = []
        self._yaws: list[float] = []
        self._frames = 0
        self._gaps = 0
        self._pending: list[dict] = []

    @property
    def running(self) -> bool:
        return self.mode is not None

    def start(self, mode: str, duration: float | None = None):
        if mode not in TEST_MODES:
            raise ValueError(f"mode must be one of {list(TEST_MODES)}")
        if duration is not None and not (1.0 <= float(duration) <= 120.0):
            raise ValueError("duration must be between 1 and 120 seconds")
        self._clear()
        self.mode = mode
        self.duration = float(duration) if duration is not None else self.cfg.test_duration_s
        self._pending.append({"kind": "eye_test_started", "mode": mode, "duration_s": self.duration})

    def cancel(self):
        mode = self.mode
        self._clear()
        if mode:
            self._pending.append({"kind": "eye_test_cancelled", "mode": mode})

    def update(self, t: float, gaze: dict | None, head_yaw: float | None) -> dict:
        events, self._pending = self._pending, []
        if not self.running:
            return {"running": False, "events": events, "last_result": self.last_result}
        if self._t0 is None:
            self._t0 = t
        self._frames += 1
        if gaze is None or gaze["gaze_deg"] is None:
            self._gaps += 1
        else:
            x, y = gaze["gaze_deg"]
            self._samples.append((t, x, y, gaze["velocity_dps"], gaze["in_saccade"]))
            if gaze.get("saccade"):
                self._saccades.append(gaze["saccade"])
                events.append({"kind": "saccade", **gaze["saccade"]})
        if head_yaw is not None:
            self._yaws.append(head_yaw)

        elapsed = t - self._t0
        if elapsed >= self.duration:
            self.last_result = self.summary()
            events.append({"kind": "eye_test_done", **self.last_result})
            self._clear()
            return {"running": False, "events": events, "last_result": self.last_result}
        return {"running": True, "mode": self.mode, "time_left": round(self.duration - elapsed, 1),
                "saccades": len(self._saccades), "events": events}

    # ---------------- results ----------------

    def summary(self) -> dict:
        cfg = self.cfg
        s = self._samples
        span = (s[-1][0] - s[0][0]) if len(s) >= 2 else 0.0
        fps = (len(s) - 1) / span if span > 0 else 0.0
        quality = {
            "samples": len(s),
            "effective_fps": round(fps, 1),
            "tracked_pct": round(100 * len(s) / max(1, self._frames)),
            "warnings": [],
        }
        if fps and fps < 50:
            quality["warnings"].append(
                f"Only {fps:.0f} frames/s: saccade speeds will read low. Use 60+ fps for speed.")
        if self._yaws and max(self._yaws) - min(self._yaws) > cfg.max_head_yaw_change:
            quality["warnings"].append("The head turned during the test; keep the head still.")
        if quality["tracked_pct"] < 80:
            quality["warnings"].append("Eyes were lost in many frames (blinks, light, face too small).")

        sacc = self._saccades
        speed = {
            "count": len(sacc),
            "per_second": round(len(sacc) / span, 2) if span > 0 else None,
            "peak_velocity_dps": _stat([x["peak_velocity_dps"] for x in sacc]),
            "mean_velocity_dps": _stat([x["mean_velocity_dps"] for x in sacc if x["mean_velocity_dps"]]),
            "amplitude_deg": _stat([x["amplitude_deg"] for x in sacc]),
            "duration_ms": _stat([x["duration_ms"] for x in sacc]),
        }
        result = {"mode": self.mode, "label": TEST_MODES[self.mode], "duration_s": round(span, 2),
                  "speed": speed, "quality": quality, "saccades": sacc}

        if self.mode == "saccades":
            primaries, corrective = self._classify_saccades(sacc)
            single = [p for p in primaries if not p["_corrected"]]
            peaks = [p["peak_velocity_dps"] for p in primaries]
            cv = (float(np.std(peaks) / np.mean(peaks)) if len(peaks) >= 2 and np.mean(peaks) > 0
                  else None)
            result["smoothness"] = {
                "score": round(100 * len(single) / len(primaries)) if primaries else None,
                "primary_saccades": len(primaries),
                "corrective_saccades": corrective,
                "single_step_pct": round(100 * len(single) / len(primaries)) if primaries else None,
                "peak_velocity_cv": round(cv, 2) if cv is not None else None,
                "explanation": "score = % of target-to-target jumps that landed in one saccade "
                               "(no small corrective jump after).",
            }
            for p in primaries:
                p.pop("_corrected", None)
        else:
            smooth, smooth_v = self._smooth_path(s)
            sacc_path = sum(x["amplitude_deg"] for x in sacc)
            total = smooth + sacc_path
            frac = smooth / total if total > 0 else None
            result["smoothness"] = {
                "score": round(100 * frac) if frac is not None else None,
                "smooth_path_pct": round(100 * frac, 1) if frac is not None else None,
                "catch_up_saccades": len(sacc),
                "catch_up_per_second": speed["per_second"],
                "pursuit_speed_dps": _stat(smooth_v),
                "explanation": "score = % of the eye's path covered by smooth tracking "
                               "rather than catch-up saccades.",
            }
        return result

    def _smooth_path(self, s):
        """Distance the eye covered smoothly (outside saccades) and its speeds,
        on 50 ms averaged positions so landmark jitter doesn't count as movement."""
        cfg = self.cfg
        if len(s) < 2:
            return 0.0, []
        bin_s = 0.05
        t0 = s[0][0]
        bins: dict[int, list] = {}
        for t, x, y, _, sac in s:
            bins.setdefault(int((t - t0) / bin_s), []).append((t, x, y, sac))
        pts = []
        for k in sorted(bins):
            b = bins[k]
            pts.append((k, float(np.mean([p[0] for p in b])), float(np.mean([p[1] for p in b])),
                        float(np.mean([p[2] for p in b])), any(p[3] for p in b)))
        path, speeds = 0.0, []
        for (k1, t1, x1, y1, s1), (k2, t2, x2, y2, s2) in zip(pts, pts[1:]):
            if k2 != k1 + 1 or s1 or s2 or t2 - t1 > cfg.max_gap_s + bin_s:
                continue
            d = math.hypot(x2 - x1, y2 - y1)
            path += d
            speeds.append(d / (t2 - t1))
        return path, speeds

    def _classify_saccades(self, sacc):
        """Split into target-to-target (primary) saccades and small corrective ones."""
        cfg = self.cfg
        primaries, corrective, last = [], 0, None
        for x in sacc:
            if last is not None and x["t_start"] - last["t_end"] <= cfg.corrective_window_s:
                same_dir = x["dx_deg"] * last["dx_deg"] + x["dy_deg"] * last["dy_deg"] > 0
                if same_dir and x["amplitude_deg"] <= cfg.corrective_max_ratio * last["amplitude_deg"]:
                    corrective += 1
                    if last in primaries:
                        last["_corrected"] = True
                    continue
            if x["amplitude_deg"] >= cfg.primary_min_deg:
                primaries.append({**x, "_corrected": False})
                last = primaries[-1]
            else:
                last = x
        return primaries, corrective

    def handle_command(self, msg: dict) -> list[dict]:
        kind = msg.get("type")
        try:
            if kind == "eye_test_start":
                self.start(msg.get("mode", ""), msg.get("duration"))
            elif kind == "eye_test_cancel":
                self.cancel()
            elif kind != "eye_test_status":
                return [{"type": "error", "error": f"unknown command {kind!r}", "command": kind}]
        except (ValueError, TypeError) as e:
            return [{"type": "error", "error": str(e), "command": kind}]
        return [{"type": "ack", "command": kind, "running": self.running, "mode": self.mode,
                 "last_result": self.last_result}]


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"median": round(float(np.median(vals)), 1), "max": round(float(np.max(vals)), 1),
            "min": round(float(np.min(vals)), 1)}


def config_from_dict(d: dict) -> MovementConfig:
    return MovementConfig(**{k: v for k, v in d.items() if k in asdict(MovementConfig())})
