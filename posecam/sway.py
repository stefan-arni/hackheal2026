"""Postural sway tests: quiet stance, tandem stance and Romberg (eyes open / closed).

Body sway is tracked as the movement of an approximate centre of mass (COM): a point
between the hip midpoint and the shoulder midpoint (`com_hip_weight` of the way to
the hips), from a camera facing the person.

Two measuring modes, picked per trial from what the camera sends:

  depth  The iPhone sends a depth map (LiDAR or TrueDepth) with each frame. The
         torso's distance is read from the depth map, so both directions are in
         real centimetres: side-to-side (ML, medio-lateral) and front-back
         (AP, antero-posterior = toward / away from the camera).
  2d     Plain video. Only side-to-side sway can be seen. Pixels are converted to
         centimetres with MediaPipe's metric 3D skeleton (trunk length in metres
         / trunk length in pixels). Front-back sway, sway area and the direction
         ratio need depth and are reported as null.

Test flow (like BESS): `start(test)` -> countdown (get into position; the last
`baseline_s` of it records the start position) -> `duration_s` of recording ->
`sway_done` result.

Errors (BESS-style points, lower is better). The trial keeps going; each balance
error is one point, counted once per event (it has to clear before it counts
again), errors starting within `error_merge_s` of each other count once, and a
trial scores at most `max_errors`:

  step              an ankle moved `step_threshold` leg-lengths from where it was
                    (left/right swaps by the model are ignored). Once the feet are
                    still again, that becomes the new position.
  lean              sideways trunk lean past `lean_limit_deg`
  drift             centre of mass more than `drift_limit_cm` from the start
  out_of_position   any of the above held for more than `out_of_position_s`

Metrics (standard posturography, Prieto et al. 1996), after resampling to a fixed
rate and a light zero-phase smoothing (`smooth_s`) that removes landmark jitter:

  path_length_cm       total distance the COM travelled
  mean_velocity_cm_s   path length / duration (also per direction: ml_, ap_)
  rms_ml_cm, rms_ap_cm root-mean-square distance from the mean position
  range_ml_cm, ...     max - min
  area_95_cm2          95% confidence ellipse area (depth only)
  directional          biggest drift from the start position forward / back /
                       left / right (the subject's left and right, facing the
                       camera), the main sway axis, and which direction dominates
  trunk_lean           sideways trunk lean (hip-mid -> shoulder-mid, relative to
                       the start posture): max to each side, and how often / how
                       long it went past `lean_limit_deg`. Crossing the limit also
                       sends a live `sway_lean` alert.

Romberg: run `romberg_eo` and `romberg_ec`; the session then reports eyes-closed /
eyes-open ratios (velocity, path, area). Above 1 = more sway without vision.

This is a demo aid, not a validated clinical measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

ERRORS = {
    "step": "Step / stumble",
    "lean": "Trunk lean past limit",
    "drift": "Swayed too far",
    "out_of_position": "Out of position > 5 s",
}

TESTS = {
    "quiet": {"label": "Quiet stance",
              "instructions": "Feet hip-width apart, arms at your sides, eyes open, look ahead"},
    "tandem": {"label": "Tandem stance",
               "instructions": "One foot directly in front of the other, heel to toe, eyes open"},
    "romberg_eo": {"label": "Romberg, eyes open",
                   "instructions": "Feet together, arms at your sides, eyes open"},
    "romberg_ec": {"label": "Romberg, eyes closed",
                   "instructions": "Feet together, arms at your sides; close your eyes when the countdown ends"},
}

L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 23, 24, 27, 28
CHI2_95_2DOF = 5.991   # chi-square, 2 degrees of freedom, 95%


@dataclass
class SwayConfig:
    duration_s: float = 30.0
    countdown_s: float = 5.0
    baseline_s: float = 1.0        # last part of the countdown = start position
    smooth_s: float = 0.2          # zero-phase moving average before velocities
    com_hip_weight: float = 0.65   # COM: this far from shoulder-mid toward hip-mid
    step_threshold: float = 0.15   # ankle moved this many leg-lengths = step
    step_hold_s: float = 0.2
    max_lost_s: float = 2.0        # body unmeasurable this long while running -> fail
    max_wait_s: float = 10.0       # no start position this long after countdown -> fail
    min_visibility: float = 0.3
    min_duration_s: float = 1.0    # shortest recording that still gets metrics
    lean_limit_deg: float = 10.0   # sideways trunk lean past this = alert + error
    lean_hold_s: float = 0.3       # ...for at least this long
    drift_limit_cm: float = 10.0   # centre of mass this far from the start = error
    drift_hold_s: float = 0.2
    restable_s: float = 0.5        # after a step, feet still this long = new position
    out_of_position_s: float = 5.0 # any error held this long = one extra error
    error_merge_s: float = 0.5     # errors starting this close together count once
    max_errors: int = 10
    trail_hz: float = 10.0         # COM path sent with the result, for plotting


def _xy(lm, i, w, h):
    return np.array([lm[i]["x"] * w, lm[i]["y"] * h])


def _vis(lm, i):
    return lm[i].get("visibility") or 0.0


def measure(pose: dict, w: int, h: int, cfg: SwayConfig) -> dict | None:
    """COM position (pixels, plus metres if `pose["depth"]` has a torso point),
    pixel scale and ankle positions for one frame. None if the trunk isn't seen."""
    if not pose.get("detected"):
        return None
    lm = pose["landmarks"]
    if min(_vis(lm, i) for i in (L_SH, R_SH, L_HIP, R_HIP)) < 0.5:
        return None
    hip = (_xy(lm, L_HIP, w, h) + _xy(lm, R_HIP, w, h)) / 2
    sho = (_xy(lm, L_SH, w, h) + _xy(lm, R_SH, w, h)) / 2
    k = cfg.com_hip_weight
    com = k * hip + (1 - k) * sho
    trunk_px = float(np.linalg.norm(sho - hip))
    v = sho - hip
    m = {"com_px": com, "m_per_px": None, "ankles": None, "leg_px": None, "xyz": None,
         # sideways lean in the image, positive = shoulders to image-right (subject's left)
         "lean_deg": math.degrees(math.atan2(v[0], -v[1]))}

    world = pose.get("world_landmarks")
    if world is not None and trunk_px > 1:
        wp = lambda i: np.array([world[i]["x"], world[i]["y"], world[i]["z"]])  # noqa: E731
        trunk_m = float(np.linalg.norm((wp(L_SH) + wp(R_SH)) / 2 - (wp(L_HIP) + wp(R_HIP)) / 2))
        if trunk_m > 0.1:
            m["m_per_px"] = trunk_m / trunk_px

    if min(_vis(lm, i) for i in (L_ANK, R_ANK)) >= cfg.min_visibility:
        m["ankles"] = {"left": _xy(lm, L_ANK, w, h), "right": _xy(lm, R_ANK, w, h)}
        m["leg_px"] = float(np.mean([np.linalg.norm(_xy(lm, L_HIP, w, h) - _xy(lm, L_ANK, w, h)),
                                     np.linalg.norm(_xy(lm, R_HIP, w, h) - _xy(lm, R_ANK, w, h))]))

    d = pose.get("depth") or {}
    if d.get("torso_m"):
        m["xyz"] = np.array(d["torso_m"], dtype=float)
    return m


# ------------------------------- metrics ------------------------------------ #

def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-phase moving average (forward and backward), edges padded."""
    if n <= 1 or len(x) < 3:
        return x
    n = min(n, len(x))
    k = np.ones(n) / n
    pad = n
    y = np.convolve(np.pad(x, pad, mode="edge"), k, mode="same")[pad:-pad]
    y = np.convolve(np.pad(y[::-1], pad, mode="edge"), k, mode="same")[pad:-pad][::-1]
    return y


def sway_metrics(t, ml, ap=None, smooth_s: float = 0.2) -> dict | None:
    """Sway metrics from COM samples. `ml`/`ap` are offsets from the start
    position in metres (ml positive = image-right, ap positive = toward the
    camera); `ap` is None without depth. Returns centimetres."""
    t = np.asarray(t, float)
    if len(t) < 5 or t[-1] - t[0] <= 0:
        return None
    dt = float(np.median(np.diff(t)))
    fs = min(60.0, max(5.0, 1.0 / dt if dt > 0 else 30.0))
    tu = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    n = max(1, int(round(smooth_s * fs)))
    x = _smooth(np.interp(tu, t, np.asarray(ml, float)), n) * 100
    y = _smooth(np.interp(tu, t, np.asarray(ap, float)), n) * 100 if ap is not None else None
    dur = float(tu[-1] - tu[0]) if len(tu) > 1 else float(t[-1] - t[0])
    dur = max(dur, 1e-6)

    dx = np.diff(x)
    out = {
        "duration_s": round(dur, 2),
        "samples": int(len(t)),
        "rate_hz": round(fs, 1),
        "ml_velocity_cm_s": round(float(np.abs(dx).sum()) / dur, 2),
        "rms_ml_cm": round(float(np.sqrt(np.mean((x - x.mean()) ** 2))), 2),
        "range_ml_cm": round(float(x.max() - x.min()), 2),
        "mean_offset_ml_cm": round(float(x.mean()), 2),
        "ap_velocity_cm_s": None, "rms_ap_cm": None, "range_ap_cm": None,
        "mean_offset_ap_cm": None, "area_95_cm2": None,
    }
    if y is None:
        path = float(np.abs(dx).sum())
    else:
        dy = np.diff(y)
        path = float(np.hypot(dx, dy).sum())
        cov = np.cov(np.vstack([x, y]))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0, None)
        major = evecs[:, int(np.argmax(evals))]
        out.update({
            "ap_velocity_cm_s": round(float(np.abs(dy).sum()) / dur, 2),
            "rms_ap_cm": round(float(np.sqrt(np.mean((y - y.mean()) ** 2))), 2),
            "range_ap_cm": round(float(y.max() - y.min()), 2),
            "mean_offset_ap_cm": round(float(y.mean()), 2),
            "area_95_cm2": round(math.pi * CHI2_95_2DOF * math.sqrt(float(evals[0] * evals[1])), 2),
            "ellipse_axes_cm": [round(math.sqrt(CHI2_95_2DOF * float(e)), 2)
                                for e in sorted(evals, reverse=True)],
            # 0 deg = side-to-side, 90 deg = front-back
            "main_axis_deg": round(math.degrees(math.atan2(abs(major[1]), abs(major[0]))), 1),
        })
    out["path_length_cm"] = round(path, 2)
    out["mean_velocity_cm_s"] = round(path / dur, 2)
    out["directional"] = _directional(x, y)
    return out


def _directional(x, y) -> dict:
    """Biggest drift from the start position per direction (subject facing the
    camera, unmirrored: image-right is the subject's left), and which axis
    dominates (RMS ratio; 1.5x or more counts as dominant)."""
    d = {"left_cm": round(max(0.0, float(x.max())), 2),
         "right_cm": round(max(0.0, float(-x.min())), 2),
         "forward_cm": None, "backward_cm": None,
         "ml_ap_ratio": None, "dominant": None}
    ex = {"left": d["left_cm"], "right": d["right_cm"]}
    if y is not None:
        d["forward_cm"] = round(max(0.0, float(y.max())), 2)
        d["backward_cm"] = round(max(0.0, float(-y.min())), 2)
        ex.update(forward=d["forward_cm"], backward=d["backward_cm"])
        rml = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
        rap = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
        ratio = rml / rap if rap > 1e-9 else float("inf")
        d["ml_ap_ratio"] = round(ratio, 2) if math.isfinite(ratio) else None
        d["dominant"] = ("side-to-side" if ratio >= 1.5 else
                         "front-back" if ratio <= 1 / 1.5 else "no dominant direction")
    d["largest_drift"] = max(ex, key=ex.get)
    return d


def romberg_ratios(eo: dict | None, ec: dict | None) -> dict | None:
    """Eyes-closed / eyes-open ratios from two sway_done results."""
    if not eo or not ec or not eo.get("metrics") or not ec.get("metrics"):
        return None
    a, b = eo["metrics"], ec["metrics"]

    def ratio(k):
        if a.get(k) is None or b.get(k) is None or a[k] <= 0:
            return None
        return round(b[k] / a[k], 2)

    return {
        "velocity_ratio": ratio("mean_velocity_cm_s"),
        "path_ratio": ratio("path_length_cm"),
        "area_ratio": ratio("area_95_cm2"),
        "rms_ml_ratio": ratio("rms_ml_cm"),
        "rms_ap_ratio": ratio("rms_ap_cm"),
        "lost_balance_eyes_closed": bool(ec.get("lost_balance")),
        "lost_balance_eyes_open": bool(eo.get("lost_balance")),
        "errors_eyes_open": eo.get("errors"),
        "errors_eyes_closed": ec.get("errors"),
        # classic Romberg sign: can stand with eyes open but not with eyes closed
        "positive": bool(ec.get("lost_balance")) and not bool(eo.get("lost_balance")),
        "mode": "depth" if eo.get("mode") == ec.get("mode") == "depth" else "2d",
    }


# ------------------------------- session ------------------------------------ #

class SwaySession:
    """One test at a time; keeps the latest result per test for the session."""

    def __init__(self, cfg: SwayConfig | None = None):
        self.cfg = cfg or SwayConfig()
        self.results: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._idle()

    def _idle(self):
        self.phase = "idle"
        self.test: str | None = None
        self.duration_s = self.cfg.duration_s
        self.countdown_s = self.cfg.countdown_s
        self._t0 = self._run_t0 = None
        self._base: list[dict] = []
        self.ref: dict | None = None
        self.mode: str | None = None
        self._samples: list[tuple[float, float, float | None]] = []
        self._last_seen: float | None = None
        self._live = None
        self._frames = 0
        self.lean_limit = self.cfg.lean_limit_deg
        self.drift_limit = self.cfg.drift_limit_cm
        self._lean = {"max": 0.0, "min": 0.0, "episodes": 0, "over_s": 0.0,
                      "over_since": None, "alerting": False, "last_t": None, "now": None}
        # errors: per type, when its condition started and whether it's active
        self.errors = 0
        self.by_type = {k: 0 for k in ERRORS}
        self.log: list[dict] = []
        self._cond_since: dict[str, float | None] = {k: None for k in ERRORS}
        self._active: dict[str, bool] = {k: False for k in ERRORS}
        self._oop_since: float | None = None
        self._settled: dict | None = None       # where the feet last settled
        self._stepping: tuple[dict, float] | None = None
        self._step_since: float | None = None
        self._step_moved = 0.0
        self._drift_now = 0.0

    # ---------------- commands ----------------

    def start(self, test: str, duration_s: float | None = None, t: float | None = None,
              lean_limit_deg: float | None = None, drift_limit_cm: float | None = None,
              countdown_s: float | None = None):
        if test not in TESTS:
            raise ValueError(f"unknown sway test {test!r} (use {', '.join(TESTS)})")
        if self.phase != "idle":
            raise ValueError("a sway test is already running")
        self._idle()
        self.phase, self.test, self._t0 = "countdown", test, t
        if duration_s:
            self.duration_s = float(duration_s)
        if lean_limit_deg:
            self.lean_limit = float(lean_limit_deg)
        if drift_limit_cm:
            self.drift_limit = float(drift_limit_cm)
        if countdown_s:   # e.g. longer, to walk back from the phone when testing alone
            self.countdown_s = min(60.0, max(self.cfg.baseline_s + 0.5, float(countdown_s)))
        self._pending.append({"kind": "sway_started", "test": test, **TESTS[test],
                              "countdown_s": self.countdown_s, "duration_s": self.duration_s,
                              "lean_limit_deg": self.lean_limit, "drift_limit_cm": self.drift_limit})

    def cancel(self):
        if self.phase != "idle":
            self._pending.append({"kind": "sway_cancelled", "test": self.test})
        self._idle()

    def reset(self):
        self.cancel()
        self.results = {}

    def summary(self) -> dict:
        scores = {k: (self.results[k]["errors"] if k in self.results else None) for k in TESTS}
        return {"results": {k: {"metrics": v.get("metrics"), "mode": v["mode"],
                                "lost_balance": v["lost_balance"], "trunk_lean": v.get("trunk_lean"),
                                "errors": v["errors"], "by_type": v["by_type"], "log": v["log"],
                                "trail": v.get("trail"), "duration_s": v["duration_s"]}
                            for k, v in self.results.items()},
                "scores": scores,
                "total_errors": sum(v for v in scores.values() if v is not None),
                "complete": all(v is not None for v in scores.values()),
                "romberg": romberg_ratios(self.results.get("romberg_eo"),
                                          self.results.get("romberg_ec"))}

    def handle_command(self, msg: dict) -> list[dict]:
        """sway_start {test, duration, lean_limit, drift_limit, countdown} /
        sway_cancel / sway_reset / sway_status."""
        kind = msg.get("type")
        try:
            if kind == "sway_start":
                self.start(msg.get("test", ""), msg.get("duration"),
                           lean_limit_deg=msg.get("lean_limit"),
                           drift_limit_cm=msg.get("drift_limit"),
                           countdown_s=msg.get("countdown"))
            elif kind == "sway_cancel":
                self.cancel()
            elif kind == "sway_reset":
                self.reset()
            elif kind != "sway_status":
                return [{"type": "error", "error": f"unknown command {kind!r}", "command": kind}]
        except (ValueError, TypeError) as e:
            return [{"type": "error", "error": str(e), "command": kind}]
        return [{"type": "ack", "command": kind, **self.summary()}]

    # ---------------- per frame ----------------

    def update(self, t: float, pose: dict, w: int, h: int) -> dict:
        events, self._pending = self._pending, []
        m = measure(pose, w, h, self.cfg) if self.phase != "idle" else None
        if self.phase == "countdown":
            self._countdown(t, m, events)
        elif self.phase == "running":
            self._record(t, m, events)
        return self._status(t, events)

    def _countdown(self, t, m, events):
        cfg = self.cfg
        if self._t0 is None:
            self._t0 = t
        elapsed = t - self._t0
        if elapsed >= self.countdown_s - cfg.baseline_s and m is not None:
            self._base.append(m)
        if elapsed < self.countdown_s:
            return
        ref = self._make_ref()
        if ref is None:
            if elapsed >= self.countdown_s + cfg.max_wait_s:
                self._fail(events, "Couldn't see the body during the countdown. Step back so the "
                                   "whole body, including the feet, is in frame.")
            else:
                self._base = self._base[-30:]
            return
        self.ref, self.mode = ref, ref["mode"]
        self._settled = ref["ankles"]
        self.phase, self._run_t0, self._last_seen = "running", t, t
        events.append({"kind": "sway_running", "test": self.test, "mode": self.mode,
                       "warnings": ref["warnings"]})

    def _make_ref(self) -> dict | None:
        b = [m for m in self._base if m is not None]
        if len(b) < 3:
            return None
        scales = [m["m_per_px"] for m in b if m["m_per_px"]]
        depth = [m["xyz"] for m in b if m["xyz"] is not None]
        mode = "depth" if len(depth) >= 0.8 * len(b) else "2d"
        if mode == "2d" and not scales:
            return None
        ref = {"mode": mode, "warnings": [],
               "com_px": np.median([m["com_px"] for m in b], axis=0),
               "m_per_px": median(scales) if scales else None,
               "xyz": np.median(depth, axis=0) if mode == "depth" else None,
               "lean_deg": median(m["lean_deg"] for m in b),
               "ankles": None, "leg_px": None}
        ank = [m for m in b if m["ankles"] is not None]
        if ank:
            ref["ankles"] = _canonical_ankles([m["ankles"] for m in ank])
            ref["leg_px"] = median(m["leg_px"] for m in ank)
        else:
            ref["warnings"].append("Feet not visible: steps can't be detected.")
        if mode == "2d":
            ref["warnings"].append("No depth data: only side-to-side sway is measured.")
        return ref

    def _offset(self, m) -> tuple[float, float | None] | None:
        """(ml, ap) offset from the start position in metres."""
        ref = self.ref
        if self.mode == "depth":
            if m["xyz"] is None:
                return None
            d = m["xyz"] - ref["xyz"]
            return float(d[0]), float(-d[2])        # toward the camera = forward
        return float((m["com_px"][0] - ref["com_px"][0]) * ref["m_per_px"]), None

    def _record(self, t, m, events):
        cfg = self.cfg
        self._frames += 1
        off = self._offset(m) if m is not None else None
        if off is not None:
            self._samples.append((t, off[0], off[1]))
            self._last_seen = t
            self._live = off
        elif t - self._last_seen > cfg.max_lost_s:
            self._fail(events, "Lost sight of the body for more than "
                               f"{cfg.max_lost_s:.0f} s. Keep the whole body in frame.")
            return

        if m is not None:
            self._track_lean(t, m["lean_deg"] - self.ref["lean_deg"], events)
            self._track_step(t, m, events)
        if off is not None:
            self._drift_now = math.hypot(off[0], off[1] or 0.0) * 100
            on = self._drift_now > (0.8 if self._active["drift"] else 1.0) * self.drift_limit
            self._condition(t, "drift", on, cfg.drift_hold_s, events)
        self._track_out_of_position(t, events)

        if t - self._run_t0 >= self.duration_s:
            self._finish(events, t)

    # ---------------- errors ----------------

    def _condition(self, t, kind, on: bool, hold: float, events):
        """Generic error condition: counts once when it has held for `hold`,
        then has to clear before it can count again."""
        if on:
            since = self._cond_since[kind] = self._cond_since[kind] or t
            if not self._active[kind] and t - since >= hold:
                self._active[kind] = True
                self._count(kind, since, events)
        else:
            self._cond_since[kind] = None
            self._active[kind] = False

    def _count(self, kind, since, events):
        t_rel = round(since - self._run_t0, 2)
        entry = {"error": kind, "label": ERRORS[kind], "t": t_rel, "counted": True}
        if kind != "out_of_position" and any(
                e["counted"] and e["error"] != "out_of_position"
                and abs(e["t"] - t_rel) < self.cfg.error_merge_s for e in self.log):
            entry.update(counted=False, not_counted_reason="simultaneous")
        elif self.errors >= self.cfg.max_errors:
            entry.update(counted=False, not_counted_reason="max errors")
        if entry["counted"]:
            self.errors += 1
            self.by_type[kind] += 1
        self.log.append(entry)
        events.append({"kind": "sway_error", **entry, "errors": self.errors})

    def _track_step(self, t, m, events):
        """A step = the feet moved from where they last settled. After a step,
        once they're still again, that's the new settled position (so the next
        step counts too). The feet stay "out of position" while they're away
        from the original stance."""
        cfg, ref = self.cfg, self.ref
        if m["ankles"] is None or ref["ankles"] is None:
            return
        leg = ref["leg_px"] or 1.0
        cur = m["ankles"]
        self._step_moved = _ankle_shift(cur, ref["ankles"]) / leg
        back_home = self._step_moved <= cfg.step_threshold     # returning to the stance isn't a step
        if self._stepping is None:
            if _ankle_shift(cur, self._settled) / leg > cfg.step_threshold and not back_home:
                self._step_since = self._step_since or t
                if t - self._step_since >= cfg.step_hold_s:
                    self._count("step", self._step_since, events)
                    self._stepping = (cur, t)
            else:
                self._step_since = None
        else:
            anchor, since = self._stepping
            if _ankle_shift(cur, anchor) / leg > 0.05:
                self._stepping = (cur, t)
            elif t - since >= cfg.restable_s or back_home:
                self._settled, self._stepping, self._step_since = anchor, None, None
        if back_home and self._stepping is None:
            self._settled = ref["ankles"]
        self._active["step"] = self._stepping is not None or self._step_moved > cfg.step_threshold

    def _track_out_of_position(self, t, events):
        any_on = any(self._active[k] for k in ("step", "lean", "drift"))
        if any_on:
            self._oop_since = self._oop_since or t
        else:
            self._oop_since = None
        self._condition(t, "out_of_position",
                        any_on and t - self._oop_since > self.cfg.out_of_position_s, 0, events)

    def _track_lean(self, t, lean, events):
        """Sideways trunk lean vs the start posture; alert + error past the limit."""
        L, cfg = self._lean, self.cfg
        if L["alerting"] and L["last_t"] is not None:
            L["over_s"] += t - L["last_t"]
        L["last_t"], L["now"] = t, lean
        L["max"], L["min"] = max(L["max"], lean), min(L["min"], lean)
        side = "left" if lean > 0 else "right"     # image-right = subject's left
        if abs(lean) > self.lean_limit:
            L["over_since"] = L["over_since"] or t
            if not L["alerting"] and t - L["over_since"] >= cfg.lean_hold_s:
                L["alerting"] = True
                L["episodes"] += 1
                events.append({"kind": "sway_lean", "event": "start", "side": side,
                               "lean_deg": round(abs(lean), 1), "limit_deg": self.lean_limit,
                               "t": round(t - self._run_t0, 2)})
                self._active["lean"] = True
                self._count("lean", L["over_since"], events)
        else:
            L["over_since"] = None
            if L["alerting"] and abs(lean) < 0.8 * self.lean_limit:
                L["alerting"] = False
                self._active["lean"] = False
                events.append({"kind": "sway_lean", "event": "end", "t": round(t - self._run_t0, 2)})

    def _lean_summary(self) -> dict:
        L = self._lean
        return {"left_deg": round(L["max"], 1), "right_deg": round(-L["min"], 1),
                "limit_deg": self.lean_limit, "times_over_limit": L["episodes"],
                "time_over_limit_s": round(L["over_s"], 2)}

    # ---------------- result ----------------

    def _trail(self) -> list:
        """COM path at ~trail_hz: [[t, ml_cm, ap_cm or None], ...] for plotting."""
        out, last = [], -1e9
        for t, ml, ap in self._samples:
            if t - last >= 1.0 / self.cfg.trail_hz:
                out.append([round(t - self._run_t0, 2), round(ml * 100, 1),
                            round(ap * 100, 1) if ap is not None else None])
                last = t
        return out

    def _finish(self, events, t):
        s = self._samples
        metrics = None
        if s and s[-1][0] - s[0][0] >= self.cfg.min_duration_s:
            ts, ml, ap = zip(*s)
            metrics = sway_metrics(ts, ml, ap if self.mode == "depth" else None, self.cfg.smooth_s)
        steps = [e for e in self.log if e["error"] == "step"]
        result = {"kind": "sway_done", "test": self.test, "label": TESTS[self.test]["label"],
                  "mode": self.mode, "duration_s": round(t - self._run_t0, 2),
                  "errors": self.errors, "by_type": dict(self.by_type), "log": list(self.log),
                  "max_errors": self.cfg.max_errors,
                  "lost_balance": bool(steps),
                  "lost_balance_at_s": steps[0]["t"] if steps else None,
                  "coverage": round(len(self._samples) / max(1, self._frames), 2),
                  "metrics": metrics, "trunk_lean": self._lean_summary(),
                  "drift_limit_cm": self.drift_limit, "trail": self._trail()}
        self.results[self.test] = result
        result["session"] = self.summary()
        events.append(result)
        self._idle()

    def _fail(self, events, reason: str):
        events.append({"kind": "sway_failed", "test": self.test, "reason": reason})
        self._idle()

    def _status(self, t, events) -> dict:
        out = {"phase": self.phase, "test": self.test, "events": events}
        if self.phase == "countdown":
            el = 0.0 if self._t0 is None else t - self._t0
            out["countdown_left"] = round(max(0.0, self.countdown_s - el), 2)
            out["waiting_for_view"] = el >= self.countdown_s
        elif self.phase == "running":
            out["mode"] = self.mode
            out["time_left"] = round(max(0.0, self.duration_s - (t - self._run_t0)), 2)
            out["errors"] = self.errors
            out["active"] = [k for k in ERRORS if self._active[k]]
            out["log"] = self.log
            out["drift_limit_cm"] = self.drift_limit
            out["drift_cm"] = round(self._drift_now, 1)
            if self._lean["now"] is not None:
                out["lean_deg"] = round(self._lean["now"], 1)
                out["lean_over_limit"] = self._lean["alerting"]
                out["lean_limit_deg"] = self.lean_limit
            if self._live is not None:
                out["ml_cm"] = round(self._live[0] * 100, 1)
                out["ap_cm"] = round(self._live[1] * 100, 1) if self._live[1] is not None else None
        return out


def _ankle_shift(a: dict, b: dict) -> float:
    """How far the feet moved (pixels, worst foot), ignoring left/right label swaps:
    the model sometimes swaps the ankles, which isn't a step."""
    same = max(float(np.linalg.norm(a[s] - b[s])) for s in ("left", "right"))
    swapped = max(float(np.linalg.norm(a["left"] - b["right"])),
                  float(np.linalg.norm(a["right"] - b["left"])))
    return min(same, swapped)


def _canonical_ankles(frames: list[dict]) -> dict:
    """Median ankle positions, with each frame's labels matched to the first frame
    so swapped frames don't average the two feet together."""
    first = frames[0]
    fixed = []
    for a in frames:
        same = sum(float(np.linalg.norm(a[s] - first[s])) for s in ("left", "right"))
        swap = (float(np.linalg.norm(a["left"] - first["right"]))
                + float(np.linalg.norm(a["right"] - first["left"])))
        fixed.append(a if same <= swap else {"left": a["right"], "right": a["left"]})
    return {s: np.median([a[s] for a in fixed], axis=0) for s in ("left", "right")}


def sway_messages(result: dict, frame_id=None) -> list[dict]:
    """Turn this frame's sway events into WebSocket messages."""
    msgs = []
    for ev in (result.get("sway") or {}).get("events", []):
        typ = {"sway_done": "result", "sway_lean": "alert", "sway_error": "event"}.get(ev["kind"], "status")
        msgs.append({"type": typ, "frame_id": frame_id, **ev})
    return msgs


class SwayTracker:
    """Passive sway recording alongside another test (e.g. a BESS stance): the
    first `ref_frames` measured frames are the start position, then COM offsets
    are collected; `metrics()` gives the same sway metrics as the sway tests."""

    def __init__(self, cfg: SwayConfig | None = None, ref_frames: int = 10):
        self.cfg = cfg or SwayConfig()
        self.ref_frames = ref_frames
        self.reset()

    def reset(self):
        self._base: list[dict] = []
        self.ref: dict | None = None
        self.mode: str | None = None
        self.samples: list[tuple[float, float, float | None]] = []
        self.live: tuple[float, float | None] | None = None

    def update(self, t: float, pose: dict, w: int, h: int):
        m = measure(pose, w, h, self.cfg)
        if m is None:
            return
        if self.ref is None:
            self._base.append(m)
            if len(self._base) >= self.ref_frames:
                depth = [b["xyz"] for b in self._base if b["xyz"] is not None]
                scales = [b["m_per_px"] for b in self._base if b["m_per_px"]]
                self.mode = "depth" if len(depth) >= 0.8 * len(self._base) else "2d"
                if self.mode == "2d" and not scales:
                    self._base = []
                    return
                self.ref = {"xyz": np.median(depth, axis=0) if self.mode == "depth" else None,
                            "com_px": np.median([b["com_px"] for b in self._base], axis=0),
                            "m_per_px": median(scales) if scales else None}
            return
        if self.mode == "depth":
            if m["xyz"] is None:
                return
            d = m["xyz"] - self.ref["xyz"]
            off = (float(d[0]), float(-d[2]))
        else:
            off = (float((m["com_px"][0] - self.ref["com_px"][0]) * self.ref["m_per_px"]), None)
        self.samples.append((t, off[0], off[1]))
        self.live = off

    def live_cm(self) -> float | None:
        if self.live is None:
            return None
        return round(math.hypot(self.live[0], self.live[1] or 0.0) * 100, 1)

    def metrics(self) -> dict | None:
        s = self.samples
        if not s or s[-1][0] - s[0][0] < self.cfg.min_duration_s:
            return None
        ts, ml, ap = zip(*s)
        m = sway_metrics(ts, ml, ap if self.mode == "depth" else None, self.cfg.smooth_s)
        if m is not None:
            m["mode"] = self.mode
        return m
