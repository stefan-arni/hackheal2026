"""Eye alignment tracking: detects when one eye drifts out of line with the other.

Uses the MediaPipe Tasks FaceLandmarker (478 landmarks, including 5 per iris).
For each eye we measure where the iris sits inside the eye opening:

  h = position along the line from one eye corner to the other (0..1, image left->right)
  v = offset perpendicular to that line, in eye-widths (+ = down in the image)

When both eyes look somewhere together, their h/v move together. When one eye
drifts (e.g. turns outward or inward while the other stays put), the difference
between the eyes changes. We learn each person's normal difference during a
short calibration, then alert when the difference stays off-baseline for long
enough to not be noise.

This is a screening aid for a demo, not a medical device.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from mp_models import ensure_face_model  # noqa: E402  (shared with the pose server)

# Face mesh indices. "right"/"left" are the SUBJECT's eyes.
EYES = {
    "right": {"corners": (33, 133), "upper": 159, "lower": 145, "iris": (468, 469, 470, 471, 472)},
    "left": {"corners": (362, 263), "upper": 386, "lower": 374, "iris": (473, 474, 475, 476, 477)},
}
NOSE_TIP, FACE_EDGE_A, FACE_EDGE_B = 1, 234, 454


# --------------------------------------------------------------------------- #
# Geometry (pure functions on pixel coordinates)
# --------------------------------------------------------------------------- #

def eye_geometry(pts: np.ndarray, eye: dict) -> dict:
    """Iris position inside one eye. `pts` is (478, 2) pixel coords."""
    a, b = pts[eye["corners"][0]], pts[eye["corners"][1]]
    if a[0] > b[0]:                      # order corners left->right in the image
        a, b = b, a
    axis = b - a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        raise ValueError("degenerate eye")
    u = axis / width                     # along the eye
    n = np.array([-u[1], u[0]])          # perpendicular (points down when eye is level)

    iris_pts = pts[list(eye["iris"])]
    center = iris_pts.mean(axis=0)
    rel = center - a
    return {
        "h": float(rel @ u / width),
        "v": float(rel @ n / width),
        "openness": float(np.linalg.norm(pts[eye["upper"]] - pts[eye["lower"]]) / width),
        "width_px": width,
        "iris_center_px": [float(center[0]), float(center[1])],
        "iris_radius_px": float(np.linalg.norm(iris_pts[1:] - center, axis=1).mean()),
        "corners_px": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
    }


def head_yaw_ratio(pts: np.ndarray) -> float:
    """Rough head turn: 0 = facing camera, +/-1 = fully sideways."""
    da = abs(pts[NOSE_TIP][0] - pts[FACE_EDGE_A][0])
    db = abs(pts[NOSE_TIP][0] - pts[FACE_EDGE_B][0])
    return float((da - db) / max(da + db, 1e-6))


# --------------------------------------------------------------------------- #
# Alert logic (no MediaPipe needed, unit-testable)
# --------------------------------------------------------------------------- #

@dataclass
class MonitorConfig:
    calibration_s: float = 2.0       # seconds of valid frames to learn the baseline
    h_threshold: float = 0.10        # horizontal drift, in eye-widths
    v_threshold: float = 0.08        # vertical drift, in eye-widths
    hold_s: float = 0.5              # drift must persist this long to alert
    clear_ratio: float = 0.7         # alert clears below threshold * clear_ratio...
    clear_s: float = 0.5             # ...for this long
    smooth_frames: int = 5           # median filter length
    min_openness: float = 0.15       # below this the eye is closed/blinking
    max_yaw: float = 0.35            # ignore frames with the head turned this far
    min_eye_width_px: float = 18.0   # face too small to measure reliably


@dataclass
class AlignmentMonitor:
    cfg: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self._calib: list[tuple[float, float, float, float]] = []
        self._calib_start: float | None = None
        self.baseline: dict | None = None
        self._hist: deque = deque(maxlen=self.cfg.smooth_frames)
        self.alerting = False
        self._over_since: float | None = None
        self._under_since: float | None = None
        self.alert_started_at: float | None = None

    def update(self, t: float, eyes: dict | None, skip_reason: str | None = None) -> dict:
        """Feed one frame. `eyes` = {"right": geom, "left": geom} or None.

        Returns status dict; `event` is "start" / "end" when an alert begins/ends.
        """
        out = {"state": "calibrating" if self.baseline is None else "monitoring",
               "alerting": self.alerting, "event": None, "skip_reason": skip_reason}
        if eyes is None or skip_reason:
            return out

        r, l = eyes["right"], eyes["left"]
        sample = (r["h"], r["v"], l["h"], l["v"])

        if self.baseline is None:
            self._calib_start = self._calib_start if self._calib_start is not None else t
            self._calib.append(sample)
            out["calibration_progress"] = round(min(1.0, (t - self._calib_start) / self.cfg.calibration_s), 2)
            if t - self._calib_start >= self.cfg.calibration_s and len(self._calib) >= 5:
                cols = list(zip(*self._calib))
                self.baseline = {k: median(c) for k, c in zip(("rh", "rv", "lh", "lv"), cols)}
                self.baseline["dh"] = self.baseline["rh"] - self.baseline["lh"]
                self.baseline["dv"] = self.baseline["rv"] - self.baseline["lv"]
                out["state"] = "monitoring"
                out["event"] = "calibrated"
            return out

        # smoothed per-eye positions
        self._hist.append(sample)
        rh, rv, lh, lv = (median(c) for c in zip(*self._hist))
        b = self.baseline
        dev_h = (rh - lh) - b["dh"]
        dev_v = (rv - lv) - b["dv"]
        score = max(abs(dev_h) / self.cfg.h_threshold, abs(dev_v) / self.cfg.v_threshold)

        # Which eye drifted: the one that moved further from its own baseline.
        # (If the person is looking straight ahead, that's the eye that wandered.)
        moved_r = math.hypot(rh - b["rh"], rv - b["rv"])
        moved_l = math.hypot(lh - b["lh"], lv - b["lv"])
        eye = "right" if moved_r >= moved_l else "left"

        if abs(dev_h) / self.cfg.h_threshold >= abs(dev_v) / self.cfg.v_threshold:
            direction = "horizontal"
        else:
            direction = "vertical"

        out.update({"deviation_h": round(dev_h, 4), "deviation_v": round(dev_v, 4),
                    "score": round(score, 3), "drifting_eye": eye, "direction": direction})

        if not self.alerting:
            if score >= 1.0:
                self._over_since = self._over_since if self._over_since is not None else t
                if t - self._over_since >= self.cfg.hold_s:
                    self.alerting = True
                    self.alert_started_at = self._over_since
                    self._under_since = None
                    out["event"] = "start"
            else:
                self._over_since = None
        else:
            if score < self.cfg.clear_ratio:
                self._under_since = self._under_since if self._under_since is not None else t
                if t - self._under_since >= self.cfg.clear_s:
                    self.alerting = False
                    self._over_since = None
                    out["event"] = "end"
                    out["alert_duration_s"] = round(t - (self.alert_started_at or t), 2)
            else:
                self._under_since = None

        out["alerting"] = self.alerting
        return out


def frame_check(eyes: dict, yaw: float, cfg: MonitorConfig) -> str | None:
    """Reason to ignore this frame, or None if it's usable."""
    if min(eyes["right"]["width_px"], eyes["left"]["width_px"]) < cfg.min_eye_width_px:
        return "face_too_small"
    if min(eyes["right"]["openness"], eyes["left"]["openness"]) < cfg.min_openness:
        return "blink"
    if abs(yaw) > cfg.max_yaw:
        return "head_turned"
    return None


# --------------------------------------------------------------------------- #
# Analyzer (MediaPipe)
# --------------------------------------------------------------------------- #

class EyeAnalyzer:
    """One per video stream (VIDEO mode needs increasing timestamps).

    Runs the eye-drift monitor and the eye-movement analysis (saccades, speed,
    smoothness; see eye_movement.py) on the same iris tracking.
    """

    def __init__(self, cfg: MonitorConfig | None = None, movement_cfg=None):
        from eye_movement import EyeMovementTest, GazeTracker, MovementConfig
        self.cfg = cfg or MonitorConfig()
        mcfg = movement_cfg or MovementConfig()
        self.gaze = GazeTracker(mcfg)
        self.test = EyeMovementTest(mcfg)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_face_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.monitor = AlignmentMonitor(self.cfg)
        self._last_ts = -1

    def _next_ts(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process_bgr(self, frame_bgr: np.ndarray, t: float | None = None, meta: dict | None = None) -> dict:
        """`meta` is the frame's message metadata; its "timestamp_ms" (capture time
        from the phone or test client) is used as the frame time when present, so
        eye speeds aren't distorted by network or processing delays."""
        if t is None and meta and meta.get("timestamp_ms") is not None:
            t = float(meta["timestamp_ms"]) / 1000.0
        h, w = frame_bgr.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        res = self._landmarker.detect_for_video(image, self._next_ts())
        ms = (time.perf_counter() - t0) * 1000
        t = time.monotonic() if t is None else t

        if not res.face_landmarks or len(res.face_landmarks[0]) < 478:
            status = self.monitor.update(t, None, "no_face")
            gaze = self.gaze.update(t, None)
            return {"face_detected": False, "inference_ms": round(ms, 1), **status,
                    "movement": gaze, "eye_test": self.test.update(t, None, None)}

        pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]])
        return self.analyze_points(pts, t, ms)

    def analyze_points(self, pts: np.ndarray, t: float, ms: float = 0.0) -> dict:
        eyes = {name: eye_geometry(pts, spec) for name, spec in EYES.items()}
        yaw = head_yaw_ratio(pts)
        skip = frame_check(eyes, yaw, self.cfg)
        status = self.monitor.update(t, eyes, skip)
        # eye movement: blinks and a too-small face break the trace; a turned
        # head is still measured (eye position is relative to the eye corners)
        usable = skip not in ("blink", "face_too_small")
        gaze = self.gaze.update(t, eyes if usable else None)
        test = self.test.update(t, gaze if usable else None, yaw)
        return {
            "face_detected": True,
            "inference_ms": round(ms, 1),
            "head_yaw": round(yaw, 3),
            "eyes": {name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}
                     for name, g in eyes.items()},
            **status,
            "movement": gaze,
            "eye_test": test,
        }

    def handle_command(self, msg: dict) -> list[dict]:
        """eye_test_start {"mode": "saccades" | "pursuit", "duration": s} /
        eye_test_cancel / eye_test_status."""
        return self.test.handle_command(msg)

    def recalibrate(self):
        self.monitor.reset()
        self.gaze.reset()

    def close(self):
        self._landmarker.close()


# --------------------------------------------------------------------------- #
# Drawing (laptop test client)
# --------------------------------------------------------------------------- #

def draw_eyes(frame: np.ndarray, r: dict | None) -> np.ndarray:
    if not r:
        return frame
    h, w = frame.shape[:2]
    if not r.get("face_detected"):
        cv2.putText(frame, "Eyes: no face", (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    color = (0, 0, 255) if r["alerting"] else (0, 220, 0)
    for name, g in r["eyes"].items():
        (ax, ay), (bx, by) = g["corners_px"]
        cv2.line(frame, (int(ax), int(ay)), (int(bx), int(by)), (200, 200, 200), 1)
        cx, cy = g["iris_center_px"]
        cv2.circle(frame, (int(cx), int(cy)), max(2, int(g["iris_radius_px"])), color, 1)
        cv2.circle(frame, (int(cx), int(cy)), 2, color, -1)

    if r["state"] == "calibrating":
        p = r.get("calibration_progress", 0)
        msg = f"Calibrating: look straight at the camera ({int(p * 100)}%)"
        if r.get("skip_reason"):
            msg += f"  [{r['skip_reason']}]"
        cv2.putText(frame, msg, (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    elif "score" in r:
        cv2.putText(frame, f"Eye drift h {r['deviation_h']:+.3f}  v {r['deviation_v']:+.3f}  "
                           f"score {r['score']:.2f}" + (f"  [{r['skip_reason']}]" if r.get("skip_reason") else ""),
                    (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if r["alerting"]:
        flash = int(time.time() * 4) % 2 == 0
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
        text = f"ALERT: {r.get('drifting_eye', '?')} eye drifting ({r.get('direction', '')})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = (w - tw) // 2
        cv2.rectangle(frame, (x - 10, h // 2 - th - 12), (x + tw + 10, h // 2 + 12), (0, 0, 180), -1)
        cv2.putText(frame, text, (x, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame
