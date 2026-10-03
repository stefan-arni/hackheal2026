"""Duck detection: fires once when the person quickly drops their head down.

Two ways to measure, picked per frame from what's visible:

  "world": feet visible -> head height above the feet in meters (MediaPipe 3D).
           A duck is the head dropping by `world_drop` (default 20%) below the
           person's recent standing height. Doesn't care how far they are from
           the camera.
  "image": feet out of frame (e.g. a webcam on a desk) -> how far the nose moves
           down in the image, measured in shoulder-widths so it scales with
           distance. A duck is a drop of `image_drop` (default 0.8) shoulder-widths.

"Recent standing height" is the highest the head has been in the last
`baseline_s` seconds, so it adapts as people shift around, and slowly sinking
into a chair over many seconds isn't a duck. After a duck, the person has to
come most of the way back up before another one can fire.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
FOOT_POINTS = (27, 28, 29, 30, 31, 32)   # ankles, heels, toes


def _get(lm, key):
    return lm[key] if isinstance(lm, dict) else getattr(lm, key)


@dataclass
class DuckConfig:
    world_drop: float = 0.20      # fraction of standing head height
    image_drop: float = 0.8       # shoulder-widths
    rearm_ratio: float = 0.4      # must recover to below this fraction of the threshold
    baseline_s: float = 2.0       # head must have been up within this window
    cooldown_s: float = 0.8       # minimum time between quacks
    smooth_frames: int = 3
    min_visibility: float = 0.5


@dataclass
class DuckDetector:
    cfg: DuckConfig = field(default_factory=DuckConfig)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.mode: str | None = None
        self._hist: deque = deque(maxlen=self.cfg.smooth_frames)
        self._baseline: deque = deque()      # (t, value) for the rolling window
        self.ducking = False
        self._last_duck_t: float | None = None
        self.count = 0

    def measure(self, landmarks, world_landmarks, width: int, height: int):
        """(mode, value) where larger value = head higher, or (None, None)."""
        def vis(i):
            return _get(landmarks[i], "visibility") or 0.0

        if vis(NOSE) < self.cfg.min_visibility:
            return None, None
        feet_ok = min(vis(i) for i in FOOT_POINTS) >= self.cfg.min_visibility
        if feet_ok and world_landmarks is not None:
            feet_y = max(_get(world_landmarks[i], "y") for i in FOOT_POINTS)
            return "world", feet_y - _get(world_landmarks[NOSE], "y")   # y points down
        if min(vis(L_SHOULDER), vis(R_SHOULDER)) < self.cfg.min_visibility:
            return None, None
        sw = abs(_get(landmarks[L_SHOULDER], "x") - _get(landmarks[R_SHOULDER], "x")) * width
        if sw < 1e-6:
            return None, None
        # image: in shoulder-widths, negated so "up" is larger like world mode
        return "image", -_get(landmarks[NOSE], "y") * height / sw

    def update(self, t: float, mode: str | None, value: float | None) -> dict:
        out = {"event": None, "ducking": self.ducking, "count": self.count, "mode": mode,
               "drop": None, "threshold": None}
        if mode is None:
            out["skip_reason"] = "head_not_visible"
            return out
        if mode != self.mode:            # units changed: start over
            self.mode = mode
            self._hist.clear()
            self._baseline.clear()
            self.ducking = False

        self._hist.append(value)
        v = median(self._hist)
        self._baseline.append((t, v))
        while self._baseline and t - self._baseline[0][0] > self.cfg.baseline_s:
            self._baseline.popleft()
        top = max(b for _, b in self._baseline)

        if mode == "world":
            drop = 1.0 - v / top if top > 1e-6 else 0.0
            thr = self.cfg.world_drop
        else:
            drop = top - v
            thr = self.cfg.image_drop
        out.update({"drop": round(drop, 3), "threshold": thr})

        cooled = self._last_duck_t is None or t - self._last_duck_t >= self.cfg.cooldown_s
        if not self.ducking and drop >= thr and cooled:
            self.ducking = True
            self.count += 1
            self._last_duck_t = t
            out["event"] = "duck"
        elif self.ducking and drop < thr * self.cfg.rearm_ratio:
            self.ducking = False
        out["ducking"], out["count"] = self.ducking, self.count
        return out
