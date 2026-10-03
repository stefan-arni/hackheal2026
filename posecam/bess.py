"""BESS (Balance Error Scoring System), firm surface, scored from pose landmarks.

Three stances (`BessConfig.duration_s` each, 10 s by default), eyes closed, hands on hips:
  "double": feet together
  "tandem": one foot in front of the other, NON-dominant foot in back
  "single": standing on the NON-dominant leg

One point per error, one rule per error type:
  hands_off_hips     wrist moves away from its hip (relative to the start position)
  eyes_open          eyes open (face check, or marked from the app). OFF by default
                     (track_eyes=False): not checked, not scored, not reported
  step_stumble_fall  a stance foot moves, the raised foot touches down, or the hips drop
  hip_angle          a hip goes past 30 degrees of flexion or abduction (see
                     hip_motion: measured mostly from the 2D image, not 3D depth)
  foot_lift          forefoot (toe) or heel of a stance foot comes off the floor
  out_of_position    out of the test position for more than 5 s straight

Each error type counts once per distinct event: it has to clear before it can
count again. As in the standard BESS, errors that happen at the same moment
(within `simultaneous_s` of each other) count as one, and a stance scores at
most 10. The result is the sum over the three stances, broken down per stance.

Flow per stance: start (button) -> `countdown_s` to get in position and close
eyes; the last second records the person's own start position (baseline) ->
`duration_s` of scoring -> result.

Pose-based scoring assumes a fixed camera that sees the whole body.
This is an aid for a demo, not a validated clinical scoring tool.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import median

import numpy as np

from balance import foot_heights

STANCES = {"double": "Feet together", "tandem": "Tandem", "single": "Single leg"}
ERROR_TYPES = ("hands_off_hips", "eyes_open", "step_stumble_fall", "hip_angle", "foot_lift",
               "out_of_position")
ERROR_LABELS = {
    "hands_off_hips": "Hands off hips",
    "eyes_open": "Eyes opened",
    "step_stumble_fall": "Step / stumble / fall",
    "hip_angle": "Hip > 30 deg",
    "foot_lift": "Heel / forefoot lifted",
    "out_of_position": "Out of position > 5 s",
}

# pose landmark indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_WR, R_WR = 11, 12, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE = 23, 24, 25, 26
L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE = 27, 28, 29, 30, 31, 32
SIDE = {
    "left": {"hip": L_HIP, "knee": L_KNEE, "ankle": L_ANK, "heel": L_HEEL, "toe": L_TOE, "wrist": L_WR},
    "right": {"hip": R_HIP, "knee": R_KNEE, "ankle": R_ANK, "heel": R_HEEL, "toe": R_TOE, "wrist": R_WR},
}
OTHER = {"left": "right", "right": "left"}


def _g(lm, k):
    return lm[k] if isinstance(lm, dict) else getattr(lm, k)


@dataclass
class BessConfig:
    duration_s: float = 10.0
    countdown_s: float = 1.0           # before scoring; just long enough to record the
    baseline_s: float = 1.0            # start position (last `baseline_s` of the countdown)
    max_wait_s: float = 10.0           # extra time allowed if the start position can't be seen
    hands_off_floor: float = 0.45      # wrist-hip distance / torso length that is always "off"
    hands_off_margin: float = 0.20     # ...or this much further than at the start
    hip_angle_deg: float = 30.0
    hip_smooth_frames: int = 5         # median filter on the hip angle
    step_ratio: float = 0.15           # stance ankle moved, in leg-lengths
    fall_ratio: float = 0.25           # hips dropped, in leg-lengths
    foot_lift_ratio: float = 0.05      # heel/toe rose, in leg-lengths
    touch_m: float = 0.03              # single leg: raised foot within this of the floor = touch
    track_eyes: bool = False           # eye tracking is off for now
    eyes_open_below: float = 0.5       # eye "blink" score below this = open
    min_error_s: float = 0.2           # condition must last this long to count
    clear_s: float = 0.3               # ...and be gone this long to end
    simultaneous_s: float = 0.5        # errors starting this close together count once
    out_of_position_s: float = 5.0
    max_errors: int = 10
    min_visibility: float = 0.5


# --------------------------------------------------------------------------- #
# Measurements from one frame of landmarks
# --------------------------------------------------------------------------- #

def _px(lms, i, w, h):
    return np.array([_g(lms[i], "x") * w, _g(lms[i], "y") * h])


def _vis(lms, *idx):
    return min((_g(lms[i], "visibility") or 0.0) for i in idx)


def hip_angles(world, side: str) -> tuple[float, float] | None:
    """(flexion, abduction) of one hip in degrees, from 3D world landmarks.

    Measured in the trunk's own frame, so leaning the whole body doesn't count.
    flexion = thigh forward/back of the trunk line (magnitude),
    abduction = thigh out to the side (positive = away from the midline).
    """
    def p(i):
        return np.array([_g(world[i], "x"), _g(world[i], "y"), _g(world[i], "z")])

    hip_mid = (p(L_HIP) + p(R_HIP)) / 2
    sh_mid = (p(L_SH) + p(R_SH)) / 2
    down = hip_mid - sh_mid
    if np.linalg.norm(down) < 1e-6:
        return None
    down /= np.linalg.norm(down)
    outward = p(SIDE[side]["hip"]) - p(SIDE[OTHER[side]]["hip"])   # points away from midline
    outward -= down * (outward @ down)
    if np.linalg.norm(outward) < 1e-6:
        return None
    outward /= np.linalg.norm(outward)
    fwd = np.cross(down, outward)
    thigh = p(SIDE[side]["knee"]) - p(SIDE[side]["hip"])
    d = thigh @ down
    flex = abs(math.degrees(math.atan2(thigh @ fwd, d)))
    abd = math.degrees(math.atan2(thigh @ outward, d))
    return flex, abd


def _seg_dir(p, q, length0: float, scale: float, zsign: float) -> np.ndarray:
    """3D unit direction of a body segment from its 2D image projection.

    The in-image part comes straight from the image. How far the segment points
    toward/away from the camera comes from how much shorter it looks than at the
    start (a segment tilted 30 deg out of the image looks cos(30) = 87% as long).
    Only the SIGN of that tilt comes from the 3D model, which gets direction right
    far more reliably than amount.
    """
    v = (np.asarray(q, float) - np.asarray(p, float)) / (length0 * scale)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    if n >= 1.0:
        v = v / n
        return np.array([v[0], v[1], 0.0])
    return np.array([v[0], v[1], zsign * math.sqrt(1.0 - n * n)])


def _zsign(wz: dict | None, a: str, b: str, min_dz: float = 0.05) -> float | None:
    """Which way segment a->b points in depth (+1 away from camera), or None if
    the 3D model's depth difference is too small to trust."""
    if not wz or a not in wz or b not in wz:
        return None
    dz = wz[b] - wz[a]
    return None if abs(dz) < min_dz else (1.0 if dz > 0 else -1.0)


def thighs_in_trunk_frame(m: dict, lengths: dict, scale: float) -> dict:
    """Each thigh's direction expressed in the trunk's own frame
    (down, sideways, forward), from mostly-2D geometry. {side: unit 3-vector}."""
    p, wz = m["pts"], m["wz"]
    if not all(k in p for k in ("sh_mid", "hip_mid", "l_hip", "r_hip")):
        return {}
    zs = _zsign(wz, "sh_mid", "hip_mid") or 1.0
    d = _seg_dir(p["sh_mid"], p["hip_mid"], lengths["trunk"], scale, zs)
    a = _seg_dir(p["r_hip"], p["l_hip"], lengths["hip_w"], scale, _zsign(wz, "r_hip", "l_hip") or 1.0)
    a = a - (a @ d) * d
    if np.linalg.norm(a) < 1e-6:
        return {}
    a /= np.linalg.norm(a)
    f = np.cross(d, a)
    out = {}
    for s in SIDE:
        hip_k, knee_k = f"{s[0]}_hip", f"{s[0]}_knee"
        if hip_k in p and knee_k in p:
            zt = _zsign(wz, hip_k, knee_k) or zs     # unknown: assume same as trunk (smaller angle)
            t = _seg_dir(p[hip_k], p[knee_k], lengths["thigh"], scale, zt)
            out[s] = np.array([t @ d, t @ a, t @ f])
    return out


def _angle_deg(u, v) -> float:
    c = float(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1.0, 1.0))
    return math.degrees(math.acos(c))


def measure(landmarks, world, w: int, h: int, min_vis: float = 0.5) -> dict:
    """Raw signals for one frame. Missing/unreliable parts are None."""
    m: dict = {}
    trunk_ok = _vis(landmarks, L_SH, R_SH, L_HIP, R_HIP) >= min_vis
    sh = (_px(landmarks, L_SH, w, h) + _px(landmarks, R_SH, w, h)) / 2
    hip = (_px(landmarks, L_HIP, w, h) + _px(landmarks, R_HIP, w, h)) / 2
    torso = float(np.linalg.norm(sh - hip)) if trunk_ok else 0.0
    m["hip_mid"] = hip if trunk_ok else None

    legs = []
    for s, ix in SIDE.items():
        if _vis(landmarks, ix["hip"], ix["ankle"]) >= min_vis:
            legs.append(float(np.linalg.norm(_px(landmarks, ix["hip"], w, h)
                                             - _px(landmarks, ix["ankle"], w, h))))
    m["leg"] = float(np.mean(legs)) if legs else None

    m["wrist_ratio"] = {}
    for s, ix in SIDE.items():
        ok = torso > 1e-6 and _vis(landmarks, ix["wrist"], ix["hip"]) >= min_vis
        m["wrist_ratio"][s] = (float(np.linalg.norm(_px(landmarks, ix["wrist"], w, h)
                                                    - _px(landmarks, ix["hip"], w, h))) / torso
                               if ok else None)

    m["ankle"], m["heel_y"], m["toe_y"] = {}, {}, {}
    for s, ix in SIDE.items():
        ok = _vis(landmarks, ix["ankle"], ix["heel"], ix["toe"]) >= min_vis
        m["ankle"][s] = _px(landmarks, ix["ankle"], w, h) if ok else None
        m["heel_y"][s] = _px(landmarks, ix["heel"], w, h)[1] if ok else None
        m["toe_y"][s] = _px(landmarks, ix["toe"], w, h)[1] if ok else None

    # points for the hip-angle geometry (image px) and their 3D depth (world z)
    pts, wz = {}, {}
    if trunk_ok:
        pts["sh_mid"], pts["hip_mid"] = sh, hip
        pts["l_hip"], pts["r_hip"] = _px(landmarks, L_HIP, w, h), _px(landmarks, R_HIP, w, h)
        if world is not None:
            wz["sh_mid"] = (_g(world[L_SH], "z") + _g(world[R_SH], "z")) / 2
            wz["hip_mid"] = (_g(world[L_HIP], "z") + _g(world[R_HIP], "z")) / 2
            wz["l_hip"], wz["r_hip"] = _g(world[L_HIP], "z"), _g(world[R_HIP], "z")
    for s, ix in SIDE.items():
        for name in ("knee", "ankle"):
            if _vis(landmarks, ix[name]) >= min_vis:
                pts[f"{s[0]}_{name}"] = _px(landmarks, ix[name], w, h)
                if world is not None:
                    wz[f"{s[0]}_{name}"] = _g(world[ix[name]], "z")
    m["pts"], m["wz"] = pts, (wz or None)

    fh = foot_heights(landmarks, world, w, h, min_vis) if world is not None else None
    m["foot_height_m"] = fh["world"] if fh else None
    m["world_ankle_z"] = ({s: _g(world[SIDE[s]["ankle"]], "z") for s in SIDE}
                          if world is not None else None)
    return m


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

@dataclass
class _ErrorTrack:
    active: bool = False
    on_since: float | None = None
    off_since: float | None = None


@dataclass
class BessSession:
    cfg: BessConfig = field(default_factory=BessConfig)

    def __post_init__(self):
        self.scores: dict[str, dict] = {}     # stance -> latest result
        self._idle()

    # ---------------- control ----------------

    def _idle(self):
        self.phase = "idle"
        self.stance: str | None = None
        self.nondominant: str | None = None
        self._t0 = None
        self._run_t0 = None
        self._base_frames: list[dict] = []
        self.base: dict | None = None
        self.warnings: list[str] = []
        self._tracks = {k: _ErrorTrack() for k in self.error_types() if k != "out_of_position"}
        self._any_since: float | None = None
        self._oop_counted = False
        self._last_counted_t: float | None = None
        self.errors = 0
        self.by_type = {k: 0 for k in self.error_types()}
        self.log: list[dict] = []
        self._pending_events: list[dict] = []
        self._manual_marks: list[str] = []
        self._hip_hist = {s: deque(maxlen=self.cfg.hip_smooth_frames) for s in SIDE}
        self.hip_now: dict = {}
        self.coverage = {"frames": 0, "measured": 0}

    def start(self, stance: str, nondominant: str = "left", t: float | None = None):
        if stance not in STANCES:
            raise ValueError(f"stance must be one of {list(STANCES)}")
        if nondominant not in ("left", "right"):
            raise ValueError("nondominant must be 'left' or 'right'")
        self._idle()
        self.phase, self.stance, self.nondominant = "countdown", stance, nondominant
        self._t0 = t      # None: the countdown starts at the next frame
        self._pending_events.append({"kind": "bess_started", "stance": stance,
                                     "nondominant": nondominant})

    def cancel(self):
        was = self.stance
        self._idle()
        if was:
            self._pending_events.append({"kind": "bess_cancelled", "stance": was})

    def reset(self):
        self.scores = {}
        self._idle()

    def error_types(self) -> tuple[str, ...]:
        """Error types in use (eyes_open only when eye tracking is on)."""
        return tuple(k for k in ERROR_TYPES if self.cfg.track_eyes or k != "eyes_open")

    def mark(self, error: str):
        """Manual error from the app, e.g. one the camera can't see."""
        if error == "eyes_open" and not self.cfg.track_eyes:
            raise ValueError("eye tracking is turned off")
        if error not in ERROR_TYPES:
            raise ValueError(f"error must be one of {list(ERROR_TYPES)}")
        if self.phase != "running":
            raise ValueError("no test is running")
        self._manual_marks.append(error)

    def handle_command(self, msg: dict) -> list[dict]:
        """WebSocket commands: bess_start / bess_cancel / bess_reset / bess_mark / bess_status."""
        kind = msg.get("type")
        try:
            if kind == "bess_start":
                self.start(msg.get("stance", ""), msg.get("nondominant", "left"))
            elif kind == "bess_cancel":
                self.cancel()
            elif kind == "bess_reset":
                self.reset()
            elif kind == "bess_mark":
                self.mark(msg.get("error", ""))
            elif kind != "bess_status":
                return [{"type": "error", "error": f"unknown command {kind!r}", "command": kind}]
        except ValueError as e:
            return [{"type": "error", "error": str(e), "command": kind}]
        return [{"type": "ack", "command": kind, **self.session_summary()}]

    # ---------------- per frame ----------------

    def update(self, t: float, pose: dict, w: int, h: int, eyes_open: bool | None = None) -> dict:
        events: list[dict] = self._pending_events
        self._pending_events = []
        m = None
        if pose.get("detected"):
            m = measure(pose["landmarks"], pose.get("world_landmarks"), w, h, self.cfg.min_visibility)
            # real distance to each ankle from the iPhone's depth map, when sent
            m["ankle_depth_m"] = (pose.get("depth") or {}).get("ankles_m")

        if self.phase == "countdown":
            self._countdown(t, m, events)
        elif self.phase == "running":
            self._score(t, m, eyes_open, events)
            if t - self._run_t0 >= self.cfg.duration_s:
                self._finish(events)
        return self._status(t, events, eyes_open)

    def _countdown(self, t, m, events):
        if self._t0 is None:
            self._t0 = t
        elapsed = t - self._t0
        if elapsed >= self.cfg.countdown_s - self.cfg.baseline_s and m is not None:
            self._base_frames.append(m)
        if elapsed < self.cfg.countdown_s:
            return
        base = self._make_baseline()
        if base is None:
            if elapsed >= self.cfg.countdown_s + self.cfg.max_wait_s:
                stance = self.stance
                self._idle()
                events.append({"kind": "bess_failed", "stance": stance,
                               "reason": "Couldn't see the full body (feet, hips, wrists). "
                                         "Step back so the whole body is in frame."})
            return
        self.base = base
        self.warnings = self._setup_warnings(base)
        self.phase, self._run_t0 = "running", t
        events.append({"kind": "bess_running", "stance": self.stance, "warnings": self.warnings})

    def _make_baseline(self) -> dict | None:
        frames = self._base_frames[-30:]
        if len(frames) < 3:
            return None

        def med(get):
            vals = [get(f) for f in frames]
            vals = [v for v in vals if v is not None]
            return median(vals) if len(vals) >= max(2, len(frames) // 2) else None

        base = {"leg": med(lambda f: f["leg"])}
        hip = [f["hip_mid"] for f in frames if f["hip_mid"] is not None]
        base["hip_y"] = float(np.median([p[1] for p in hip])) if hip else None
        base["wrist_ratio"] = {s: med(lambda f, s=s: f["wrist_ratio"][s]) for s in SIDE}
        base["ankle"] = {}
        for s in SIDE:
            pts = [f["ankle"][s] for f in frames if f["ankle"][s] is not None]
            base["ankle"][s] = np.median(np.array(pts), axis=0) if len(pts) >= 2 else None
        base["heel_y"] = {s: med(lambda f, s=s: f["heel_y"][s]) for s in SIDE}
        base["toe_y"] = {s: med(lambda f, s=s: f["toe_y"][s]) for s in SIDE}
        fh = [f["foot_height_m"] for f in frames if f["foot_height_m"] is not None]
        base["foot_height_m"] = ({s: median(x[s] for x in fh) for s in SIDE} if fh else None)
        d = [f["ankle_depth_m"] for f in frames if f.get("ankle_depth_m")]
        base["ankle_depth_m"] = {s: median(x[s] for x in d) for s in SIDE} if d else None

        base.update(self._hip_baseline(frames))

        stance_feet = self._stance_feet()
        if base["leg"] is None or base["hip_y"] is None:
            return None
        if any(base["ankle"][s] is None or base["heel_y"][s] is None for s in stance_feet):
            return None
        return base

    def _hip_baseline(self, frames) -> dict:
        """Segment lengths and each thigh's start direction in the trunk frame."""
        def seg_len(f, a, b):
            p = f["pts"]
            return float(np.linalg.norm(p[b] - p[a])) if a in p and b in p else None

        def med(vals):
            vals = [v for v in vals if v is not None]
            return median(vals) if len(vals) >= max(2, len(frames) // 2) else None

        standing = self._stance_feet()
        lengths = {
            "trunk": med(seg_len(f, "sh_mid", "hip_mid") for f in frames),
            "hip_w": med(seg_len(f, "r_hip", "l_hip") for f in frames),
            # thigh length from a standing leg: the raised leg may already be tilted
            "thigh": med(seg_len(f, f"{s[0]}_hip", f"{s[0]}_knee") for f in frames for s in standing),
            "shank": {s: med(seg_len(f, f"{s[0]}_knee", f"{s[0]}_ankle") for f in frames) for s in SIDE},
        }
        out = {"hip_lengths": lengths, "thigh_in_trunk": {}}
        if None in (lengths["trunk"], lengths["hip_w"], lengths["thigh"]):
            return out
        per_side = {s: [] for s in SIDE}
        for f in frames:
            for s, c in thighs_in_trunk_frame(f, lengths, self._scale(f, lengths)).items():
                per_side[s].append(c)
        for s, cs in per_side.items():
            if len(cs) >= 2:
                c = np.median(np.array(cs), axis=0)
                out["thigh_in_trunk"][s] = c / np.linalg.norm(c)
        return out

    def _scale(self, m, lengths) -> float:
        """How much bigger/smaller the person looks than at the start (moving
        toward/away from the camera), from the shins of the standing legs, which
        stay upright in every BESS stance."""
        ratios = []
        for s in self._stance_feet():
            k, a = f"{s[0]}_knee", f"{s[0]}_ankle"
            L0 = (lengths.get("shank") or {}).get(s)
            if L0 and k in m["pts"] and a in m["pts"]:
                ratios.append(float(np.linalg.norm(m["pts"][a] - m["pts"][k])) / L0)
        return float(np.clip(median(ratios), 0.8, 1.25)) if ratios else 1.0

    def hip_motion(self, m) -> dict:
        """Hip angle per side in degrees, from mostly-2D geometry.

        Standing legs: how far the thigh has rotated relative to the trunk since the
        start position, which cancels each person's natural posture. Whole-body sway
        moves trunk and thigh together, so it doesn't count. The raised leg in the
        single-leg stance starts at ~20 deg on purpose, so it's measured as its
        absolute angle from the trunk line instead.
        """
        b = self.base
        lengths = b.get("hip_lengths") or {}
        if not b.get("thigh_in_trunk") or None in (lengths.get("trunk"), lengths.get("hip_w"),
                                                   lengths.get("thigh")):
            return {}
        now = thighs_in_trunk_frame(m, lengths, self._scale(m, lengths))
        raised = OTHER[self.nondominant] if self.stance == "single" else None
        out = {}
        for s, c in now.items():
            if s == raised:
                out[s] = _angle_deg(c, np.array([1.0, 0.0, 0.0]))     # vs trunk "down"
            elif s in b["thigh_in_trunk"]:
                out[s] = _angle_deg(c, b["thigh_in_trunk"][s])
        return out

    def _stance_feet(self) -> list[str]:
        return [self.nondominant] if self.stance == "single" else ["left", "right"]

    @staticmethod
    def _back_foot(base) -> str | None:
        """Which foot is further from the camera in the start position, or None
        if it's not clear. Uses the iPhone's depth at each ankle when available;
        otherwise the image: from a camera above floor level the back foot's
        lowest point (heel / toe) sits higher in the picture. (MediaPipe's own
        3D depth of the feet is too unreliable for this, especially in tandem
        where one foot hides the other.)"""
        d = base.get("ankle_depth_m")
        if d and None not in d.values():
            if abs(d["left"] - d["right"]) < 0.05:
                return None
            return max(d, key=d.get)
        low = {}
        for s in SIDE:
            ys = [y for y in (base["heel_y"][s], base["toe_y"][s]) if y is not None]
            if not ys:
                return None
            low[s] = max(ys)
        if abs(low["left"] - low["right"]) < 0.02 * (base["leg"] or 1.0):
            return None
        return min(low, key=low.get)          # higher in the image = further back

    def _setup_warnings(self, base) -> list[str]:
        warn = []
        if None in base["wrist_ratio"].values():
            warn.append("Wrists not visible: hands-on-hips can't be checked.")
        elif max(base["wrist_ratio"].values()) > self.cfg.hands_off_floor:
            warn.append("Hands don't look like they're on the hips at the start.")
        nd, dom = self.nondominant, OTHER[self.nondominant]
        if self.stance == "single" and base["foot_height_m"]:
            if base["foot_height_m"][dom] < 0.05:
                warn.append(f"Expected the {dom} (dominant) foot to be raised.")
        if self.stance == "tandem":
            back = self._back_foot(base)
            if back is not None and back != nd:
                warn.append(f"Expected the {nd} (non-dominant) foot to be in back.")
        if self.stance == "double" and base["ankle"]["left"] is not None and base["ankle"]["right"] is not None:
            gap = np.linalg.norm(base["ankle"]["left"] - base["ankle"]["right"]) / base["leg"]
            if gap > 0.35:
                warn.append("Feet don't look together.")
        return warn

    def conditions(self, m: dict | None, eyes_open: bool | None) -> dict:
        """True = error condition present, False = fine, None = can't tell this frame."""
        c = {k: None for k in ERROR_TYPES if k != "out_of_position"}
        c["eyes_open"] = eyes_open if self.cfg.track_eyes else None
        if m is None:
            return c
        b, cfg = self.base, self.cfg
        leg = m["leg"] or b["leg"]

        # hands off hips: either wrist clearly further from its hip than at the start
        flags = []
        for s in SIDE:
            r, r0 = m["wrist_ratio"][s], b["wrist_ratio"][s]
            if r is not None:
                limit = max(cfg.hands_off_floor, (r0 or 0) + cfg.hands_off_margin)
                flags.append(r > limit)
        c["hands_off_hips"] = any(flags) if flags else None

        # step / stumble / fall
        flags = []
        for s in self._stance_feet():
            a, a0 = m["ankle"][s], b["ankle"][s]
            if a is not None and a0 is not None:
                flags.append(float(np.linalg.norm(a - a0)) / leg > cfg.step_ratio)
        if self.stance == "single" and m["foot_height_m"] is not None:
            flags.append(m["foot_height_m"][OTHER[self.nondominant]] < cfg.touch_m)
        if m["hip_mid"] is not None:
            flags.append((m["hip_mid"][1] - b["hip_y"]) / leg > cfg.fall_ratio)
        c["step_stumble_fall"] = any(flags) if flags else None

        # hip flexion / abduction past 30 degrees, either leg (median of last frames)
        for s, a in self.hip_motion(m).items():
            self._hip_hist[s].append(a)
        smoothed = {s: median(h) for s, h in self._hip_hist.items() if h}
        self.hip_now = {s: round(a, 1) for s, a in smoothed.items()}
        c["hip_angle"] = (any(a > cfg.hip_angle_deg for a in smoothed.values()) if smoothed else None)

        # heel or forefoot of a stance foot comes up off the floor
        flags = []
        for s in self._stance_feet():
            for key in ("heel_y", "toe_y"):
                y, y0 = m[key][s], b[key][s]
                if y is not None and y0 is not None:
                    flags.append((y0 - y) / leg > cfg.foot_lift_ratio)
        c["foot_lift"] = any(flags) if flags else None
        return c

    def _score(self, t, m, eyes_open, events):
        cfg = self.cfg
        cond = self.conditions(m, eyes_open)
        self.coverage["frames"] += 1
        if any(v is not None for k, v in cond.items() if k != "eyes_open"):
            self.coverage["measured"] += 1

        new = []
        for k, tr in self._tracks.items():
            v = cond[k]
            if v is True:
                tr.off_since = None
                tr.on_since = tr.on_since if tr.on_since is not None else t
                if not tr.active and t - tr.on_since >= cfg.min_error_s:
                    tr.active = True
                    new.append((k, tr.on_since))
            elif v is False:
                tr.on_since = None
                if tr.active:
                    tr.off_since = tr.off_since if tr.off_since is not None else t
                    if t - tr.off_since >= cfg.clear_s:
                        tr.active, tr.off_since = False, None
            # None: can't see it this frame; keep the current state

        for k in self._manual_marks:
            new.append((k, t))
        self._manual_marks = []

        for k, t_start in sorted(new, key=lambda x: x[1]):
            self._record(k, t_start, t, events)

        # out of position: any error condition held continuously for > 5 s
        any_active = any(tr.active for tr in self._tracks.values())
        if any_active:
            self._any_since = self._any_since if self._any_since is not None else min(
                tr.on_since or t for tr in self._tracks.values() if tr.active)
            if not self._oop_counted and t - self._any_since > cfg.out_of_position_s:
                self._oop_counted = True
                self._record("out_of_position", t, t, events, simultaneous_rule=False)
        else:
            self._any_since, self._oop_counted = None, False

    def _record(self, kind, t_start, t, events, simultaneous_rule=True):
        cfg = self.cfg
        rel = round(t_start - self._run_t0, 2)
        if self.errors >= cfg.max_errors:
            counted, why = False, "max_reached"
        elif (simultaneous_rule and self._last_counted_t is not None
              and abs(t_start - self._last_counted_t) < cfg.simultaneous_s):
            counted, why = False, "simultaneous"
        else:
            counted, why = True, None
            self.errors += 1
            self.by_type[kind] += 1
            self._last_counted_t = t_start
        entry = {"error": kind, "label": ERROR_LABELS[kind], "t": max(rel, 0.0),
                 "counted": counted, "errors": self.errors}
        if why:
            entry["not_counted_reason"] = why
        self.log.append(entry)
        events.append({"kind": "bess_error", "stance": self.stance, **entry})

    def _finish(self, events):
        result = {
            "stance": self.stance,
            "label": STANCES[self.stance],
            "nondominant": self.nondominant,
            "errors": self.errors,
            "by_type": dict(self.by_type),
            "log": list(self.log),
            "warnings": list(self.warnings),
            "coverage": round(self.coverage["measured"] / max(1, self.coverage["frames"]), 2),
        }
        self.scores[self.stance] = result
        self._idle()
        events.append({"kind": "bess_done", **result, "session": self.session_summary()})

    # ---------------- reporting ----------------

    def session_summary(self) -> dict:
        per = {s: (self.scores[s]["errors"] if s in self.scores else None) for s in STANCES}
        done = [v for v in per.values() if v is not None]
        return {"scores": per, "total": sum(done), "complete": len(done) == len(STANCES),
                # per-stance breakdown by error type, and how much of each trial was measured
                "by_stance": {s: (dict(self.scores[s]["by_type"]) if s in self.scores else None)
                              for s in STANCES},
                "coverage": {s: (self.scores[s]["coverage"] if s in self.scores else None)
                             for s in STANCES},
                "error_types": list(self.error_types())}

    def _status(self, t, events, eyes_open) -> dict:
        out = {"phase": self.phase, "stance": self.stance, "nondominant": self.nondominant,
               "duration_s": self.cfg.duration_s,      # stance length, for the screens' timers
               "events": events, "session": self.session_summary()}
        if self.phase == "countdown":
            elapsed = t - self._t0 if self._t0 is not None else 0.0
            out["countdown_left"] = round(max(0.0, self.cfg.countdown_s - elapsed), 1)
            out["waiting_for_view"] = elapsed >= self.cfg.countdown_s
        elif self.phase == "running":
            out.update({
                "time_left": round(max(0.0, self.cfg.duration_s - (t - self._run_t0)), 1),
                "errors": self.errors,
                "by_type": dict(self.by_type),
                "active": [k for k, tr in self._tracks.items() if tr.active],
                "warnings": self.warnings,
                "log": self.log[-5:],
                "hip_angles": self.hip_now,
            })
            if self.cfg.track_eyes:
                out["eyes"] = {True: "open", False: "closed", None: "unknown"}[eyes_open]
        return out


# --------------------------------------------------------------------------- #
# Eyes closed check (face blendshapes on a crop around the head)
# --------------------------------------------------------------------------- #

class EyeClosure:
    """Is the person's eyes open? Uses MediaPipe FaceLandmarker blink scores on a
    head crop found from the pose landmarks, so it still works when the whole
    body is in frame. Returns None when the head is too small to judge."""

    MIN_HEAD_PX = 28

    def __init__(self, open_below: float = 0.5):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        from mp_models import ensure_face_model
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_face_model())),
            running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=True)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self.open_below = open_below
        self.last_score: float | None = None

    def is_open(self, frame_bgr, landmarks) -> bool | None:
        import cv2
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        if _vis(landmarks, NOSE, L_EAR, R_EAR) < 0.5:
            return None
        nose = _px(landmarks, NOSE, w, h)
        ear_d = float(np.linalg.norm(_px(landmarks, L_EAR, w, h) - _px(landmarks, R_EAR, w, h)))
        if ear_d < self.MIN_HEAD_PX:
            return None
        half = ear_d * 1.3
        x0, y0 = int(max(0, nose[0] - half)), int(max(0, nose[1] - half * 1.1))
        x1, y1 = int(min(w, nose[0] + half)), int(min(h, nose[1] + half * 0.9))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (256, int(256 * crop.shape[0] / max(crop.shape[1], 1))))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(img)
        if not res.face_blendshapes:
            return None
        scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
        blink = (scores.get("eyeBlinkLeft", 0) + scores.get("eyeBlinkRight", 0)) / 2
        self.last_score = blink
        return blink < self.open_below

    def close(self):
        self._lm.close()


def bess_messages(result: dict, frame_id=None) -> list[dict]:
    """Turn this frame's BESS events into WebSocket messages."""
    b = result.get("bess") or {}
    msgs = []
    for ev in b.get("events", []):
        kind = ev["kind"]
        if kind == "bess_error":
            msgs.append({"type": "event", "frame_id": frame_id, **ev})
        elif kind == "bess_done":
            msgs.append({"type": "result", "frame_id": frame_id, **ev})
        else:
            msgs.append({"type": "status", "frame_id": frame_id, **ev})
    return msgs


def config_from_dict(d: dict) -> BessConfig:
    return BessConfig(**{k: v for k, v in d.items() if k in asdict(BessConfig())})
