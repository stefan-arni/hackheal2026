"""MediaPipe full-body pose detection + spine angle calculation.

Uses the MediaPipe Tasks PoseLandmarker (33 body landmarks). Recent MediaPipe
releases removed the legacy `mp.solutions.pose` API, so this uses the Tasks API.

MediaPipe has no landmarks on the spine itself, so the "spine" here is the
trunk line from the hip midpoint to the shoulder midpoint.
"""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = Path(__file__).parent / "models"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# Skeleton edges for drawing: (start_idx, end_idx)
POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]


def ensure_model(variant: str = "full") -> Path:
    """Download the .task model on first use and return its path."""
    if variant not in MODEL_URLS:
        raise ValueError(f"model variant must be one of {list(MODEL_URLS)}")
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose model ({variant}) -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[variant], tmp)
        tmp.rename(path)
    return path


# --------------------------------------------------------------------------- #
# Spine angle math (pure functions, easy to unit test)
# --------------------------------------------------------------------------- #

def _mid(a, b):
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def spine_angles_2d(landmarks, width: int, height: int) -> dict:
    """Trunk angle in the image plane.

    `landmarks` is a list of 33 objects/dicts with normalized x, y (0..1).
    Returns the angle of hip-mid -> shoulder-mid from image vertical, in degrees.
    Positive = shoulders are to the image-right of the hips.

    From a SIDE view this is the forward/backward lean (most reliable).
    From a FRONT view this is the sideways (lateral) lean.
    """
    def px(i):
        lm = landmarks[i]
        x, y = (lm["x"], lm["y"]) if isinstance(lm, dict) else (lm.x, lm.y)
        return np.array([x * width, y * height])

    hip = _mid(px(L_HIP), px(R_HIP))
    sho = _mid(px(L_SHOULDER), px(R_SHOULDER))
    v = sho - hip                       # image y grows downward
    angle = math.degrees(math.atan2(v[0], -v[1]))
    return {
        "trunk_angle_deg": round(angle, 2),
        "hip_mid_px": [round(float(hip[0]), 1), round(float(hip[1]), 1)],
        "shoulder_mid_px": [round(float(sho[0]), 1), round(float(sho[1]), 1)],
    }


def spine_angles_3d(world_landmarks) -> dict:
    """Trunk angles from MediaPipe world landmarks (meters, origin at hip centre).

    MediaPipe world axes: +x = image right, +y = DOWN, +z = away from camera.
    Returns:
      inclination_deg: total angle of the trunk from vertical (0 = upright)
      flexion_deg:     sagittal lean, + = leaning toward the camera
      lateral_deg:     sideways lean, + = toward image-right
    Depth (z) is estimated from a single camera, so flexion from a front view
    is noisy; prefer the 2D angle from a side view for forward lean.
    """
    def pt(i):
        lm = world_landmarks[i]
        return np.array([lm["x"], lm["y"], lm["z"]] if isinstance(lm, dict) else [lm.x, lm.y, lm.z])

    hip = _mid(pt(L_HIP), pt(R_HIP))
    sho = _mid(pt(L_SHOULDER), pt(R_SHOULDER))
    v = sho - hip
    up = -v[1]                          # flip y so "up" is positive
    length = float(np.linalg.norm(v))
    inclination = math.degrees(math.acos(max(-1.0, min(1.0, up / length)))) if length > 1e-6 else 0.0
    return {
        "inclination_deg": round(inclination, 2),
        "flexion_deg": round(math.degrees(math.atan2(-v[2], up)), 2),
        "lateral_deg": round(math.degrees(math.atan2(v[0], up)), 2),
        "trunk_length_m": round(length, 3),
    }


def trunk_visibility(landmarks) -> float:
    """Minimum visibility of the 4 landmarks the spine angle depends on."""
    vals = []
    for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        lm = landmarks[i]
        vals.append(lm["visibility"] if isinstance(lm, dict) else (lm.visibility or 0.0))
    return float(min(vals))


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #

@dataclass
class PoseResult:
    detected: bool
    landmarks: list | None = None          # normalized image coords
    world_landmarks: list | None = None    # meters, hip-centred
    spine: dict | None = None
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "landmarks": self.landmarks,
            "world_landmarks": self.world_landmarks,
            "spine": self.spine,
            "inference_ms": round(self.inference_ms, 1),
        }


class PoseAnalyzer:
    """Wraps MediaPipe PoseLandmarker in VIDEO mode (uses temporal tracking).

    Create one analyzer per video stream: VIDEO mode needs increasing timestamps
    from a single source.
    """

    def __init__(self, model: str = "full", min_visibility: float = 0.5,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model(model))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        self.min_visibility = min_visibility

    def _next_ts(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process_bgr(self, frame_bgr: np.ndarray) -> PoseResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t0 = time.perf_counter()
        res = self._landmarker.detect_for_video(image, self._next_ts())
        ms = (time.perf_counter() - t0) * 1000

        if not res.pose_landmarks:
            return PoseResult(detected=False, inference_ms=ms)

        lms = res.pose_landmarks[0]
        wlms = res.pose_world_landmarks[0]
        landmarks = [
            {"name": LANDMARK_NAMES[i], "x": lm.x, "y": lm.y, "z": lm.z,
             "visibility": lm.visibility, "presence": lm.presence}
            for i, lm in enumerate(lms)
        ]
        world = [{"name": LANDMARK_NAMES[i], "x": lm.x, "y": lm.y, "z": lm.z,
                  "visibility": lm.visibility} for i, lm in enumerate(wlms)]

        vis = trunk_visibility(landmarks)
        spine = {
            **spine_angles_2d(landmarks, w, h),
            **spine_angles_3d(world),
            "trunk_visibility": round(vis, 3),
            "reliable": vis >= self.min_visibility,
        }
        return PoseResult(True, landmarks, world, spine, ms)

    def process_jpeg(self, data: bytes) -> PoseResult:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("could not decode image bytes")
        return self.process_bgr(frame)

    def close(self):
        self._landmarker.close()


# --------------------------------------------------------------------------- #
# Drawing helper (used by the laptop test client)
# --------------------------------------------------------------------------- #

def draw_pose(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw skeleton + spine line + angles onto a BGR frame (in place)."""
    h, w = frame.shape[:2]
    if not result.get("detected"):
        cv2.putText(frame, "No person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame

    lms = result["landmarks"]
    pts = [(int(l["x"] * w), int(l["y"] * h)) for l in lms]
    for a, b in POSE_CONNECTIONS:
        if lms[a]["visibility"] > 0.5 and lms[b]["visibility"] > 0.5:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p, l in zip(pts, lms):
        if l["visibility"] > 0.5:
            cv2.circle(frame, p, 4, (0, 200, 255), -1)

    s = result["spine"]
    hip = tuple(int(v) for v in s["hip_mid_px"])
    sho = tuple(int(v) for v in s["shoulder_mid_px"])
    color = (0, 255, 0) if s["reliable"] else (0, 165, 255)
    cv2.line(frame, hip, sho, color, 4)
    cv2.line(frame, hip, (hip[0], hip[1] - 120), (180, 180, 180), 1)  # vertical ref

    lines = [
        f"2D trunk angle: {s['trunk_angle_deg']:+.1f} deg",
        f"3D incline: {s['inclination_deg']:.1f}  flex: {s['flexion_deg']:+.1f}  lat: {s['lateral_deg']:+.1f}",
        f"trunk visibility: {s['trunk_visibility']:.2f}" + ("" if s["reliable"] else "  (low)"),
    ]
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame
