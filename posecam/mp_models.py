"""Shared MediaPipe model downloads (no tracking code here).

Both the eye server (iris tracking) and the pose server (BESS eyes-closed
check) use the face model file; each keeps its own code that runs it.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")


def ensure_face_model() -> Path:
    """Download the FaceLandmarker model on first use and return its path."""
    path = MODEL_DIR / "face_landmarker.task"
    if not path.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face model -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        tmp.rename(path)
    return path
