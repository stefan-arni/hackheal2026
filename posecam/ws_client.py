"""Shared helpers for the laptop test clients (pose and eyes)."""

from __future__ import annotations

import base64
import json
import time
from contextlib import ExitStack

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_frame(frame, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def open_source(source: str):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"could not open source {source!r} "
                         "(check camera permissions, or try --source 1)")
    return cap


def resize_to_width(frame, width: int):
    if width and frame.shape[1] > width:
        scale = width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


class RemoteClient:
    """Sends frames over WebSocket the way the iPhone app will, returns each result.

    `result_type` is the reply type to wait for ("pose" or "eyes"). Other message
    types the server sends in between (alerts, status, pong) are skipped here;
    the clients read the same information from the result itself.
    """

    def __init__(self, url: str, result_type: str, use_json: bool = False, timeout: float = 15.0):
        from websockets.sync.client import connect
        self._stack = ExitStack()  # newer websockets wants connect() used as a context manager
        try:
            self.ws = self._stack.enter_context(
                connect(url, max_size=8 * 1024 * 1024, open_timeout=5))
        except (OSError, TimeoutError) as e:
            raise SystemExit(f"Could not connect to {url}: {e}\n"
                             "Is the server running in the other terminal?") from None
        self.result_type = result_type
        self.use_json = use_json
        self.timeout = timeout
        self.frame_id = 0
        self.replies: list[dict] = []   # replies to commands (acks, command errors)

    def __call__(self, frame, quality: int = 80, timestamp_ms: float | None = None) -> dict:
        """timestamp_ms: when the frame was captured (default: now). Only sent in
        JSON mode; binary frames carry no metadata."""
        jpeg = encode_frame(frame, quality)
        self.frame_id += 1
        if self.use_json:
            self.ws.send(json.dumps({
                "type": "frame", "frame_id": self.frame_id,
                "timestamp_ms": round(timestamp_ms if timestamp_ms is not None else time.time() * 1000, 3),
                "image": base64.b64encode(jpeg).decode("ascii"),
            }))
        else:
            self.ws.send(jpeg)
        while True:
            try:
                msg = json.loads(self.ws.recv(timeout=self.timeout))
            except TimeoutError:
                raise SystemExit(
                    f"Server accepted the frame but didn't reply within {self.timeout:.0f}s.\n"
                    "Check the server terminal for an error. On Windows, if you clicked inside\n"
                    "that window it may be paused (title starts with 'Select'): press Esc there."
                ) from None
            if msg.get("type") == self.result_type:
                return msg
            if msg.get("type") == "ack" or (msg.get("type") == "error" and "command" in msg):
                self.replies.append(msg)
            elif msg.get("type") == "error":
                raise SystemExit(f"Server error: {msg['error']}")

    def send_json(self, obj: dict):
        self.ws.send(json.dumps(obj))

    def take_replies(self) -> list[dict]:
        r, self.replies = self.replies, []
        return r

    def close(self):
        self._stack.close()
