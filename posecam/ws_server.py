"""Shared WebSocket plumbing used by both servers (pose and eyes).

Handles: decoding frames (binary JPEG or base64 JSON), keeping only the newest
frame when the model is busy, running the model off the event loop, ping/pong,
recalibrate requests and error reporting. It knows nothing about pose or eyes:
each server passes in an analyzer factory, the reply message type, and an
optional function that turns a result into extra messages (e.g. alerts).

Client -> server:
  - Binary message: raw JPEG bytes of one frame.
  - {"type": "frame", "image": "<base64 JPEG>", "frame_id": 42,
     "timestamp_ms": 1696291200000, "rotate": 0}
    `rotate` (0/90/180/270, clockwise) is optional, for frames that arrive sideways.
    Optional depth map (LiDAR / TrueDepth): "depth", "depth_format", "depth_size",
    "intrinsics", see depth.py. It's rotated with the image and handed to the
    analyzer as meta["_depth"] (metres) and meta["_intrinsics"].
  - {"type": "record_start", "name": "optional"} / {"type": "record_stop"} ->
    save every incoming message to recordings/<name>.jsonl (replay: playback.py)
  - {"type": "ping"} -> {"type": "pong"}
  - {"type": "recalibrate"} -> calls analyzer.recalibrate() before the next frame
    (only if the analyzer has one)
  - any other {"type": ...} -> analyzer.handle_command(msg), if it has one; its
    replies (e.g. {"type": "ack"} or {"type": "error"}) are sent back right away
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
import threading
from typing import Callable

import cv2
import numpy as np
import websockets

log = logging.getLogger("ws-server")

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

EventsFn = Callable[[dict, object], list]


def decode_message(message) -> tuple[np.ndarray | None, dict]:
    """Turn an incoming message into (BGR frame or None, metadata)."""
    if isinstance(message, (bytes, bytearray)):
        meta, jpeg = {}, bytes(message)
    else:
        meta = json.loads(message)
        if meta.get("type") != "frame":
            return None, meta
        jpeg = base64.b64decode(meta.pop("image"))

    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode JPEG")
    rot = int(meta.get("rotate", 0)) % 360
    if rot in ROTATIONS:
        frame = cv2.rotate(frame, ROTATIONS[rot])
    if "depth" in meta:   # LiDAR / TrueDepth map from the iPhone, rotated the same way
        from depth import decode_depth
        meta["_depth"], meta["_intrinsics"] = decode_depth(meta, rot)
    return frame, meta


class Recorder:
    """Saves every incoming message (frames incl. depth, and commands) as one JSON
    line each, so a session can be replayed exactly with playback.py."""

    def __init__(self, directory: str = "recordings", name: str | None = None):
        import os
        import time as _time
        os.makedirs(directory, exist_ok=True)
        name = name or _time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c for c in name if c.isalnum() or c in "-_") or "session"
        self.path = os.path.join(directory, safe + ".jsonl")
        self._f = open(self.path, "a")
        self.messages = 0

    def write(self, message):
        if isinstance(message, (bytes, bytearray)):
            message = json.dumps({"type": "frame", "image": base64.b64encode(message).decode("ascii")})
        self._f.write(message.rstrip("\n") + "\n")
        self.messages += 1

    def close(self):
        self._f.close()


class Connection:
    """One client stream. Keeps only the newest frame and processes it."""

    def __init__(self, ws, analyzer, result_type: str, events_fn: EventsFn | None = None,
                 record_dir: str | None = None, record_all: bool = False):
        self.ws = ws
        self.record_dir = record_dir or "recordings"
        self.recorder: Recorder | None = Recorder(self.record_dir) if record_all else None
        if self.recorder:
            log.info("recording to %s", self.recorder.path)
        self.analyzer = analyzer
        self.result_type = result_type
        self.events_fn = events_fn
        self.latest: tuple[np.ndarray, dict] | None = None
        self.ready = asyncio.Event()
        self.dropped = 0
        self.processed = 0
        self.lock = threading.Lock()  # never close the model while it's mid-frame
        self.recalibrate_pending = False
        # analyzers that accept the frame's metadata (e.g. its capture timestamp) get it
        import inspect
        try:
            self._takes_meta = "meta" in inspect.signature(analyzer.process_bgr).parameters
        except (TypeError, ValueError):
            self._takes_meta = False

    def _run(self, frame, meta=None):
        with self.lock:
            if self.recalibrate_pending and hasattr(self.analyzer, "recalibrate"):
                self.analyzer.recalibrate()
                self.recalibrate_pending = False
            result = (self.analyzer.process_bgr(frame, meta=meta) if self._takes_meta
                      else self.analyzer.process_bgr(frame))
        return result.to_dict() if hasattr(result, "to_dict") else result

    def _close(self):
        with self.lock:
            self.analyzer.close()

    def _command(self, msg):
        with self.lock:
            return self.analyzer.handle_command(msg)

    def _record_command(self, meta) -> dict | None:
        """record_start {name?} / record_stop. Returns the reply, or None."""
        kind = meta.get("type")
        if kind == "record_start":
            if self.recorder is None:
                self.recorder = Recorder(self.record_dir, meta.get("name"))
                log.info("recording to %s", self.recorder.path)
            return {"type": "ack", "command": kind, "recording": True, "file": self.recorder.path}
        if kind == "record_stop":
            out = {"type": "ack", "command": kind, "recording": False}
            if self.recorder is not None:
                out.update(file=self.recorder.path, messages=self.recorder.messages)
                log.info("recording stopped: %s (%d messages)", self.recorder.path, self.recorder.messages)
                self.recorder.close()
                self.recorder = None
            return out
        return None

    def close_recorder(self):
        if self.recorder is not None:
            log.info("recording saved: %s (%d messages)", self.recorder.path, self.recorder.messages)
            self.recorder.close()
            self.recorder = None

    async def receive_loop(self):
        async for message in self.ws:
            if self.recorder is not None:
                self.recorder.write(message)
            try:
                frame, meta = decode_message(message)
            except Exception as e:  # bad payload: tell the client, keep going
                await self.ws.send(json.dumps({"type": "error", "error": str(e)}))
                continue
            if frame is None:
                if meta.get("type") == "ping":
                    await self.ws.send(json.dumps({"type": "pong"}))
                elif meta.get("type") in ("record_start", "record_stop"):
                    await self.ws.send(json.dumps(self._record_command(meta)))
                elif meta.get("type") == "recalibrate":
                    self.recalibrate_pending = True
                    log.info("recalibration requested")
                elif hasattr(self.analyzer, "handle_command"):
                    try:
                        replies = await asyncio.to_thread(self._command, meta)
                    except Exception as e:
                        log.exception("command failed")
                        replies = [{"type": "error", "error": f"command failed: {e!r}",
                                    "command": meta.get("type")}]
                    for r in replies:
                        await self.ws.send(json.dumps(r))
                else:
                    await self.ws.send(json.dumps(
                        {"type": "error", "error": f"unknown message type {meta.get('type')!r}",
                         "command": meta.get("type")}))
                continue
            if self.latest is not None:
                self.dropped += 1
            self.latest = (frame, meta)
            self.ready.set()

    async def process_loop(self):
        while True:
            await self.ready.wait()
            self.ready.clear()
            frame, meta = self.latest
            self.latest = None
            # MediaPipe is blocking; run it off the event loop.
            try:
                result = await asyncio.to_thread(self._run, frame, meta)
            except Exception as e:
                log.exception("frame processing failed")
                await self.ws.send(json.dumps({"type": "error", "error": f"processing failed: {e!r}"}))
                continue
            self.processed += 1
            if self.processed == 1:
                log.info("first frame processed (%dx%d, %.0f ms)",
                         frame.shape[1], frame.shape[0], result.get("inference_ms", 0))
            frame_id = meta.get("frame_id", self.processed)
            await self.ws.send(json.dumps({
                "type": self.result_type,
                "frame_id": frame_id,
                "client_timestamp_ms": meta.get("timestamp_ms"),
                "image_size": [frame.shape[1], frame.shape[0]],
                **result,
                "dropped_frames": self.dropped,
            }))
            if self.events_fn:
                for msg in self.events_fn(result, frame_id):
                    await self.ws.send(json.dumps(msg))


def make_handler(analyzer_factory, result_type: str = "result", events_fn: EventsFn | None = None,
                 record_dir: str | None = None, record_all: bool = False):
    async def handler(ws):
        peer = ws.remote_address
        log.info("client connected: %s (loading model...)", peer)
        try:
            analyzer = await asyncio.to_thread(analyzer_factory)
        except Exception as e:
            log.exception("could not create analyzer")
            await ws.send(json.dumps({"type": "error", "error": f"model failed to load: {e!r}"}))
            return
        log.info("model ready for %s", peer)
        conn = Connection(ws, analyzer, result_type, events_fn, record_dir, record_all)
        worker = asyncio.create_task(conn.process_loop())
        try:
            await conn.receive_loop()
        except websockets.ConnectionClosed:
            pass
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            conn.close_recorder()
            await asyncio.to_thread(conn._close)
            log.info("client %s disconnected (processed=%d dropped=%d)",
                     peer, conn.processed, conn.dropped)
    return handler


def lan_ip() -> str:
    """Best guess at this machine's LAN IP, for the iPhone to connect to."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


async def serve(host: str, port: int, analyzer_factory, result_type: str = "result",
                events_fn: EventsFn | None = None, name: str = "server",
                record_dir: str | None = None, record_all: bool = False):
    async with websockets.serve(make_handler(analyzer_factory, result_type, events_fn,
                                             record_dir, record_all),
                                host, port, max_size=8 * 1024 * 1024):
        log.info("%s listening on ws://%s:%d  (iPhone: ws://%s:%d)",
                 name, host, port, lan_ip(), port)
        await asyncio.Future()
