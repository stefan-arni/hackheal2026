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
import time
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
    return frame, meta


class Connection:
    """One client stream. Keeps only the newest frame and processes it."""

    def __init__(self, ws, analyzer, result_type: str, events_fn: EventsFn | None = None, forwarder=None,
                 publisher=None):
        self.ws = ws
        self.forwarder = forwarder  # optional replay_forward.ReplayForwarder (frames + events)
        self.publisher = publisher  # optional replay_publish.LivePublisher (dashboard summaries + events)
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

    async def receive_loop(self):
        async for message in self.ws:
            try:
                frame, meta = decode_message(message)
            except Exception as e:  # bad payload: tell the client, keep going
                await self.ws.send(json.dumps({"type": "error", "error": str(e)}))
                continue
            if frame is None:
                if meta.get("type") == "ping":
                    await self.ws.send(json.dumps({"type": "pong"}))
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
            if self.forwarder is not None or self.publisher is not None:
                recv_ms = time.time() * 1000
                if self.forwarder is not None:  # every received frame (even ones the model drops)
                    self._offer(message, frame, meta, recv_ms)
                meta = {**meta, "server_recv_ms": recv_ms}  # copy: the analyzer's meta is unchanged otherwise
            if self.latest is not None:
                self.dropped += 1
            self.latest = (frame, meta)
            self.ready.set()

    def _offer(self, message, frame, meta, recv_ms):
        """Hand the full frame to the forwarder: the JPEG as received, re-encoded only if rotated."""
        try:
            if int(meta.get("rotate", 0)) % 360:
                jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            elif isinstance(message, (bytes, bytearray)):
                jpeg = bytes(message)
            else:
                jpeg = base64.b64decode(json.loads(message)["image"])
            self.forwarder.offer_frame(jpeg, meta.get("timestamp_ms"), (frame.shape[1], frame.shape[0]), recv_ms)
        except Exception:
            log.exception("replay forward: could not offer frame")

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
            msgs = []
            if self.events_fn:
                msgs = self.events_fn(result, frame_id)
                for msg in msgs:  # every event carries the capture time of the frame that triggered it
                    msg.setdefault("client_timestamp_ms", meta.get("timestamp_ms"))
                    await self.ws.send(json.dumps(msg))
                if self.forwarder is not None and msgs:
                    self.forwarder.on_messages(msgs, meta.get("timestamp_ms"), meta.get("server_recv_ms"))
            if self.publisher is not None:
                try:
                    self.publisher.on_result(result, msgs, meta.get("timestamp_ms"), meta.get("server_recv_ms"),
                                             (frame.shape[1], frame.shape[0]))
                except Exception:
                    log.exception("live publish: could not queue result")


def make_handler(analyzer_factory, result_type: str = "result", events_fn: EventsFn | None = None,
                 *, forwarder_factory=None, publisher_factory=None):
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
        forwarder = forwarder_factory() if forwarder_factory else None
        conn = Connection(ws, analyzer, result_type, events_fn, forwarder=forwarder,
                          publisher=publisher_factory(forwarder) if publisher_factory else None)
        worker = asyncio.create_task(conn.process_loop())
        try:
            await conn.receive_loop()
        except websockets.ConnectionClosed:
            pass
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
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
                events_fn: EventsFn | None = None, name: str = "server", forwarder_factory=None,
                publisher_factory=None):
    async with websockets.serve(make_handler(analyzer_factory, result_type, events_fn,
                                             forwarder_factory=forwarder_factory,
                                             publisher_factory=publisher_factory),
                                host, port, max_size=8 * 1024 * 1024):
        log.info("%s listening on ws://%s:%d  (iPhone: ws://%s:%d)",
                 name, host, port, lan_ip(), port)
        await asyncio.Future()
