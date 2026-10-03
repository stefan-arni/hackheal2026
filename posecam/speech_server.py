"""Speech server: analyzes a patient's speech for the whole call.

Run:
    python speech_server.py                  # ws://0.0.0.0:8767
    python speech_server.py --storage local  # save to ./speech_data instead of Supabase
    python speech_server.py --no-transcribe  # skip speech-to-text (no words/min, no lexical)

Supabase is used when SUPABASE_URL and SUPABASE_SERVICE_KEY are set, in the
environment or in a .env file next to this script (copy .env.example; see
supabase_schema.sql).

At the end of each call the voice features are scored against the patient's
prior sessions with voice.py (personal z-scores + 0-100 voice score).

Protocol (client -> server)
  {"type": "session_start", "patient_id": "p-123", "visit_id": "optional", "sample_rate": 16000}
  <binary>  raw audio: 16-bit little-endian mono PCM at the session's sample rate
            (send ~100-500 ms chunks for the whole call)
  {"type": "record_start", "label": "reading passage"}   start recording a section
  {"type": "record_stop"}                                stop it -> saved (Supabase)
  {"type": "session_end"}                                end of call -> final report
  {"type": "status"} / {"type": "ping"}

Protocol (server -> client)
  {"type": "ack", "command": ..., ...}
  {"type": "status", "kind": "speech_live", "elapsed_s", "speaking_time_s", "speech_rate",
   "voice", "pauses", "words", "recording"}                         every ~5 s of audio
  {"type": "status", "kind": "section_recording", "label", ...}
  {"type": "result", "kind": "section_saved", "label", "duration_s", "metrics", "saved"}
  {"type": "result", "kind": "speech_report", "summary", "comparison", "sections", ...}
  {"type": "error", "error": ..., "command": ...}

If the connection drops before session_end, the call is ended and stored anyway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading

import websockets

from speech_analysis import SpeechConfig, make_transcriber, pcm16_to_float
from speech_session import CallSession
from ws_server import lan_ip

log = logging.getLogger("speech-server")


class SpeechConnection:
    """Everything for one call runs on one worker thread, in arrival order."""

    def __init__(self, ws, loop, make_session, status_every_s: float = 5.0):
        self.ws, self.loop = ws, loop
        self.make_session = make_session
        self.status_every_s = status_every_s
        self.session: CallSession | None = None
        self._next_status = status_every_s
        self.q: queue.Queue = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # -- called from the event loop --
    def put(self, item):
        self.q.put(item)

    def close(self):
        self.q.put(("disconnect", None))
        self.worker.join(timeout=600)

    # -- worker thread --
    def _send(self, msg: dict):
        try:
            asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(msg, default=str)), self.loop)
        except Exception:
            pass

    def _run(self):
        while True:
            kind, payload = self.q.get()
            try:
                if kind == "audio":
                    self._audio(payload)
                elif kind == "cmd":
                    self._command(payload)
                elif kind == "disconnect":
                    if self.session is not None and not self.session.ended:
                        r = self.session.end()
                        log.warning("connection dropped: call ended and stored (%s, %.0fs)",
                                    r["visit_id"], r["call_duration_s"])
                    return
            except Exception as e:
                log.exception("speech processing failed")
                cmd = payload.get("type") if isinstance(payload, dict) else None
                self._send({"type": "error", "error": f"{e.__class__.__name__}: {e}", "command": cmd})

    def _ensure_session(self):
        if self.session is None or self.session.ended:
            self.session = self.make_session({})
            log.warning("audio before session_start: started a session for patient 'unknown'")
            self._send({"type": "status", "kind": "session_started", "patient_id": "unknown",
                        "visit_id": self.session.visit_id,
                        "note": "Send session_start with patient_id before audio."})

    def _audio(self, data: bytes):
        backlog = self.q.qsize() * 0.25          # ~seconds of audio waiting (250 ms chunks)
        if backlog > 10 and not getattr(self, "_warned_backlog", False):
            self._warned_backlog = True
            log.warning("analysis is falling behind real time (%.0f s of audio queued); "
                        "try --whisper-model tiny.en or --no-transcribe", backlog)
        self._ensure_session()
        for m in self.session.feed(pcm16_to_float(data)):
            self._send(m)
        if self.session.elapsed_s >= self._next_status:
            self._next_status = self.session.elapsed_s + self.status_every_s
            self._send(self.session.live())

    def _command(self, msg: dict):
        t = msg.get("type")
        if t == "session_start":
            if self.session is not None and not self.session.ended:
                self.session.end()
            self.session = self.make_session(msg)
            self._next_status = self.status_every_s
            log.info("session started: patient %s, visit %s", self.session.patient_id,
                     self.session.visit_id)
            self._send({"type": "ack", "command": t, "patient_id": self.session.patient_id,
                        "visit_id": self.session.visit_id, "sample_rate": self.session.in_sr})
        elif t == "record_start":
            self._ensure_session()
            m = self.session.start_section(msg.get("label", "section"))
            log.info("recording section '%s'", m["label"])
            self._send({"type": "ack", "command": t, "label": m["label"]})
            for prev in m["stopped_previous"]:
                self._send(prev)
            self._send({k: v for k, v in m.items() if k != "stopped_previous"})
        elif t == "record_stop":
            if self.session is None or self.session.section is None:
                self._send({"type": "error", "error": "no section is being recorded", "command": t})
                return
            m = self.session.stop_section()
            log.info("section '%s' saved (%.1fs) -> %s", m["label"], m["duration_s"],
                     m.get("saved") or m.get("save_error"))
            self._send({"type": "ack", "command": t})
            self._send(m)
        elif t == "session_end":
            if self.session is None:
                self._send({"type": "error", "error": "no session", "command": t})
                return
            r = self.session.end()
            s = r["summary"]
            log.warning("CALL ENDED %s: %.0fs, speech %.0fs, %s syll/s, jitter %s%%, shimmer %s%% | %s",
                        r["visit_id"], r["call_duration_s"], s["speaking_time_s"],
                        s["speech_rate"]["syllables_per_s"], s["voice"]["jitter_local_pct"],
                        s["voice"]["shimmer_local_pct"], r["baseline"]["summary_text"])
            self._send({"type": "ack", "command": t})
            self._send(r)
        elif t == "status":
            self._send(self.session.live() if self.session else
                       {"type": "status", "kind": "no_session"})
        else:
            self._send({"type": "error", "error": f"unknown message type {t!r}", "command": t})


def make_handler(make_session):
    async def handler(ws):
        loop = asyncio.get_running_loop()
        conn = SpeechConnection(ws, loop, make_session)
        log.info("client connected: %s", ws.remote_address)
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    conn.put(("audio", bytes(message)))
                    continue
                try:
                    msg = json.loads(message)
                except ValueError:
                    await ws.send(json.dumps({"type": "error", "error": "invalid JSON"}))
                    continue
                if msg.get("type") == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                else:
                    conn.put(("cmd", msg))
        except websockets.ConnectionClosed as e:
            if e.rcvd is None and e.sent is not None and e.sent.code == 1011:
                log.warning("connection closed by the server: %s", e.sent.reason or "internal error")
        except Exception:
            log.exception("connection handler failed")
            raise
        finally:
            await asyncio.to_thread(conn.close)
            log.info("client disconnected: %s", ws.remote_address)
    return handler


async def serve(host, port, make_session):
    # generous keepalive: a slow laptop busy transcribing must not drop the call
    async with websockets.serve(make_handler(make_session), host, port, max_size=16 * 1024 * 1024,
                                ping_interval=20, ping_timeout=120):
        log.info("speech server listening on ws://%s:%d  (iPhone: ws://%s:%d)", host, port, lan_ip(), port)
        await asyncio.Future()


def main():
    p = argparse.ArgumentParser(description="Speech analysis WebSocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--storage", choices=["auto", "supabase", "local"], default="auto",
                   help="auto = Supabase if SUPABASE_URL/SUPABASE_SERVICE_KEY are set, else local files")
    p.add_argument("--no-transcribe", action="store_true", help="skip speech-to-text")
    p.add_argument("--whisper-model", default="base.en",
                   help="faster-whisper model: tiny.en (fastest), base.en, small.en (most accurate)")
    p.add_argument("--max-pause", type=float, default=3.0,
                   help="silences longer than this (s) are turn gaps, not pauses")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from env import load_env
    from speech_storage import make_storage
    env_files = load_env()          # .env -> SUPABASE_URL, SUPABASE_SERVICE_KEY
    if env_files:
        log.info("settings loaded from %s", ", ".join(env_files))
    sink, store, where = make_storage(args.storage)
    log.info("storage: %s", where)

    transcriber = None if args.no_transcribe else make_transcriber("whisper", args.whisper_model)
    if transcriber:
        log.info("transcription: faster-whisper %s", args.whisper_model)
    cfg = SpeechConfig(max_pause_s=args.max_pause)

    def make_session(msg: dict) -> CallSession:
        return CallSession(patient_id=str(msg.get("patient_id") or "unknown"),
                           visit_id=msg.get("visit_id"),
                           sample_rate=int(msg.get("sample_rate") or 16000),
                           cfg=SpeechConfig(**vars(cfg)), transcriber=transcriber,
                           sink=sink, store=store)

    try:
        asyncio.run(serve(args.host, args.port, make_session))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
