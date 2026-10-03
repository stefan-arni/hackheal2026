"""Speech test client: stands in for the iPhone app's audio stream.

Streams a WAV file (or your microphone) to the speech server for a whole
"call", records sections on command, and prints the final report.

Usage:
    python speech_server.py                                   # terminal 1
    python speech_client.py --file visit.wav --patient p-123  # terminal 2
    python speech_client.py --file visit.wav --record "reading:30-75"   # record 30s..75s
    python speech_client.py --mic --patient p-123             # live microphone
    python speech_client.py --file visit.wav --fast           # don't wait in real time

Microphone mode, type into this terminal:
    r <label>   start recording a section     s   stop and save it
    q           end the call and show the report
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np

SR = 16000
CHUNK_S = 0.25


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def load_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def parse_records(specs: list[str]) -> list[tuple[str, float, float]]:
    out = []
    for s in specs:
        try:
            label, rng = s.rsplit(":", 1)
            a, b = (float(v) for v in rng.split("-"))
        except ValueError:
            raise SystemExit(f"--record must look like 'label:START-END' in seconds, got {s!r}")
        out.append((label, a, b))
    return out


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #

def fmt(v, unit=""):
    return "-" if v is None else f"{v}{unit}"


def print_live(m):
    sr, v, p = m["speech_rate"], m["voice"], m["pauses"]
    rec = f"  [recording '{m['recording']}']" if m.get("recording") else ""
    print(f"   {m['elapsed_s']:6.0f}s  speech {m['speaking_time_s']:5.0f}s  "
          f"{fmt(sr['words_per_min'])} wpm  {fmt(sr['articulation_rate_syll_per_s'])} syll/s  "
          f"jitter {fmt(v['jitter_local_pct'], '%')}  shimmer {fmt(v['shimmer_local_pct'], '%')}  "
          f"pauses {p['count']} ({fmt(p['mean_s'], 's')} avg){rec}")


def print_section(m):
    mt = m["metrics"]
    where = m.get("saved") or {}
    loc = where.get("path") or where.get("row_id") or m.get("save_error")
    print(f">> section '{m['label']}' saved ({m['duration_s']:.1f}s) -> {where.get('backend', '?')}: {loc}")
    print(f"     speech rate {fmt(mt['speech_rate']['words_per_min'])} wpm, "
          f"jitter {fmt(mt['voice']['jitter_local_pct'], '%')}, "
          f"shimmer {fmt(mt['voice']['shimmer_local_pct'], '%')}, pauses {mt['pauses']['count']}")


def print_report(r):
    s = r["summary"]
    sr, v, p, lx = s["speech_rate"], s["voice"], s["pauses"], s["lexical"] or {}
    print("\n" + "=" * 66)
    print(f"SPEECH REPORT  patient {r['patient_id']}  visit {r['visit_id']}")
    print(f"call {r['call_duration_s']:.0f}s, patient speaking {s['speaking_time_s']:.0f}s")
    print("-" * 66)
    rows = [
        ("Speech rate", f"{fmt(sr['words_per_min'])} words/min, {fmt(sr['syllables_per_s'])} syll/s"),
        ("Articulation rate", f"{fmt(sr['articulation_rate_syll_per_s'])} syll/s (excluding pauses)"),
        ("Jitter", fmt(v["jitter_local_pct"], "%")),
        ("Shimmer", fmt(v["shimmer_local_pct"], "%")),
        ("Pitch / HNR", f"{fmt(v['f0_mean_hz'], ' Hz')} (sd {fmt(v['f0_sd_hz'])}), HNR {fmt(v['hnr_db'], ' dB')}"),
        ("Pauses", f"{p['count']} ({fmt(p['per_min'])}/min), mean {fmt(p['mean_s'], 's')}, "
                   f"max {fmt(p['max_s'], 's')}, {fmt(p['pause_time_pct'], '%')} of talk time"),
        ("Lexical richness", f"MATTR {fmt(lx.get('mattr'))}, TTR {fmt(lx.get('type_token_ratio'))}, "
                             f"{fmt(lx.get('words'))} words"),
        ("Lexical density", f"{fmt(lx.get('lexical_density'))} (content words / all words)"),
    ]
    for k, val in rows:
        print(f"  {k:18s} {val}")
    b = r["baseline"]
    print("-" * 66)
    print(f"  {b['summary_text']}")
    for d in b["deviations"]:
        mark = "  (lowers score)" if d["lowers_score"] and abs(d["z"]) >= 0.5 else ""
        print(f"    {d['label']:26s} {d['value']}  vs baseline {d['baseline_mean']}  "
              f"z {d['z']:+.2f} {d['direction']}{mark}")
    print(f"  {b['disclaimer']}")
    if r["sections"]:
        print("-" * 66)
        for sct in r["sections"]:
            print(f"  section '{sct['label']}' at {sct['started_offset_s']:.0f}s, {sct['duration_s']:.0f}s "
                  f"-> {(sct.get('saved') or {}).get('backend', sct.get('save_error'))}")
    print("-" * 66)
    print(f"  stored: {r.get('stored') or r.get('store_error')}")
    for n in r["notes"]:
        print(f"  note: {n}")
    print("=" * 66)


# --------------------------------------------------------------------------- #

class Client:
    def __init__(self, url):
        from websockets.sync.client import connect
        try:
            self.ws = connect(url, max_size=16 * 1024 * 1024, open_timeout=5)
        except (OSError, TimeoutError) as e:
            raise SystemExit(f"Could not connect to {url}: {e}\nIs speech_server.py running?") from None
        self.report = None
        self.done = threading.Event()
        self.closed = threading.Event()
        threading.Thread(target=self._recv, daemon=True).start()

    def send(self, obj) -> bool:
        """Send a command or audio chunk. Returns False once the connection is gone."""
        if self.closed.is_set():
            return False
        from websockets.exceptions import ConnectionClosed
        try:
            self.ws.send(json.dumps(obj) if isinstance(obj, dict) else obj)
            return True
        except ConnectionClosed as e:
            self._on_closed(e)
            return False

    def _on_closed(self, e=None):
        if self.closed.is_set():
            return
        self.closed.set()
        rcvd = getattr(e, "rcvd", None)
        code = rcvd.code if rcvd else None
        reason = (rcvd.reason if rcvd else "") or ""
        if self.report is None:
            print(f"\n>> connection closed by the server (code {code}{': ' + reason if reason else ''}).")
            if code == 1011:
                print("   The server hit an error: check the speech_server terminal for the traceback.")
        self.done.set()

    def _recv(self):
        try:
            for raw in self.ws:
                m = json.loads(raw)
                kind = m.get("kind")
                if m["type"] == "error":
                    print(f">> error: {m['error']}")
                elif kind == "speech_live":
                    print_live(m)
                elif kind == "section_recording":
                    print(f">> recording section '{m['label']}' from {m['started_offset_s']:.1f}s")
                elif kind == "section_saved":
                    print_section(m)
                elif kind == "speech_report":
                    self.report = m
                    print_report(m)
                    self.done.set()
                elif m["type"] == "ack" and m["command"] == "session_start":
                    print(f">> session started: patient {m['patient_id']}, visit {m['visit_id']}")
        except Exception as e:
            self._on_closed(e)
        finally:
            self._on_closed()


def run_file(c, args):
    x, sr = load_audio(args.file)
    records = parse_records(args.record or [])
    c.send({"type": "session_start", "patient_id": args.patient, "visit_id": args.visit,
            "sample_rate": sr})
    n = int(CHUNK_S * sr)
    active = None
    t0 = time.time()
    print(f">> streaming {args.file} ({len(x) / sr:.0f}s){' fast' if args.fast else ' in real time'}")
    for i in range(0, len(x), n):
        pos = i / sr
        for label, a, b in records:
            if active is None and a <= pos < b:
                c.send({"type": "record_start", "label": label})
                active = (label, b)
        if active and pos >= active[1]:
            c.send({"type": "record_stop"})
            active = None
        if not c.send(to_pcm16(x[i:i + n])):
            return
        if not args.fast:
            time.sleep(max(0.0, t0 + pos + CHUNK_S - time.time()))
    if active:
        c.send({"type": "record_stop"})
    c.send({"type": "session_end"})


def run_mic(c, args):
    try:
        import sounddevice as sd
    except Exception as e:
        raise SystemExit(f"Microphone needs the sounddevice package ({e}). pip install sounddevice")
    print(f">> connected to {args.url}")
    c.send({"type": "session_start", "patient_id": args.patient, "visit_id": args.visit,
            "sample_rate": SR})
    stop = threading.Event()

    def commands():
        print(">> type: r <label> = record section, s = stop & save, q = end call")
        for line in sys.stdin:
            line = line.strip()
            if c.closed.is_set():
                return
            if line.startswith("r"):
                c.send({"type": "record_start", "label": line[1:].strip() or "section"})
            elif line == "s":
                c.send({"type": "record_stop"})
            elif line == "q":
                stop.set()
                return

    threading.Thread(target=commands, daemon=True).start()

    def on_audio(indata, frames, t, status):
        if not c.send(to_pcm16(indata[:, 0])):
            stop.set()                            # connection gone: stop the mic quietly

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=int(CHUNK_S * SR), callback=on_audio):
        print(">> listening... (q + Enter to end the call)")
        while not stop.wait(0.2):
            if c.closed.is_set():
                break
    c.send({"type": "session_end"})


def main():
    p = argparse.ArgumentParser(description="Test client for the speech server")
    p.add_argument("--url", default="ws://127.0.0.1:8767")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="WAV/FLAC file to stream as the call audio")
    src.add_argument("--mic", action="store_true", help="stream the microphone")
    p.add_argument("--patient", default="test-patient")
    p.add_argument("--visit", help="visit id (default: generated)")
    p.add_argument("--record", action="append",
                   help="file mode: record a section, 'label:START-END' in seconds (repeatable)")
    p.add_argument("--fast", action="store_true", help="file mode: send as fast as possible")
    p.add_argument("--json-out", help="also save the final report to this JSON file")
    args = p.parse_args()

    c = Client(args.url)
    (run_mic if args.mic else run_file)(c, args)
    c.done.wait(timeout=600)
    if args.json_out and c.report:
        with open(args.json_out, "w") as f:
            json.dump(c.report, f, indent=2, default=str)
        print(f">> report saved to {args.json_out}")


if __name__ == "__main__":
    main()
