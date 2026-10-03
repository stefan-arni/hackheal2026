"""Speech analysis tests on synthetic speech (no microphone, model or network needed).

    pytest -q
"""

import asyncio
import json

import numpy as np
import pytest
import websockets

import speech_server
from speech_analysis import (SpeechAnalyzer, analyze_audio, lexical_measures, pcm16_to_float,
                             resample)
from speech_session import CallSession
from speech_storage import (LocalRecordingSink, LocalVisitStore, SupabaseRecordingSink,
                            SupabaseVisitStore)
from synth_speech import SR, utterance

PATTERN = [("pause", 1.0), ("speech", 3.0), ("pause", 0.6), ("speech", 2.0), ("pause", 1.2),
           ("speech", 4.0), ("pause", 5.0), ("speech", 3.0), ("pause", 0.4), ("speech", 2.0),
           ("pause", 1.0)]


@pytest.fixture(scope="module")
def clean():
    return utterance(PATTERN, jitter=0.005, shimmer=0.03)


class FakeTranscriber:
    """2.5 words per second of audio from a fixed passage."""
    WORDS = ("the quick brown fox jumps over the lazy dog and then it runs into "
             "the old forest where many birds sing").split()

    def __init__(self):
        self.calls = 0

    def __call__(self, audio, sr=SR):
        self.calls += 1
        n = int(len(audio) / sr * 2.5)
        return " ".join(self.WORDS[i % len(self.WORDS)] for i in range(n))


# ------------------------------- measures ------------------------------------ #

def test_speech_time_pauses_and_turn_gaps(clean):
    r = analyze_audio(clean)
    assert r["speaking_time_s"] == pytest.approx(14.0, abs=0.3)
    assert r["speech_segments"] == 5
    p = r["pauses"]
    assert p["count"] == 3 and p["turn_gaps"] == 1          # the 5 s silence is a turn gap
    assert p["mean_s"] == pytest.approx((0.6 + 1.2 + 0.4) / 3, abs=0.06)
    assert p["max_s"] == pytest.approx(1.2, abs=0.06)


def test_articulation_rate(clean):
    r = analyze_audio(clean)
    assert r["speech_rate"]["articulation_rate_syll_per_s"] == pytest.approx(4.0, abs=0.4)
    slow = analyze_audio(utterance(PATTERN, syll_rate=2.5))
    assert slow["speech_rate"]["articulation_rate_syll_per_s"] == pytest.approx(2.5, abs=0.4)


def test_jitter_tracks_voice_perturbation():
    readings = [analyze_audio(utterance(PATTERN, jitter=j))["voice"]["jitter_local_pct"]
                for j in (0.003, 0.01, 0.02)]
    assert readings[0] < readings[1] < readings[2]
    # local jitter of random period noise ~ 1.13 x its relative sd
    assert readings[1] == pytest.approx(1.13, rel=0.25)


def test_shimmer_tracks_amplitude_perturbation():
    lo = analyze_audio(utterance(PATTERN, shimmer=0.02))["voice"]["shimmer_local_pct"]
    hi = analyze_audio(utterance(PATTERN, shimmer=0.12))["voice"]["shimmer_local_pct"]
    assert hi > lo + 2


def test_pitch(clean):
    assert analyze_audio(clean)["voice"]["f0_mean_hz"] == pytest.approx(120, abs=2)
    assert analyze_audio(utterance(PATTERN, f0=210))["voice"]["f0_mean_hz"] == pytest.approx(210, abs=4)


def test_streaming_in_chunks_matches_one_shot(clean):
    one = analyze_audio(clean)
    a = SpeechAnalyzer()
    for i in range(0, len(clean), 1234):                 # odd chunk size on purpose
        a.feed(clean[i:i + 1234])
    chunked = a.finish().summary()
    assert chunked["pauses"] == one["pauses"]
    assert chunked["speaking_time_s"] == one["speaking_time_s"]
    assert chunked["voice"]["jitter_local_pct"] == pytest.approx(one["voice"]["jitter_local_pct"], rel=0.02)


def test_other_sample_rate_is_resampled(clean):
    x48 = resample(clean, SR, 48000)
    r = analyze_audio(x48, sr=48000)
    assert r["pauses"]["count"] == 3 and r["speaking_time_s"] == pytest.approx(14.0, abs=0.3)


def test_silence_and_noise_only():
    r = analyze_audio(np.random.default_rng(0).normal(0, 0.001, SR * 10).astype(np.float32))
    assert r["speech_segments"] == 0 and r["voice"]["jitter_local_pct"] is None
    assert r["speech_rate"]["syllables_per_s"] is None


def test_louder_room_noise_adapts():
    loud = utterance(PATTERN, noise=0.01)                # ~ -40 dB room noise
    r = analyze_audio(loud)
    assert r["pauses"]["count"] == 3 and r["speech_segments"] == 5


def test_lexical_measures():
    lx = lexical_measures("The cat sat on the mat. Um, the cat was happy, uh, very happy.")
    assert lx["words"] == 14
    assert lx["unique_words"] == 10
    assert lx["type_token_ratio"] == pytest.approx(10 / 14, abs=0.001)
    # content words: cat sat mat cat happy happy -> 6 / 14
    assert lx["lexical_density"] == pytest.approx(6 / 14, abs=0.001)
    assert lx["fillers"] == 2
    assert lexical_measures("") is None
    rich = lexical_measures(" ".join(f"w{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(100)))
    poor = lexical_measures(" ".join(["cat", "dog", "cat", "sat"] * 25))
    assert rich["mattr"] == 1.0 and poor["mattr"] < 0.1


def test_words_per_minute_with_transcriber(clean):
    tx = FakeTranscriber()
    r = analyze_audio(clean, transcriber=tx, include_transcript=True)
    assert tx.calls >= 1 and r["transcribed"]
    talk = r["speaking_time_s"] + r["pauses"]["total_s"]
    assert r["speech_rate"]["words_per_min"] == pytest.approx(r["lexical"]["words"] / (talk / 60), rel=0.02)
    assert r["lexical"]["words"] > 20 and "transcript" in r


def test_pcm16_roundtrip():
    x = np.array([0, 0.5, -0.5, 0.999], np.float32)
    pcm = (x * 32767).astype("<i2").tobytes()
    assert np.allclose(pcm16_to_float(pcm), x, atol=1e-3)


# ------------------------------- session ------------------------------------- #

class MemSink:
    def __init__(self, fail=False):
        self.saved, self.fail = [], fail

    def save(self, wav, meta):
        if self.fail:
            raise RuntimeError("storage down")
        self.saved.append((wav, meta))
        return {"backend": "memory", "n": len(self.saved)}


def feed_in_chunks(session, x, chunk=4000):
    for i in range(0, len(x), chunk):
        session.feed(x[i:i + chunk])


def test_session_section_recording_and_report(clean, tmp_path):
    sink, store = MemSink(), LocalVisitStore(tmp_path / "visits.json")
    s = CallSession("p-1", sink=sink, store=store)
    feed_in_chunks(s, clean[: SR * 5])
    s.start_section("reading")
    feed_in_chunks(s, clean[SR * 5: SR * 15])
    saved = s.stop_section()
    feed_in_chunks(s, clean[SR * 15:])
    assert saved["kind"] == "section_saved" and saved["duration_s"] == pytest.approx(10.0, abs=0.01)
    wav, meta = sink.saved[0]
    assert wav[:4] == b"RIFF" and meta["label"] == "reading" and meta["patient_id"] == "p-1"
    assert meta["started_offset_s"] == pytest.approx(5.0, abs=0.01)
    assert meta["metrics"]["speaking_time_s"] > 5
    r = s.end()
    assert r["kind"] == "speech_report" and r["baseline"]["provisional"]
    assert r["baseline"]["voice_score"] == 75 and r["baseline"]["baseline_session_count"] == 0
    assert set(r["features"]) >= {"speech_rate", "pause_ratio", "f0_sd_hz", "jitter_pct",
                                  "shimmer_pct", "duration_seconds", "voiced_seconds"}
    assert "features" in sink.saved[0][1]["metrics"]           # sections carry them too
    assert r["summary"]["pauses"]["count"] == 3
    assert s.end() is r                                    # ending twice is harmless
    assert store.previous("p-1")["visit_id"] == s.visit_id


def test_storage_failure_does_not_break_the_call(clean):
    s = CallSession("p-1", sink=MemSink(fail=True))
    s.start_section("x")
    feed_in_chunks(s, clean[: SR * 4])
    m = s.stop_section()
    assert "storage down" in m["save_error"]
    feed_in_chunks(s, clean[SR * 4:])
    assert s.end()["summary"]["speech_segments"] >= 4


def test_long_section_auto_saves(clean):
    sink = MemSink()
    s = CallSession("p-1", sink=sink, max_section_s=6.0)
    s.start_section("long")
    msgs = []
    for i in range(0, len(clean), 4000):
        msgs += s.feed(clean[i:i + 4000])
    assert [m["reason"] for m in msgs] == ["max_length"] and len(sink.saved) == 1


# ------------------------------- Supabase (mocked) --------------------------- #

class FakeQuery:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self._op, self._row = db, table, [], None, None

    def insert(self, row):
        self._op, self._row = "insert", row
        return self

    def upsert(self, row, on_conflict=None):
        self._op, self._row = "upsert", row
        return self

    def select(self, *_):
        self._op = "select"
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, n):
        self.n = n
        return self

    def execute(self):
        rows = self.db.setdefault(self.table, [])
        if self._op in ("insert", "upsert"):
            row = {**self._row, "id": len(rows) + 1, "created_at": len(rows)}
            rows.append(row)
            return type("R", (), {"data": [row]})()
        out = [r for r in reversed(rows) if all(r.get(k) == v for k, v in self.filters)]
        return type("R", (), {"data": out[: getattr(self, "n", 100)]})()


class FakeSupabase:
    def __init__(self):
        self.db, self.uploads = {}, []
        sup = self

        class Bucket:
            def __init__(self, name):
                self.name = name

            def upload(self, path, data, opts):
                sup.uploads.append((self.name, path, data, opts))

        class Storage:
            def from_(self, name):
                return Bucket(name)

        self.storage = Storage()

    def table(self, name):
        return FakeQuery(self.db, name)


def test_supabase_recording_sink_and_visit_store():
    sb = FakeSupabase()
    sink = SupabaseRecordingSink(sb, bucket="speech")
    out = sink.save(b"RIFFdata", {"patient_id": "p 1/x", "visit_id": "v1", "label": "reading passage",
                                  "duration_s": 12.5, "metrics": {"a": 1}})
    bucket, path, data, opts = sb.uploads[0]
    assert bucket == "speech" and data == b"RIFFdata" and opts["content-type"] == "audio/wav"
    assert path.startswith("p_1_x/v1/") and path.endswith(".wav") and "reading_passage" in path
    row = sb.db["speech_recordings"][0]
    assert row["storage_path"] == f"speech/{path}" and row["metrics"] == {"a": 1}
    assert out["backend"] == "supabase" and out["row_id"] == 1

    store = SupabaseVisitStore(sb)
    store.save("p1", "v1", {"x": 1})
    store.save("p1", "v2", {"x": 2})
    assert store.previous("p1", before_visit_id="v2")["visit_id"] == "v1"
    assert store.previous("nobody") is None


# ------------------------------- server --------------------------------------- #

def _pcm(x):
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def test_server_full_call(clean, tmp_path):
    sink, store = MemSink(), LocalVisitStore(tmp_path / "v.json")
    tx = FakeTranscriber()

    def make_session(msg):
        return CallSession(str(msg.get("patient_id") or "unknown"), msg.get("visit_id"),
                           int(msg.get("sample_rate") or SR), transcriber=tx, sink=sink, store=store)

    async def go():
        async with websockets.serve(speech_server.make_handler(make_session), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            msgs = []
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "session_start", "patient_id": "p-5", "sample_rate": SR}))
                chunk = SR // 4
                for i in range(0, len(clean), chunk):
                    if i == SR * 5:
                        await ws.send(json.dumps({"type": "record_start", "label": "story"}))
                    if i == SR * 12:
                        await ws.send(json.dumps({"type": "record_stop"}))
                    await ws.send(_pcm(clean[i:i + chunk]))
                await ws.send(json.dumps({"type": "record_stop"}))      # nothing recording -> error
                await ws.send(json.dumps({"type": "session_end"}))
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 20))
                    msgs.append(m)
                    if m.get("kind") == "speech_report":
                        break
            return msgs

    msgs = asyncio.run(go())
    kinds = [(m["type"], m.get("kind") or m.get("command")) for m in msgs]
    assert ("ack", "session_start") in kinds and ("status", "speech_live") in kinds
    saved = [m for m in msgs if m.get("kind") == "section_saved"]
    assert len(saved) == 1 and saved[0]["label"] == "story" and saved[0]["duration_s"] == pytest.approx(7, abs=0.3)
    assert any(m["type"] == "error" and m["command"] == "record_stop" for m in msgs)
    rep = msgs[-1]
    assert rep["patient_id"] == "p-5" and rep["summary"]["pauses"]["count"] == 3
    assert rep["summary"]["lexical"]["words"] > 0 and rep["summary"]["speech_rate"]["words_per_min"]
    assert store.previous("p-5")["visit_id"] == rep["visit_id"]


def test_server_stores_call_if_connection_drops(clean, tmp_path):
    store = LocalVisitStore(tmp_path / "v.json")

    def make_session(msg):
        return CallSession(str(msg.get("patient_id") or "unknown"), sample_rate=SR, store=store)

    async def go():
        async with websockets.serve(speech_server.make_handler(make_session), "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "session_start", "patient_id": "p-drop"}))
                await ws.send(_pcm(clean[: SR * 10]))
            await asyncio.sleep(1.0)

    asyncio.run(go())
    prev = store.previous("p-drop")
    assert prev is not None and prev["summary"]["speaking_time_s"] > 3


def test_local_recording_sink(tmp_path):
    out = LocalRecordingSink(tmp_path).save(b"RIFFxx", {"patient_id": "p1", "visit_id": "v1",
                                                         "label": "story", "duration_s": 3})
    from pathlib import Path
    wav = Path(out["path"])
    assert wav.read_bytes() == b"RIFFxx" and json.loads(wav.with_suffix(".json").read_text())["label"] == "story"


# ------------------------------- .env ----------------------------------------- #

def test_env_file_loading(tmp_path, monkeypatch):
    from env import load_env, parse_env
    f = tmp_path / ".env"
    f.write_text(
        "# Supabase\n"
        "SUPABASE_URL=https://abc.supabase.co\n"
        'SUPABASE_SERVICE_KEY="eyJ.key=with=equals"\n'
        "export OTHER_SETTING=cid   # trailing comment\n"
        "EMPTY=\n"
        "not a valid line\n")
    assert parse_env(f.read_text())["SUPABASE_SERVICE_KEY"] == "eyJ.key=with=equals"
    for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OTHER_SETTING", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://real-env-wins.supabase.co")
    assert load_env(f) == [str(f.resolve())]
    import os
    assert os.environ["SUPABASE_URL"] == "https://real-env-wins.supabase.co"   # not overridden
    assert os.environ["SUPABASE_SERVICE_KEY"] == "eyJ.key=with=equals"
    assert os.environ["OTHER_SETTING"] == "cid" and os.environ["EMPTY"] == ""
    assert load_env(tmp_path / "missing.env") == []


def test_make_storage_uses_env(tmp_path, monkeypatch):
    import speech_storage
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert speech_storage.make_storage()[2].startswith("local")
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
    import supabase
    monkeypatch.setattr(supabase, "create_client", lambda url, key: FakeSupabase())
    sink, store, where = speech_storage.make_storage()
    assert isinstance(sink, SupabaseRecordingSink) and where.startswith("supabase")


# --------------------------- voice.py baseline -------------------------------- #

import voice
from speech_analysis import voice_features
from speech_session import prior_features, score_against_baseline


def _visit(store, patient, **voice_kw):
    s = CallSession(patient, store=store)
    feed_in_chunks(s, utterance(PATTERN, **voice_kw))
    return s.end()


def test_voice_features_mapping(clean):
    f = voice_features(analyze_audio(clean))
    assert f["speech_rate"] == pytest.approx(analyze_audio(clean)["speech_rate"]["syllables_per_s"])
    assert f["pause_ratio"] == pytest.approx(0.136, abs=0.02)           # 2.2 s pauses / 16.2 s talk
    assert f["mean_pause_ms"] == pytest.approx(733, abs=60)
    assert f["jitter_pct"] > 0 and f["shimmer_pct"] > 0 and f["f0_sd_hz"] is not None
    assert f["duration_seconds"] == pytest.approx(len(clean) / SR, abs=0.1)
    assert 10 < f["voiced_seconds"] <= 14.5                             # voiced part of 14 s speech
    assert voice._usable(f)


def test_provisional_until_five_usable_sessions(tmp_path):
    store = LocalVisitStore(tmp_path / "v.json")
    scores = []
    for i in range(6):
        r = _visit(store, "p-b", seed=i)
        scores.append((r["baseline"]["baseline_session_count"], r["baseline"]["provisional"],
                       r["baseline"]["voice_score"]))
    assert scores[:5] == [(n, True, 75) for n in range(5)]
    n, provisional, score = scores[5]
    assert n == 5 and not provisional and 0 <= score <= 100


def test_deviation_from_baseline_lowers_score(tmp_path):
    store = LocalVisitStore(tmp_path / "v.json")
    for i in range(6):                                                   # stable baseline
        _visit(store, "p-d", seed=i, jitter=0.005 + 0.0005 * (i % 3), shimmer=0.03)
    typical = _visit(store, "p-d", seed=20, jitter=0.0055, shimmer=0.03)["baseline"]
    worse = _visit(store, "p-d", seed=21, jitter=0.02, shimmer=0.12)["baseline"]
    assert worse["voice_score"] < typical["voice_score"]
    assert worse["z"]["jitter_z"] > 3
    jit = next(d for d in worse["deviations"] if d["feature"] == "jitter_pct")
    assert jit["direction"] == "above baseline" and jit["lowers_score"]
    assert "not a diagnostic" in worse["disclaimer"]


def test_unusable_sessions_excluded_from_baseline(tmp_path):
    store = LocalVisitStore(tmp_path / "v.json")
    short = [("speech", 1.0), ("pause", 0.5), ("speech", 1.0)]          # 2.5 s: below the 8 s floor
    for i in range(5):
        s = CallSession("p-u", store=store)
        feed_in_chunks(s, utterance(short, seed=i))
        r = s.end()
        assert not r["baseline"]["current_session_usable"]
        assert any("usable floor" in n for n in r["notes"])
    assert _visit(store, "p-u")["baseline"]["baseline_session_count"] == 0


def test_current_session_never_in_its_own_baseline(tmp_path):
    store = LocalVisitStore(tmp_path / "v.json")
    s = CallSession("p-c", visit_id="same-visit", store=store)
    feed_in_chunks(s, utterance(PATTERN))
    store.save("p-c", "same-visit", {"features": {"duration_seconds": 99, "voiced_seconds": 99}})
    assert s.end()["baseline"]["baseline_session_count"] == 0


def test_score_matches_voice_engine_directly():
    priors = [{"duration_seconds": 30, "voiced_seconds": 10, "speech_rate": 3.0 + 0.1 * i,
               "pause_ratio": 0.2, "f0_sd_hz": 20 + i, "jitter_pct": 1.0 + 0.05 * i,
               "shimmer_pct": 8 + 0.2 * i} for i in range(6)]
    cur = {"duration_seconds": 30, "voiced_seconds": 10, "speech_rate": 2.5, "pause_ratio": 0.3,
           "f0_sd_hz": 22, "jitter_pct": 1.4, "shimmer_pct": 9}
    ours = score_against_baseline(cur, priors)
    ref = voice.compare_with_baseline(cur, priors)
    assert ours["voice_score"] == ref["voice_score"]
    assert ours["z"]["speech_rate_z"] == pytest.approx(ref["speech_rate_z"], abs=1e-3)


def test_prior_features_from_old_rows():
    old_row = {"summary": analyze_audio(utterance(PATTERN))}             # saved before features existed
    f = prior_features(old_row)
    assert f["jitter_pct"] is not None and voice._usable(f)


def test_long_unbroken_speech_is_split_without_fake_pauses():
    long_talk = utterance([("pause", 0.5), ("speech", 25.0), ("pause", 0.5)])
    r = analyze_audio(long_talk)
    assert r["speech_segments"] >= 3                     # split into <=10 s pieces
    assert r["pauses"]["count"] == 0                     # the splits aren't pauses
    assert r["speaking_time_s"] == pytest.approx(25.0, abs=0.3)
