"""One call's speech analysis, from start to end.

    session = CallSession(patient_id="p-123", sink=..., store=...)
    session.feed(audio_chunk)               # all call long
    session.start_section("reading passage")
    session.feed(...)
    session.stop_section()                  # saved via the sink (Supabase now)
    report = session.end()                  # full-call results vs the person's baseline

At the end of the call the voice features (voice.py's metrics) are scored
against this person's prior usable sessions with voice.compare_with_baseline:
personal z-scores plus a 0-100 voice score, provisional (75) until there are
5 usable prior sessions. Observational deviation-from-baseline signal only,
not a diagnostic output.

The whole-call audio isn't kept, only its running measurements. A recorded
section is kept in memory until it's stopped and saved.
"""

from __future__ import annotations

import time
import uuid

import numpy as np

import voice
from speech_analysis import SpeechAnalyzer, SpeechConfig, float_to_wav_bytes, resample, voice_features

# voice.py z-score key -> (feature key, label, direction that lowers the score)
Z_FIELDS = {
    "speech_rate_z": ("speech_rate", "Speech rate", "below"),
    "pause_ratio_z": ("pause_ratio", "Pause ratio", "above"),
    "f0_sd_z": ("f0_sd_hz", "Pitch variability (F0 SD)", "either"),
    "jitter_z": ("jitter_pct", "Jitter", "above"),
    "shimmer_z": ("shimmer_pct", "Shimmer", "above"),
}


class CallSession:
    def __init__(self, patient_id: str, visit_id: str | None = None, sample_rate: int = 16000,
                 cfg: SpeechConfig | None = None, transcriber=None, sink=None, store=None,
                 max_section_s: float = 600.0, baseline_limit: int = 200):
        self.cfg = cfg or SpeechConfig()
        self.patient_id = patient_id
        self.visit_id = visit_id or f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
        self.in_sr = sample_rate
        self.transcriber, self.sink, self.store = transcriber, sink, store
        self.max_section_s = max_section_s
        self.baseline_limit = baseline_limit
        self.call = SpeechAnalyzer(self.cfg, transcriber)
        self.started_at = time.time()
        self.section: dict | None = None
        self.sections: list[dict] = []
        self.ended = False
        self.report: dict | None = None

    # ---------------- audio ----------------

    def feed(self, samples: np.ndarray) -> list[dict]:
        """Add audio (float in [-1, 1] at the session's sample rate). Returns
        any messages produced (e.g. a section auto-saved for being too long)."""
        if self.ended:
            return []
        x = resample(samples, self.in_sr, self.cfg.sample_rate)
        self.call.feed(x)
        out = []
        if self.section is not None:
            self.section["chunks"].append(x)
            self.section["analyzer"].feed(x)
            if self.section["analyzer"].total_s >= self.max_section_s:
                out.append(self.stop_section(reason="max_length"))
        return out

    @property
    def elapsed_s(self) -> float:
        return self.call.total_s

    # ---------------- sections ----------------

    def start_section(self, label: str = "section") -> dict:
        msgs = []
        if self.section is not None:
            msgs.append(self.stop_section(reason="new_section_started"))
        self.section = {"label": label or "section", "started_at": time.time(),
                        "started_offset_s": round(self.elapsed_s, 2), "chunks": [],
                        "analyzer": SpeechAnalyzer(self.cfg, self.transcriber)}
        return {"type": "status", "kind": "section_recording", "label": self.section["label"],
                "started_offset_s": self.section["started_offset_s"], "stopped_previous": msgs}

    def stop_section(self, reason: str = "stopped") -> dict:
        sec, self.section = self.section, None
        if sec is None:
            raise ValueError("no section is being recorded")
        audio = np.concatenate(sec["chunks"]) if sec["chunks"] else np.zeros(0, np.float32)
        metrics = sec["analyzer"].finish().summary(include_transcript=False)
        metrics["features"] = voice_features(metrics)
        wav = float_to_wav_bytes(audio, self.cfg.sample_rate)
        meta = {"patient_id": self.patient_id, "visit_id": self.visit_id, "label": sec["label"],
                "started_at": sec["started_at"], "started_offset_s": sec["started_offset_s"],
                "duration_s": round(len(audio) / self.cfg.sample_rate, 2), "metrics": metrics}
        saved, error = None, None
        if self.sink is not None:
            try:
                saved = self.sink.save(wav, meta)
            except Exception as e:                    # keep the call going if storage fails
                error = f"{e.__class__.__name__}: {e}"
        record = {"label": sec["label"], "started_offset_s": sec["started_offset_s"],
                  "duration_s": meta["duration_s"], "metrics": metrics, "saved": saved,
                  "reason": reason}
        if error:
            record["save_error"] = error
        self.sections.append(record)
        return {"type": "result", "kind": "section_saved", **record}

    # ---------------- reporting ----------------

    def live(self) -> dict:
        s = self.call.summary()
        return {"type": "status", "kind": "speech_live", "elapsed_s": round(self.elapsed_s, 1),
                "recording": self.section["label"] if self.section else None,
                "speaking_time_s": s["speaking_time_s"], "speech_rate": s["speech_rate"],
                "voice": s["voice"], "pauses": s["pauses"],
                "words": (s["lexical"] or {}).get("words")}

    def end(self) -> dict:
        """Finish the call: close any section, compute final results, score them
        against this person's prior sessions (voice.py), store this visit."""
        if self.ended:
            return self.report
        if self.section is not None:
            self.stop_section(reason="call_ended")
        summary = self.call.finish().summary(include_transcript=False)
        features = voice_features(summary)
        summary["features"] = features
        priors, store_info, store_error = [], None, None
        if self.store is not None:
            try:
                # audit-#16: only PRIOR sessions, never the current one
                rows = self.store.history(self.patient_id, before_visit_id=self.visit_id,
                                          limit=self.baseline_limit)
                priors = [prior_features(r) for r in rows]
            except Exception as e:
                store_error = f"loading prior sessions failed: {e}"
        baseline = score_against_baseline(features, priors)
        summary["voice_baseline"] = {k: baseline[k] for k in
                                     ("voice_score", "provisional", "baseline_session_count", "z")}
        if self.store is not None:
            try:
                store_info = self.store.save(self.patient_id, self.visit_id, summary)
            except Exception as e:
                store_error = f"saving this visit failed: {e}"
        self.ended = True
        self.report = {
            "type": "result", "kind": "speech_report",
            "patient_id": self.patient_id, "visit_id": self.visit_id,
            "call_duration_s": round(self.elapsed_s, 1),
            "summary": summary,
            "features": features,
            "baseline": baseline,
            "sections": self.sections,
            "stored": store_info,
            "notes": _notes(summary, baseline),
        }
        if store_error:
            self.report["store_error"] = store_error
        return self.report


def prior_features(row: dict) -> dict:
    """Voice features of a stored visit (computed from its summary if it was
    saved before features were stored)."""
    s = row.get("summary") or {}
    return s.get("features") or voice_features(s)


def score_against_baseline(features: dict, priors: list[dict]) -> dict:
    """voice.compare_with_baseline plus readable, observational detail."""
    res = voice.compare_with_baseline(features, priors)
    n = res["baseline_session_count"]
    provisional = n < voice.MIN_BASELINE_SESSIONS
    deviations = []
    for zkey, (fkey, label, adverse) in Z_FIELDS.items():
        z = res.get(zkey)
        if z is None:
            continue
        pop = [p[fkey] for p in priors if voice._usable(p) and p.get(fkey) is not None]
        mean = sum(pop) / len(pop) if pop else None
        lowers = (adverse == "below" and z < 0) or (adverse == "above" and z > 0) or adverse == "either"
        deviations.append({
            "feature": fkey, "label": label, "value": features.get(fkey),
            "baseline_mean": round(mean, 4) if mean is not None else None,
            "z": round(z, 2),
            "direction": "above baseline" if z > 0 else "below baseline" if z < 0 else "at baseline",
            "lowers_score": bool(lowers and abs(z) > 0),
        })
    if provisional:
        text = (f"Voice score {res['voice_score']} (provisional): personal baseline still forming, "
                f"{n} of {voice.MIN_BASELINE_SESSIONS} usable prior sessions.")
    else:
        text = (f"Voice score {res['voice_score']}/100: deviation from this person's own "
                f"baseline of {n} prior sessions.")
    return {
        **res,
        "z": {k: (round(v, 3) if v is not None else None) for k, v in res.items() if k.endswith("_z")},
        "provisional": provisional,
        "sessions_needed": max(0, voice.MIN_BASELINE_SESSIONS - n),
        "current_session_usable": voice._usable(features),
        "deviations": deviations,
        "summary_text": text,
        "disclaimer": "Observational deviation-from-personal-baseline signal, not a diagnostic output.",
    }


def _notes(s: dict, baseline: dict) -> list[str]:
    notes = []
    if s["speaking_time_s"] < 20:
        notes.append("Less than 20 s of speech: measures are unreliable.")
    if not baseline["current_session_usable"]:
        notes.append(f"This session is below the usable floor ({voice.USABLE_MIN_DURATION_S:.0f} s "
                     f"audio, {voice.USABLE_MIN_VOICED_S:.0f} s voiced), so it won't count toward "
                     "future baselines.")
    if not s["transcribed"]:
        notes.append("No transcription: words/min and lexical measures are missing "
                     "(install faster-whisper). The voice score doesn't need them.")
    return notes
