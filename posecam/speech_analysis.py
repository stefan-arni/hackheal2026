"""Speech analysis: speech rate, jitter, shimmer, pauses, lexical richness/density.

Works on a continuous audio stream (16 kHz mono float in [-1, 1]), so it can run
for an entire call: feed audio as it arrives, read a running summary at any
time, and get the full report at the end. The same analyzer also runs on a
recorded section.

Pipeline
  1. Voice activity (energy-based, adapts to the room's noise floor) splits the
     stream into speech segments and the silences between them.
  2. Pauses: silences between 0.25 s and 3 s inside someone's speech. Longer
     silences are turn gaps (listening to the clinician, thinking about a
     question) and are counted separately, so they don't distort pause stats or
     speech rate.
  3. Each speech segment goes to Praat (via parselmouth) for jitter, shimmer,
     pitch and harmonics-to-noise ratio, and an acoustic syllable count
     (intensity peaks in voiced speech, after de Jong & Wempe 2009), which gives
     speech rate without a transcript.
  4. Optional transcription (faster-whisper by default) gives words per minute
     and lexical richness (type-token ratio, moving-average TTR) and density
     (share of content words).

Clinical notes: jitter and shimmer are normally measured on sustained vowels;
on conversational speech they're higher and noisier, so compare a person
against their own past visits (as the report does), not against published
norms. The audio should be the patient's voice only (e.g. their own mic or a
per-participant Zoom stream), not the mixed call.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

SR = 16000


@dataclass
class SpeechConfig:
    sample_rate: int = SR
    frame_s: float = 0.02              # VAD frame
    speech_above_floor_db: float = 12  # speech = this much louder than the noise floor
    noise_window_s: float = 30.0       # noise floor = quiet percentile over this window
    min_pause_s: float = 0.25          # shorter silences are part of speech
    max_pause_s: float = 3.0           # longer silences are turn gaps, not pauses
    min_segment_s: float = 0.12        # shorter "speech" blips are noise
    max_segment_s: float = 10.0        # long unbroken speech is analyzed in pieces this long
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 500.0
    transcribe_every_s: float = 20.0   # send speech to the transcriber in chunks this long
    mattr_window: int = 50


# --------------------------------------------------------------------------- #
# Voice quality and syllables on one speech segment (Praat via parselmouth)
# --------------------------------------------------------------------------- #

def voice_measures(samples: np.ndarray, sr: int, cfg: SpeechConfig) -> dict | None:
    """Jitter, shimmer, pitch, HNR and syllable count for one speech segment."""
    import parselmouth
    from parselmouth.praat import call

    if len(samples) < 0.1 * sr:
        return None
    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=cfg.pitch_floor_hz,
                         pitch_ceiling=cfg.pitch_ceiling_hz)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    out = {"duration_s": len(samples) / sr, "voiced_frames": int(len(voiced)),
           "syllables": count_syllables(snd, pitch)}
    if len(voiced) < 3:
        return out
    pp = call(snd, "To PointProcess (periodic, cc)", cfg.pitch_floor_hz, cfg.pitch_ceiling_hz)
    n_periods = call(pp, "Get number of periods", 0, 0, 1 / cfg.pitch_ceiling_hz,
                     1 / cfg.pitch_floor_hz, 1.3)
    if n_periods < 3:
        return out
    jitter = call(pp, "Get jitter (local)", 0, 0, 1 / cfg.pitch_ceiling_hz, 1 / cfg.pitch_floor_hz, 1.3)
    shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 1 / cfg.pitch_ceiling_hz,
                   1 / cfg.pitch_floor_hz, 1.3, 1.6)
    harm = call(snd, "To Harmonicity (cc)", 0.01, cfg.pitch_floor_hz, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    out.update({
        "periods": int(n_periods),
        "jitter": None if math.isnan(jitter) else float(jitter),
        "shimmer": None if math.isnan(shimmer) else float(shimmer),
        "f0_mean_hz": float(np.mean(voiced)),
        "f0_sd_hz": float(np.std(voiced)),
        "hnr_db": None if math.isnan(hnr) else float(hnr),
    })
    return out


def count_syllables(snd, pitch, dip_db: float = 2.0, below_peak_db: float = 25.0) -> int:
    """Syllable nuclei: intensity peaks that are voiced, loud enough, and separated
    by a dip of at least `dip_db` (de Jong & Wempe 2009, simplified)."""
    inten = snd.to_intensity(minimum_pitch=50, time_step=0.01)
    vals = inten.values[0]
    times = inten.xs()
    if len(vals) < 3:
        return 0
    floor = max(np.median(vals), np.max(vals) - below_peak_db)
    peaks = [i for i in range(1, len(vals) - 1)
             if vals[i] >= vals[i - 1] and vals[i] > vals[i + 1] and vals[i] > floor]
    count, last_peak = 0, None
    for i in peaks:
        f0 = pitch.get_value_at_time(times[i])
        if not f0 or math.isnan(f0):
            continue                               # unvoiced peak (e.g. "s")
        if last_peak is not None:
            dip = vals[last_peak:i + 1].min()
            if min(vals[last_peak], vals[i]) - dip < dip_db:
                if vals[i] > vals[last_peak]:
                    last_peak = i                  # same syllable, keep the higher peak
                continue
        count += 1
        last_peak = i
    return count


# --------------------------------------------------------------------------- #
# Lexical richness / density from a transcript
# --------------------------------------------------------------------------- #

FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their mine yours hers ours theirs
i me you he him she it we us they them myself yourself himself herself itself ourselves
themselves who whom whose which what whatever whoever someone somebody something anyone
anybody anything everyone everybody everything noone nobody nothing one ones
and or but nor so yet for because although though while whereas if unless until since
as than whether either neither both
in on at by to of from with without into onto upon about above below over under
between among through during before after around against along across behind beyond
near off out up down toward towards via per within
is am are was were be been being have has had having do does did doing done
will would shall should can could may might must ought
not no yes n't 's 're 've 'd 'll 'm
there here then now just very too also only even still really quite rather
all any each every few many more most much other some such own same
how when where why
um uh er ah oh hmm mm like yeah okay ok well
""".split())

FILLERS = {"um", "uh", "er", "ah", "hmm", "mm"}
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def lexical_measures(text: str, mattr_window: int = 50) -> dict | None:
    words = tokenize(text)
    if not words:
        return None
    content = [w for w in words if w not in FUNCTION_WORDS]
    n = len(words)
    win = min(mattr_window, n)
    if n > win:
        ttrs = [len(set(words[i:i + win])) / win for i in range(n - win + 1)]
        mattr = float(np.mean(ttrs))
    else:
        mattr = len(set(words)) / n
    return {
        "words": n,
        "unique_words": len(set(words)),
        "type_token_ratio": round(len(set(words)) / n, 3),
        "mattr": round(mattr, 3),
        "lexical_density": round(len(content) / n, 3),
        "fillers": sum(w in FILLERS for w in words),
        "fillers_per_100_words": round(100 * sum(w in FILLERS for w in words) / n, 1),
    }


# --------------------------------------------------------------------------- #
# Transcription (pluggable)
# --------------------------------------------------------------------------- #

class FasterWhisperTranscriber:
    """Local speech-to-text with faster-whisper. The model downloads on first use."""

    def __init__(self, model: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def __call__(self, samples: np.ndarray, sr: int = SR) -> str:
        segments, _ = self.model.transcribe(samples.astype(np.float32), language="en",
                                            vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments)


def make_transcriber(kind: str = "whisper", model: str = "base.en"):
    """'whisper' (local faster-whisper) or 'none'. Returns None if unavailable."""
    if kind == "none":
        return None
    try:
        return FasterWhisperTranscriber(model)
    except Exception as e:  # not installed / model can't download
        print(f"[speech] transcription off ({e.__class__.__name__}: {e}); "
              "words/min and lexical measures will be missing")
        return None


# --------------------------------------------------------------------------- #
# Streaming analyzer
# --------------------------------------------------------------------------- #

@dataclass
class SpeechAnalyzer:
    """Feed audio continuously; read summary() any time; finish() at the end."""
    cfg: SpeechConfig = field(default_factory=SpeechConfig)
    transcriber: object = None

    def __post_init__(self):
        c = self.cfg
        self.sr = c.sample_rate
        self.frame_n = int(round(c.frame_s * self.sr))
        self._buf = np.zeros(0, np.float32)          # samples not yet framed
        self._frames_seen = 0
        self._energies: deque = deque(maxlen=int(c.noise_window_s / c.frame_s))
        self._floor_db: float | None = None
        self._seg_frames: list[np.ndarray] = []      # current speech segment
        self._seg_start: float | None = None
        self._silence_frames: list[np.ndarray] = []  # silence that may still be inside the segment
        self._last_seg_end: float | None = None
        self.segments: list[dict] = []
        self.pauses: list[float] = []                # within-turn pauses (s)
        self.turn_gaps: list[float] = []             # longer silences (s)
        self._tx_audio: list[np.ndarray] = []
        self._tx_len = 0
        self.transcript: list[str] = []
        self.total_s = 0.0

    # ---- input ----

    def feed(self, samples: np.ndarray):
        x = np.asarray(samples, np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        self.total_s += len(x) / self.sr
        self._buf = np.concatenate([self._buf, x])
        n = len(self._buf) // self.frame_n
        for i in range(n):
            self._frame(self._buf[i * self.frame_n:(i + 1) * self.frame_n])
        self._buf = self._buf[n * self.frame_n:]

    def _frame(self, fr: np.ndarray):
        c = self.cfg
        t = self._frames_seen * c.frame_s
        self._frames_seen += 1
        db = 10 * math.log10(float(np.mean(fr.astype(np.float64) ** 2)) + 1e-12)
        self._energies.append(db)
        if self._floor_db is None or self._frames_seen % 25 == 0:
            self._floor_db = float(np.percentile(self._energies, 10))
        speech = db > max(self._floor_db + c.speech_above_floor_db, -60.0)

        if speech:
            if self._seg_start is None:
                self._seg_start = t
            elif self._silence_frames:                 # short silence: still the same segment
                self._seg_frames.extend(self._silence_frames)
            self._silence_frames = []
            self._seg_frames.append(fr)
            if len(self._seg_frames) * c.frame_s >= c.max_segment_s:
                self._close_segment()                  # keeps each analysis step short
        elif self._seg_start is not None:
            self._silence_frames.append(fr)
            if len(self._silence_frames) * c.frame_s >= c.min_pause_s:
                self._close_segment()

    def _close_segment(self):
        c = self.cfg
        audio = np.concatenate(self._seg_frames) if self._seg_frames else np.zeros(0, np.float32)
        start, dur = self._seg_start, len(audio) / self.sr
        self._seg_frames, self._silence_frames, self._seg_start = [], [], None
        if dur < c.min_segment_s:
            return
        if self._last_seg_end is not None:
            gap = start - self._last_seg_end
            if gap >= c.min_pause_s - 1e-6:              # 0 when a long segment was split
                (self.pauses if gap <= c.max_pause_s else self.turn_gaps).append(gap)
        self._last_seg_end = start + dur
        vm = voice_measures(audio, self.sr, c) or {"duration_s": dur, "syllables": 0}
        vm["start_s"] = round(start, 2)
        self.segments.append(vm)
        if self.transcriber is not None:
            self._tx_audio.append(audio)
            self._tx_len += len(audio)
            if self._tx_len >= c.transcribe_every_s * self.sr:
                self._transcribe()

    def _transcribe(self):
        if not self._tx_audio or self.transcriber is None:
            return
        gap = np.zeros(int(0.3 * self.sr), np.float32)    # keep words from running together
        audio = np.concatenate([a for seg in self._tx_audio for a in (seg, gap)])
        self._tx_audio, self._tx_len = [], 0
        try:
            text = self.transcriber(audio, self.sr)
        except Exception as e:
            text = ""
            print(f"[speech] transcription failed: {e}")
        if text.strip():
            self.transcript.append(text.strip())

    def finish(self) -> "SpeechAnalyzer":
        """Close any open segment and transcribe what's left."""
        if self._seg_start is not None:
            self._close_segment()
        self._transcribe()
        return self

    # ---- output ----

    def summary(self, include_transcript: bool = False) -> dict:
        c = self.cfg
        segs = self.segments
        speaking = sum(s["duration_s"] for s in segs)
        within = sum(self.pauses)
        talk_time = speaking + within                  # speaking + hesitations, excl. turn gaps
        syll = sum(s.get("syllables", 0) for s in segs)

        def wmean(key, weight="periods"):
            vals = [(s[key], s.get(weight, 0)) for s in segs if s.get(key) is not None and s.get(weight)]
            tot = sum(w for _, w in vals)
            return sum(v * w for v, w in vals) / tot if tot else None

        jit, shim = wmean("jitter"), wmean("shimmer")
        text = " ".join(self.transcript)
        lex = lexical_measures(text, c.mattr_window) if text else None
        words = lex["words"] if lex else None
        p = np.array(self.pauses) if self.pauses else None
        voiced = sum(s.get("voiced_frames", 0) for s in segs) * 0.01   # pitch frames are 10 ms
        out = {
            "duration_s": round(self.total_s, 1),
            "speaking_time_s": round(speaking, 1),
            "voiced_time_s": round(voiced, 2),
            "speech_segments": len(segs),
            "speech_rate": {
                "words_per_min": round(words / (talk_time / 60), 1) if words and talk_time else None,
                "syllables_per_s": round(syll / talk_time, 2) if talk_time else None,
                "articulation_rate_syll_per_s": round(syll / speaking, 2) if speaking else None,
            },
            "voice": {
                "jitter_local_pct": round(100 * jit, 3) if jit is not None else None,
                "shimmer_local_pct": round(100 * shim, 2) if shim is not None else None,
                "f0_mean_hz": round(wmean("f0_mean_hz"), 1) if wmean("f0_mean_hz") else None,
                "f0_sd_hz": round(wmean("f0_sd_hz"), 1) if wmean("f0_sd_hz") else None,
                "hnr_db": round(wmean("hnr_db"), 1) if wmean("hnr_db") is not None else None,
            },
            "pauses": {
                "count": len(self.pauses),
                "per_min": round(len(self.pauses) / (talk_time / 60), 1) if talk_time else None,
                "mean_s": round(float(p.mean()), 2) if p is not None else None,
                "median_s": round(float(np.median(p)), 2) if p is not None else None,
                "max_s": round(float(p.max()), 2) if p is not None else None,
                "total_s": round(within, 1),
                "pause_time_pct": round(100 * within / talk_time, 1) if talk_time else None,
                "turn_gaps": len(self.turn_gaps),
            },
            "lexical": lex,
            "transcribed": self.transcriber is not None,
        }
        if include_transcript:
            out["transcript"] = text
        return out


# --------------------------------------------------------------------------- #
# Features for the voice baseline engine (voice.py)
# --------------------------------------------------------------------------- #

def voice_features(summary: dict) -> dict:
    """Map a summary() onto the flat feature names voice.py uses.

    Units are fixed so sessions stay comparable with each other:
      speech_rate        syllables per second over talk time (pauses included).
                         Acoustic, so it doesn't depend on the transcript.
      articulation_rate  syllables per second while actually speaking
      pause_ratio        0-1, time in pauses / talk time (turn gaps excluded)
      mean_pause_ms      mean within-turn pause, ms
      f0_sd_hz           pitch variability, Hz
      jitter_pct, shimmer_pct   Praat local jitter / shimmer, %
      duration_seconds   audio analyzed;  voiced_seconds   voiced (pitched) speech
      voice_onset_ms     not measured here (None)
    """
    sr, v, p = summary["speech_rate"], summary["voice"], summary["pauses"]
    pr = p.get("pause_time_pct")
    mp = p.get("mean_s")
    return {
        "duration_seconds": summary.get("duration_s"),
        "voiced_seconds": summary.get("voiced_time_s",
                                      summary.get("speaking_time_s")),   # older summaries
        "speech_rate": sr.get("syllables_per_s"),
        "articulation_rate": sr.get("articulation_rate_syll_per_s"),
        "pause_ratio": round(pr / 100, 4) if pr is not None else None,
        "mean_pause_ms": round(mp * 1000, 1) if mp is not None else None,
        "f0_sd_hz": v.get("f0_sd_hz"),
        "jitter_pct": v.get("jitter_local_pct"),
        "shimmer_pct": v.get("shimmer_local_pct"),
        "voice_onset_ms": None,
    }


def analyze_audio(samples: np.ndarray, sr: int = SR, transcriber=None,
                  cfg: SpeechConfig | None = None, include_transcript: bool = False) -> dict:
    """One-shot analysis of a recording (e.g. a section)."""
    cfg = cfg or SpeechConfig()
    if sr != cfg.sample_rate:
        samples = resample(samples, sr, cfg.sample_rate)
    a = SpeechAnalyzer(cfg, transcriber)
    a.feed(samples)
    return a.finish().summary(include_transcript)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, np.float32)
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(np.asarray(x, np.float32), sr_out // g, sr_in // g).astype(np.float32)


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    import io
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def config_from_dict(d: dict) -> SpeechConfig:
    return SpeechConfig(**{k: v for k, v in d.items() if k in asdict(SpeechConfig())})
