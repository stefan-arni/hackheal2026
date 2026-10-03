"""Voice biomarker baseline engine — pure functions.

Mirrors api.clinical.typing: personal z-scores vs the athlete's prior
usable sessions, lower-is-better inversion for adverse features, a 0-100
composite, and a provisional score while the baseline is forming.

SaMD: observational deviation-from-personal-baseline signal, not a
diagnostic output. UI copy must stay observational.
"""
from __future__ import annotations

import math
from typing import Optional

MIN_BASELINE_SESSIONS = 5
USABLE_MIN_DURATION_S = 8.0      # combined task floor (vowel ~4s + passage)
USABLE_MIN_VOICED_S = 3.0        # require real voiced content
PROVISIONAL_SCORE = 75

# Features where a HIGHER value is WORSE (rise = adverse). For these we keep
# the raw z (positive z is already adverse). For higher-is-better features
# (speech_rate, articulation_rate) we keep the raw z too, but the composite
# penalises the NEGATIVE direction. f0_sd is two-sided (magnitude only).
LOWER_IS_BETTER = {"jitter_pct", "shimmer_pct", "pause_ratio", "mean_pause_ms",
                   "voice_onset_ms"}


def z_score(value: Optional[float], population: list[float]) -> Optional[float]:
    """Sample-sd z-score. None when value missing, population too small,
    or sd effectively zero. Uses n-1 denominator (parity with typing)."""
    if value is None or len(population) < MIN_BASELINE_SESSIONS:
        return None
    n = len(population)
    mean = sum(population) / n
    variance = sum((v - mean) ** 2 for v in population) / (n - 1)
    sd = math.sqrt(variance)
    if sd <= 0.0001:
        return None
    return (value - mean) / sd


def adverse_z(value: Optional[float], population: list[float], *, key: str
              ) -> Optional[float]:
    """Raw personal z. (Inversion of lower-is-better features is handled by
    the composite's sign treatment, kept here as identity so callers can
    store the raw z and the engine remains the single source of truth.)"""
    return z_score(value, population)


def composite_score(
    speech_rate_z: Optional[float],
    pause_ratio_z: Optional[float],
    f0_sd_z: Optional[float],
    jitter_z: Optional[float],
    shimmer_z: Optional[float],
) -> int:
    """Penalty model. Slower speech (speech_rate_z < 0), higher pause ratio,
    higher jitter, higher shimmer (each z > 0), and larger pitch instability
    (|f0_sd_z|) all penalise. Per-component caps. Missing dims contribute 0.
    Returns int in [0, 100]. Half-away-from-zero rounding (Swift/typing parity)."""
    penalty = 0.0
    if speech_rate_z is not None and speech_rate_z < 0:
        penalty += min(abs(speech_rate_z) * 9.0, 25.0)
    if pause_ratio_z is not None and pause_ratio_z > 0:
        penalty += min(pause_ratio_z * 8.0, 20.0)
    if f0_sd_z is not None:
        penalty += min(abs(f0_sd_z) * 5.0, 15.0)
    if jitter_z is not None and jitter_z > 0:
        penalty += min(jitter_z * 7.0, 20.0)
    if shimmer_z is not None and shimmer_z > 0:
        penalty += min(shimmer_z * 7.0, 20.0)
    raw = 100.0 - penalty
    return max(0, min(100, math.floor(raw + 0.5)))


def _usable(s: dict) -> bool:
    return (
        (s.get("duration_seconds") or 0) >= USABLE_MIN_DURATION_S
        and (s.get("voiced_seconds") or 0) >= USABLE_MIN_VOICED_S
    )


def _population(sessions: list[dict], key: str) -> list[float]:
    return [s[key] for s in sessions if s.get(key) is not None]


def compare_with_baseline(current: dict, baseline: list[dict]) -> dict:
    """Compare current session against prior usable sessions.

    ``baseline`` must contain only PRIOR sessions (audit-#16: exclude the
    current row). When fewer than MIN_BASELINE_SESSIONS usable priors exist,
    voice_score is PROVISIONAL_SCORE (75) and all z's are None.
    """
    usable = [s for s in baseline if _usable(s)]
    if len(usable) < MIN_BASELINE_SESSIONS:
        return {
            "speech_rate_z": None, "pause_ratio_z": None, "f0_sd_z": None,
            "jitter_z": None, "shimmer_z": None,
            "voice_score": PROVISIONAL_SCORE,
            "baseline_session_count": len(usable),
        }

    speech_rate_z = adverse_z(current.get("speech_rate"),
                              _population(usable, "speech_rate"), key="speech_rate")
    pause_ratio_z = adverse_z(current.get("pause_ratio"),
                              _population(usable, "pause_ratio"), key="pause_ratio")
    f0_sd_z = adverse_z(current.get("f0_sd_hz"),
                        _population(usable, "f0_sd_hz"), key="f0_sd_hz")
    jitter_z = adverse_z(current.get("jitter_pct"),
                         _population(usable, "jitter_pct"), key="jitter_pct")
    shimmer_z = adverse_z(current.get("shimmer_pct"),
                          _population(usable, "shimmer_pct"), key="shimmer_pct")

    return {
        "speech_rate_z": speech_rate_z, "pause_ratio_z": pause_ratio_z,
        "f0_sd_z": f0_sd_z, "jitter_z": jitter_z, "shimmer_z": shimmer_z,
        "voice_score": composite_score(speech_rate_z, pause_ratio_z, f0_sd_z,
                                       jitter_z, shimmer_z),
        "baseline_session_count": len(usable),
    }
