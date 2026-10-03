"""Voice biomarker endpoints — Phase 3 in-app acoustic task.

Privacy invariant: DERIVED FEATURES ONLY. extra="forbid" rejects any field
that could carry audio or text content. No audio or transcript ever crosses
the network (spec: 2026-06-12-voice-biomarker-design.md).

Tenant model mirrors typing/cognitive_tests: athletes write for themselves
(JWT identity); linked clinicians read via resolve_patient_scope.

Triple ingestion on every VALID session (quality gate passed):
  1. voice_sessions — full per-session features + baseline comparison;
  2. cognitive_test_results (test_type='speech') — voice as a cognitive signal;
  3. personal_metric_samples (source='voice') — daily aggregates for insights.

A session is VALID iff snr_db >= MIN_SNR_DB AND voiced_seconds >= MIN_VOICED_S.
Quality-gate-failed sessions record (valid=0) for QA but skip #2 and #3.
"""
import json
import logging
from datetime import datetime as _dt
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.auth.tenant_isolation import require_auth, resolve_patient_scope
from api.billing.hooks import safe_record
from api.clinical.baseline_window import today_for_user
from api.clinical.voice import (
    compare_with_baseline,
    USABLE_MIN_DURATION_S,
    USABLE_MIN_VOICED_S,
)
from api.database import get_connection
from api.db import get_backend, sql as dsql
from api.dependencies import get_db

logger = logging.getLogger("api.voice_sessions")
router = APIRouter(prefix="/api/voice", tags=["voice"])

MIN_SNR_DB = 10.0       # quality gate: below this the room is too noisy
MIN_VOICED_S = 3.0      # quality gate: need real voiced content

VALID_TASK_TYPES = {"read_passage", "sustained_vowel", "combined"}

# metric_key -> (voice_sessions column, label, unit)
PMS_KEYS = {
    "voice_speech_rate":  ("speech_rate",  "Speech rate",              "syl_s"),
    "voice_pause_ratio":  ("pause_ratio",  "Pause ratio",              "ratio"),
    "voice_f0_sd":        ("f0_sd_hz",     "Pitch variability (F0 SD)", "hz"),
    "voice_jitter":       ("jitter_pct",   "Jitter",                   "pct"),
    "voice_shimmer":      ("shimmer_pct",  "Shimmer",                  "pct"),
}


class VoiceSessionCreate(BaseModel):
    """POST body for /api/voice/sessions.

    extra="forbid" is the privacy invariant — any key not listed here (such
    as "transcript", "audio", "text") is rejected with HTTP 422.
    """
    model_config = ConfigDict(extra="forbid")

    started_at: str
    duration_seconds: float = Field(gt=0, le=3600)
    task_type: str = "combined"
    prompt_id: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"
    )
    client_session_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )

    # Acoustic features — all optional (device may not support all)
    speech_rate:      Optional[float] = Field(default=None, ge=0, le=20)
    articulation_rate: Optional[float] = Field(default=None, ge=0, le=20)
    pause_count:      Optional[int]   = Field(default=None, ge=0, le=10_000)
    pause_ratio:      Optional[float] = Field(default=None, ge=0, le=1)
    mean_pause_ms:    Optional[float] = Field(default=None, ge=0)
    voice_onset_ms:   Optional[float] = Field(default=None, ge=0)
    f0_mean_hz:       Optional[float] = Field(default=None, ge=0, le=1000)
    f0_sd_hz:         Optional[float] = Field(default=None, ge=0, le=1000)
    jitter_pct:       Optional[float] = Field(default=None, ge=0, le=100)
    shimmer_pct:      Optional[float] = Field(default=None, ge=0, le=100)
    snr_db:           Optional[float] = Field(default=None, ge=-50, le=120)
    voiced_seconds:   Optional[float] = Field(default=None, ge=0)

    @field_validator("started_at")
    @classmethod
    def _started_at_iso(cls, v: str) -> str:
        try:
            _dt.fromisoformat(v)
        except ValueError:
            raise ValueError("started_at must be ISO-8601")
        return v


def _require_athlete(user: dict) -> None:
    if user.get("role") != "athlete":
        raise HTTPException(status_code=403, detail="Athlete-only endpoint")


def _quality_ok(req: VoiceSessionCreate) -> bool:
    return (
        req.snr_db is not None and req.snr_db >= MIN_SNR_DB
        and req.voiced_seconds is not None and req.voiced_seconds >= MIN_VOICED_S
    )


def _load_prior_sessions(conn, user_id: str) -> list[dict]:
    """Prior VALID sessions only — self-contamination guard (audit #16)."""
    rows = conn.execute(
        """SELECT speech_rate, pause_ratio, f0_sd_hz, jitter_pct, shimmer_pct,
                  duration_seconds, voiced_seconds
           FROM voice_sessions
           WHERE user_id = ? AND valid = 1
           ORDER BY started_at DESC LIMIT 60""",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _upsert_daily_aggregates(conn, user_id: str, sample_date: str) -> None:
    """Daily mean of each PMS key across the day's VALID sessions."""
    upsert = dsql.upsert(
        get_backend(), "personal_metric_samples",
        ["user_id", "source", "metric_key", "metric_label", "sample_date",
         "sample_start", "sample_end", "value", "unit"],
        conflict_cols=["user_id", "source", "metric_key", "sample_date"],
        update_cols=["metric_label", "sample_start", "sample_end", "value", "unit"],
    )
    for metric_key, (col, label, unit) in PMS_KEYS.items():
        # col comes from the trusted PMS_KEYS literal above, never from input.
        row = conn.execute(
            f"""SELECT AVG({col}) v FROM voice_sessions
                WHERE user_id = ? AND valid = 1
                  AND substr(started_at, 1, 10) = ?
                  AND {col} IS NOT NULL""",
            (user_id, sample_date),
        ).fetchone()
        if row is None or row["v"] is None:
            continue
        conn.execute(upsert, (user_id, "voice", metric_key, label,
                              sample_date, None, None, float(row["v"]), unit))


@router.post("/sessions", status_code=201)
def create_session(
    req: VoiceSessionCreate,
    user: dict = Depends(require_auth),
    db=Depends(get_db),
):
    """POST /api/voice/sessions — record one in-app voice task session.

    Identity always comes from the JWT, never the request body.
    Returns HTTP 201 on first write and HTTP 201 with ``"idempotent": true``
    in the response body on replay (same client_session_id for the same user).
    """
    _require_athlete(user)
    if req.task_type not in VALID_TASK_TYPES:
        raise HTTPException(status_code=400, detail="Invalid task_type")

    user_id = user["id"]
    sample_date = req.started_at[:10]

    with get_connection() as conn:
        # Idempotency: existing (user_id, client_session_id) -> return prior row.
        existing = conn.execute(
            "SELECT id, valid, voice_score FROM voice_sessions "
            "WHERE user_id = ? AND client_session_id = ?",
            (user_id, req.client_session_id),
        ).fetchone()
        if existing:
            return {
                "id": existing["id"],
                "user_id": user_id,
                "valid": bool(existing["valid"]),
                "voice_score": existing["voice_score"],
                "speech_rate_z": None,
                "pause_ratio_z": None,
                "f0_sd_z": None,
                "jitter_z": None,
                "shimmer_z": None,
                "baseline_session_count": 0,
                "idempotent": True,
            }

        valid = _quality_ok(req)
        baseline = _load_prior_sessions(conn, user_id) if valid else []
        cmp = compare_with_baseline(req.model_dump(), baseline) if valid else {
            "speech_rate_z": None, "pause_ratio_z": None, "f0_sd_z": None,
            "jitter_z": None, "shimmer_z": None,
            "voice_score": None,
            "baseline_session_count": 0,
        }

        cur = conn.execute(
            """INSERT INTO voice_sessions
               (user_id, started_at, duration_seconds, task_type, prompt_id,
                client_session_id, valid,
                speech_rate, articulation_rate, pause_count, pause_ratio,
                mean_pause_ms, voice_onset_ms, f0_mean_hz, f0_sd_hz,
                jitter_pct, shimmer_pct, snr_db, voiced_seconds,
                speech_rate_z, pause_ratio_z, f0_sd_z, jitter_z, shimmer_z,
                voice_score, baseline_session_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (user_id, req.started_at, req.duration_seconds, req.task_type,
             req.prompt_id, req.client_session_id, 1 if valid else 0,
             req.speech_rate, req.articulation_rate, req.pause_count,
             req.pause_ratio, req.mean_pause_ms, req.voice_onset_ms,
             req.f0_mean_hz, req.f0_sd_hz, req.jitter_pct, req.shimmer_pct,
             req.snr_db, req.voiced_seconds,
             cmp["speech_rate_z"], cmp["pause_ratio_z"], cmp["f0_sd_z"],
             cmp["jitter_z"], cmp["shimmer_z"],
             cmp["voice_score"], cmp["baseline_session_count"]),
        )
        session_id = cur.lastrowid

        if valid:
            # 2. cognitive_test_results — voice as a cognitive signal.
            # raw_data carries derived features + z's only (never audio/text).
            raw = req.model_dump()
            raw.update({
                "speech_rate_z": cmp["speech_rate_z"],
                "pause_ratio_z": cmp["pause_ratio_z"],
                "f0_sd_z": cmp["f0_sd_z"],
                "jitter_z": cmp["jitter_z"],
                "shimmer_z": cmp["shimmer_z"],
                "voice_session_id": session_id,
            })
            conn.execute(
                """INSERT INTO cognitive_test_results
                   (id, user_id, test_type, score, raw_data, taken_at)
                   VALUES (?, ?, 'speech', ?, ?, ?)""",
                (
                    f"voice-{session_id}", user_id,
                    float(cmp["voice_score"]),
                    json.dumps(raw),
                    req.started_at,
                ),
            )
            # 3. personal_metric_samples daily aggregates.
            _upsert_daily_aggregates(conn, user_id, sample_date)

        conn.commit()

    # Post-commit hooks — all defensive, never break ingest.
    if valid:
        try:
            safe_record(
                patient_id=user_id, device_id="cognitive",
                activity_kinds=["voice"], event_date=sample_date,
                source="routers.voice_sessions", source_ref=str(session_id),
            )
        except Exception:
            logger.warning("voice: billing hook failed for %s", user_id)
        try:
            from api.clinical.readiness.service import mark_today_stale
            mark_today_stale(user_id)
        except Exception:
            logger.warning("voice: today stale hook failed for %s", user_id)
        try:
            from api.clinical.condition_scores import mark_condition_scores_stale
            mark_condition_scores_stale(user_id)
        except Exception:
            logger.warning("voice: condition-score stale hook failed for %s", user_id)
        try:
            from api.clinical.condition_scores.service import compute_voice_condition_scores
            from api.database import get_connection as _gc
            with _gc() as _c:
                compute_voice_condition_scores(
                    user_id=user_id, db=_c, date=sample_date
                )
                _c.commit()
        except Exception:
            logger.warning("voice: v2_voice condition compute failed for %s", user_id)
        try:
            from api.services.audit import log_action
            log_action(
                user_id, "voice_session_logged", "voice_session",
                str(session_id),
                {"task_type": req.task_type, "score": cmp["voice_score"]},
            )
        except Exception:
            logger.warning("voice: audit hook failed for %s", user_id)

    return {
        "id": session_id,
        "user_id": user_id,
        "valid": valid,
        "voice_score": cmp["voice_score"],
        "speech_rate_z": cmp["speech_rate_z"],
        "pause_ratio_z": cmp["pause_ratio_z"],
        "f0_sd_z": cmp["f0_sd_z"],
        "jitter_z": cmp["jitter_z"],
        "shimmer_z": cmp["shimmer_z"],
        "baseline_session_count": cmp["baseline_session_count"],
    }


# ---------------------------------------------------------------------------
# GET endpoints — Task 4
# ---------------------------------------------------------------------------

_BASELINE_COLS = (
    "speech_rate", "pause_ratio", "f0_sd_hz", "jitter_pct", "shimmer_pct",
)


@router.get("/sessions")
def list_sessions(
    user_id: Optional[str] = None,
    limit: int = 50,
    user: dict = Depends(require_auth),
    db=Depends(get_db),
):
    """Athlete reads own session history; linked clinician passes ?user_id=.

    Tenant model: resolve_patient_scope raises 403 if athlete A tries
    to pass ?user_id=B, or if a clinician is not linked to the target.
    """
    target = resolve_patient_scope(user, user_id)
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM voice_sessions WHERE user_id = ?
               ORDER BY started_at DESC LIMIT ?""",
            (target, min(max(limit, 1), 200)),
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/baseline")
def get_baseline(
    user_id: Optional[str] = None,
    user: dict = Depends(require_auth),
    db=Depends(get_db),
):
    """Per-metric baseline mean/std/n over usable valid sessions.

    Mirrors GET /api/typing/baseline. Linked clinicians may pass ?user_id=.

    NOTE — do NOT replace this implementation with the shared
    ``api.clinical.baseline_stats`` helper.  That helper aggregates over
    ``personal_metric_samples`` (one row per metric per day), which is the
    right source for readiness / Brain Score baselines.  This endpoint
    intentionally aggregates over raw per-session feature rows in
    ``voice_sessions`` (up to 60 individual sessions), so the baseline
    captures within-session variation rather than blending it into a daily
    mean first.  The two data sources have different semantics; substituting
    ``baseline_stats`` here would silently change the underlying population
    and break the z-score comparison logic in ``clinical/voice.py``.  This
    mirrors how the typing router computes its baseline directly from
    ``typing_sessions`` rows.
    """
    target = resolve_patient_scope(user, user_id)
    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT {', '.join(_BASELINE_COLS)}
               FROM voice_sessions
               WHERE user_id = ? AND valid = 1
                 AND duration_seconds >= ? AND voiced_seconds >= ?
               ORDER BY started_at DESC LIMIT 60""",
            (target, USABLE_MIN_DURATION_S, USABLE_MIN_VOICED_S),
        ).fetchall()
    count = len(rows)
    metrics: dict = {}
    for col in _BASELINE_COLS:
        vals = [r[col] for r in rows if r[col] is not None]
        n = len(vals)
        if n >= 2:
            mean = sum(vals) / n
            sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
            metrics[col] = {"mean": round(mean, 4), "std": round(sd, 4), "n": n}
        elif n == 1:
            metrics[col] = {"mean": round(vals[0], 4), "std": None, "n": 1}
        else:
            metrics[col] = {"mean": None, "std": None, "n": 0}
    return {"count": count, "metrics": metrics}


class VoiceConsentBody(BaseModel):
    """POST body for /api/voice/consent.

    extra="forbid" rejects any field not listed here, matching the pattern
    used by VoiceSessionCreate and the wider router convention.
    """
    model_config = ConfigDict(extra="forbid")

    consent_version: Optional[str] = Field(default=None, max_length=64)


@router.post("/consent")
def record_consent(
    body: VoiceConsentBody,
    user: dict = Depends(require_auth),
    db=Depends(get_db),
):
    """POST /api/voice/consent — record server-side voice consent (Spec §7).

    Writes an append-only audit_log row with event_type='voice_consent_granted'
    so the server has a durable record that the athlete granted voice consent.
    Idempotent / safe to call repeatedly — audit_log is append-only with no
    uniqueness constraint on this event type.  Returns {ok: true}.

    Athlete-only — the consent is for the athlete's own voice data.
    """
    _require_athlete(user)
    user_id = user["id"]
    detail: dict = {"channel": "in_app"}
    if body.consent_version is not None:
        detail["consent_version"] = body.consent_version
    try:
        from api.services.audit import log_action
        log_action(
            user_id, "voice_consent_granted", "voice_consent", None, detail
        )
    except Exception:
        logger.warning("voice: consent audit write failed for %s", user_id)
    return {"ok": True}


@router.get("/status")
def get_status(
    user: dict = Depends(require_auth),
    db=Depends(get_db),
):
    """Today-card support: has the athlete done a valid voice check today?

    Athlete-only — clinicians managing patient data go through the
    roster, not this personal-status endpoint.

    Day boundary: uses today_for_user (users.timezone, migration 059,
    UTC default) so the date matches the athlete's local calendar day,
    consistent with the typing /status endpoint.
    """
    _require_athlete(user)
    user_id = user["id"]
    with get_connection() as conn:
        today = today_for_user(user_id, conn)
        today_iso = today.isoformat()
        row = conn.execute(
            """SELECT COUNT(*) n FROM voice_sessions
               WHERE user_id = ? AND valid = 1
                 AND substr(started_at, 1, 10) = ?""",
            (user_id, today_iso),
        ).fetchone()
    done_today = (row["n"] or 0) > 0
    return {"done_today": done_today, "date": today_iso}
