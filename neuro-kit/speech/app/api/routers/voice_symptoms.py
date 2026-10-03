"""Voice-first symptom entry endpoints.

Sprint: 2026-05-04 plan refresh of 2026-05-02 sprint plan.

Three endpoints all gated behind ``Depends(require_auth)``:

- ``POST /api/symptoms/voice/upload`` (multipart audio blob; stores
  the blob in temporary storage; returns ``upload_id``)
- ``POST /api/symptoms/voice/transcribe/{upload_id}`` (reads the blob,
  sends it to the AI inference adapter, persists transcript +
  ``transcribed_at``, returns the transcript)
- ``POST /api/symptoms/voice/extract`` (body ``{transcript, upload_id?}``;
  runs the structured-field extractor; on success deletes the audio
  blob + sets ``deleted_at``; on extraction error preserves the blob
  + sets ``extraction_error``)

SaMD positioning: voice extraction is a *suggestion* layer. The
structured fields go into the existing symptom-log form pre-filled;
the athlete reviews + edits + clicks Submit on that form. The Submit
is the affirmative act, NOT the audio upload or the extraction.

Tenant isolation: every endpoint that takes an ``upload_id`` checks
``WHERE id=? AND user_id=?`` so cross-tenant access returns 404.

Audio retention (locked at sprint kickoff 2026-05-04): discard blob
on extraction success; preserve blob ONLY when extraction errors
(athlete can re-attempt without re-recording).

Per-user daily quota: ``VOICE_MAX_UPLOADS_PER_DAY`` env var
(default 30). Soft-fails with HTTP 429 + 'voice limit reached'
copy so the UI can fall back to manual entry.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict

from api.ai.inference import (
    InferenceAuthError,
    InferenceError,
    InferenceTransientError,
    get_inference_client,
)
from api.ai.voice_extraction import extract_symptoms_from_transcript
from api.auth.tenant_isolation import require_auth
from api.database import get_connection
from api.db import get_backend, sql
# Audit Phase 6 — these storage helpers moved to a shared service so
# teams.py can import them top-level instead of reaching into this router's
# privates. Aliased to the legacy private names so internal call sites here
# stay unchanged.
from api.services.audio_uploads import (
    audio_tmp_dir as _audio_tmp_dir,
    blob_path as _blob_path,
    write_blob as _write_blob,
    delete_blob_safely as _delete_blob_safely,
    load_upload_for_user as _load_upload_for_user,
)


log = logging.getLogger(__name__)

from api.dependencies import get_db
router = APIRouter(prefix="/api/symptoms/voice", tags=["voice-symptoms"])


# ---------------------------------------------------------------------------
# Configuration (env-driven)
# ---------------------------------------------------------------------------


# Allow-list of audio mime types — prevents the upload endpoint from
# accepting arbitrary binary data. Aligns with browser MediaRecorder
# defaults (Chrome → audio/webm, Safari → audio/mp4) plus a couple of
# common fallbacks.
ALLOWED_MIME_TYPES = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "audio/m4a",
    "audio/x-m4a",
}


def _max_upload_bytes() -> int:
    """Cap per-upload size. 5 MB by default — covers ~10min of Opus
    @ 24kbps with headroom. Override via env."""
    raw = os.environ.get("VOICE_MAX_UPLOAD_BYTES") or "5242880"  # 5 MB
    try:
        return max(1024, int(raw))
    except ValueError:
        return 5_242_880


def _max_uploads_per_day() -> int:
    """Cost guard. 30 uploads/day per user by default."""
    raw = os.environ.get("VOICE_MAX_UPLOADS_PER_DAY") or "30"
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


# ---------------------------------------------------------------------------
# Helpers (audio-blob storage lives in services/audio_uploads.py — see the
# aliased import above)
# ---------------------------------------------------------------------------


def _quota_used_today(db, user_id: str) -> int:
    """Count uploads in the last 24h for this user (soft cost guard)."""
    be = get_backend()
    row = db.execute(
        "SELECT COUNT(*) AS c FROM audio_uploads "
        f"WHERE user_id = ? AND created_at >= {sql.datetime_offset(be, -1)}",
        (user_id,),
    ).fetchone()
    return int(row["c"]) if row else 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


_DEPRECATION_HEADERS = {
    "X-Deprecated": "true",
    "X-Sunset": "2026-08-01",
}
"""Soft-deprecation headers added to /upload and /transcribe endpoints.

The native on-device dictation sprint (2026-05-21) replaces the
MediaRecorder + upload + transcribe flow with native OS speech
recognition. These endpoints remain active during the 4-week sunset
window (ending 2026-08-01) so any lingering old frontend builds keep
working. The frontend's voiceLog client no longer calls them.
"""


@router.post("/upload", status_code=201)
async def upload_voice_clip(
    file: UploadFile = File(...),
    mime_type: str = Form(...),
    user: dict = Depends(require_auth),
    response: Response = None, db=Depends(get_db),
):
    """DEPRECATED (sunset 2026-08-01) — Receive an audio blob.

    This endpoint is superseded by native on-device dictation
    (SpeechRecognizer Capacitor plugin). The frontend no longer calls
    this endpoint. Retained during the 4-week sunset window only.

    Returns ``{upload_id, byte_size, sha256}``.
    """
    log.warning(
        "DEPRECATED endpoint hit: POST /api/symptoms/voice/upload — "
        "frontend should be using native dictation. user_id=%s",
        user.get("id"),
    )
    if response is not None:
        for k, v in _DEPRECATION_HEADERS.items():
            response.headers[k] = v
    user_id = user["id"]
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio mime type. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
        )

    # Stream-read the upload with a hard size cap. Reading all into
    # memory is fine for the 5MB ceiling; for a future larger cap we
    # would chunk to disk.
    cap = _max_upload_bytes()
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    if len(audio_bytes) > cap:
        raise HTTPException(
            status_code=413,
            detail=f"Audio upload exceeds {cap} byte cap",
        )

    sha256 = hashlib.sha256(audio_bytes).hexdigest()

    with get_connection() as db:
        used = _quota_used_today(db, user_id)
        if used >= _max_uploads_per_day():
            raise HTTPException(
                status_code=429,
                detail="Voice limit reached for today. Please use manual symptom entry.",
            )
        cur = db.execute(
            "INSERT INTO audio_uploads (user_id, mime_type, byte_size, sha256) "
            "VALUES (?, ?, ?, ?)",
            (user_id, mime_type, len(audio_bytes), sha256),
        )
        upload_id = int(cur.lastrowid)
        db.commit()

    # Persist the blob AFTER the row commits so we never have a blob
    # without a row. (Reverse order would orphan blobs on commit
    # failure.)
    try:
        _write_blob(upload_id, audio_bytes)  # 0600, PHI (audit I1)
    except OSError as exc:
        # Roll the row back so the client can retry cleanly. The blob
        # may exist partially; remove it best-effort.
        log.warning("voice blob write failed for upload %s: %s", upload_id, exc)
        _delete_blob_safely(upload_id)
        with get_connection() as db:
            db.execute("DELETE FROM audio_uploads WHERE id = ?", (upload_id,))
            db.commit()
        raise HTTPException(
            status_code=500,
            detail="Audio storage write failed. Please retry.",
        )

    return {
        "upload_id": upload_id,
        "byte_size": len(audio_bytes),
        "sha256": sha256,
    }


@router.post("/transcribe/{upload_id}")
def transcribe_voice_clip(
    upload_id: int,
    user: dict = Depends(require_auth),
    response: Response = None, db=Depends(get_db),
):
    """DEPRECATED (sunset 2026-08-01) — Transcribe a previously uploaded blob.

    This endpoint is superseded by native on-device dictation
    (SpeechRecognizer Capacitor plugin). The frontend no longer calls
    this endpoint. Retained during the 4-week sunset window only.

    On success: persists transcript + transcribed_at on the row and
    returns ``{transcript, model}``.

    On inference failure: returns 503 and leaves transcribed_at NULL
    so the client can retry (audio blob still on disk).
    """
    log.warning(
        "DEPRECATED endpoint hit: POST /api/symptoms/voice/transcribe/%s — "
        "frontend should be using native dictation. user_id=%s",
        upload_id,
        user.get("id"),
    )
    if response is not None:
        for k, v in _DEPRECATION_HEADERS.items():
            response.headers[k] = v
    user_id = user["id"]
    with get_connection() as db:
        upload = _load_upload_for_user(db, upload_id, user_id)
    if upload is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    if upload["deleted_at"] is not None:
        # Successful extraction already cleaned this up. No way to
        # re-transcribe an already-deleted blob.
        raise HTTPException(
            status_code=410,
            detail="Audio has been removed (extraction completed). Re-record to retry.",
        )
    if upload["transcribed_at"] is not None and upload["transcript"]:
        # Idempotent — already transcribed. Return the cached value.
        return {
            "transcript": upload["transcript"],
            "model": "cached",
            "cached": True,
        }

    blob = _blob_path(upload_id)
    if not blob.exists():
        # Row exists but blob is gone — should only happen if an admin
        # cleaned the tmp dir manually. Surface as 410 so client knows
        # to re-record.
        raise HTTPException(
            status_code=410,
            detail="Audio blob is no longer available. Re-record to retry.",
        )

    audio_bytes = blob.read_bytes()
    client = get_inference_client()
    try:
        result = client.transcribe(
            audio_bytes=audio_bytes,
            mime_type=upload["mime_type"],
        )
    except (InferenceAuthError, InferenceError) as exc:
        # InferenceAuthError is also an InferenceError so the order
        # matters for the 503 vs 502 distinction:
        if isinstance(exc, InferenceAuthError):
            log.error("voice transcription auth failure: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Voice transcription is unavailable (provider auth).",
            )
        if isinstance(exc, InferenceTransientError):
            raise HTTPException(
                status_code=503,
                detail="Voice transcription is temporarily unavailable. Please retry.",
            )
        log.warning("voice transcription failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Voice transcription failed. Please retry.",
        )
    except NotImplementedError as exc:
        # The configured provider doesn't support audio (Anthropic
        # implementation pending). Surface clearly.
        log.error("voice transcription unsupported: %s", exc)
        raise HTTPException(
            status_code=501,
            detail=str(exc),
        )

    transcript = (result.text or "").strip()
    with get_connection() as db:
        db.execute(
            f"UPDATE audio_uploads SET transcript = ?, transcribed_at = {sql.now_expr(get_backend())} "
            "WHERE id = ? AND user_id = ?",
            (transcript, upload_id, user_id),
        )
        db.commit()

    return {
        "transcript": transcript,
        "model": result.model,
        "cached": False,
    }


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    """Body for ``POST /api/symptoms/voice/extract``.

    Two modes:

    - **Athlete-self (legacy)**: ``target_user_id`` and ``team_id`` both
      omitted. Caller is extracting symptoms they're about to log on
      themselves. Behaves exactly as the original 2026-05-04 endpoint.

    - **Scribe (AT extension, sprint 2026-05-05)**: caller is an
      AT/coach/head AT; ``target_user_id`` is the athlete being scribed
      for; ``team_id`` is the team the consent gate is grounded on.
      Both must be set together (one without the other → 400).
    """

    model_config = ConfigDict(extra="forbid")

    transcript: str
    upload_id: Optional[int] = None
    target_user_id: Optional[str] = None
    team_id: Optional[int] = None


def _assert_scribe_mode_allowed(
    *, caller_id: str, target_user_id: str, team_id: int
) -> None:
    """Gate scribe mode (target_user_id + team_id present in payload).

    Raises ``HTTPException(403)`` if any gate fails. Gates:

      1. Caller MUST be a current ``team_memberships`` row for
         ``team_id`` with a VALID_MEMBERSHIP_ROLES role (AT, head AT,
         coach). Athletes and clinicians are blocked at the role gate
         in the route, but this is defense-in-depth.
      2. Target athlete MUST be on the team's active roster.
      3. Active ``team_share_consents`` row for (athlete, team) MUST
         include ``at_symptom_scribe`` in its scope list.

    The gate is evaluated against the SPECIFIC team_id passed by the
    caller (per the locked decision 2026-05-05). A consent on a
    different team does not unlock scribe mode here.
    """
    from api.clinical.teams import service as teams_service
    from api.clinical.teams import consents as _consents

    with get_connection() as db:
        author_role = teams_service._user_team_membership(
            db, caller_id, team_id
        )
        if author_role not in teams_service.VALID_MEMBERSHIP_ROLES:
            raise HTTPException(status_code=403, detail="Access denied")
        on_roster = db.execute(
            "SELECT 1 FROM team_athletes "
            "WHERE team_id=? AND athlete_user_id=? AND departed_at IS NULL "
            "LIMIT 1",
            (team_id, target_user_id),
        ).fetchone()
        if not on_roster:
            raise HTTPException(status_code=403, detail="Access denied")
    scopes = _consents.get_share_scopes(target_user_id, team_id)
    if "at_symptom_scribe" not in scopes:
        raise HTTPException(status_code=403, detail="Access denied")


@router.post("/extract")
def extract_symptoms(
    body: ExtractRequest,
    user: dict = Depends(require_auth), db=Depends(get_db),
):
    """Run structured-field extraction on a transcript.

    The ``upload_id`` is optional — extract is callable on a free
    transcript too (e.g. after the user edits the transcribed text
    before extraction). When provided, the upload must belong to the
    caller (cross-tenant id → 404), AND the retention bookkeeping
    runs:

    - Extraction success → blob deleted, deleted_at set.
    - Extraction error → extraction_error stored on the row, blob
      preserved for debug per the locked retention decision.

    Scribe mode (sprint 2026-05-05): set ``target_user_id`` + ``team_id``
    together. Caller must be a team member (AT/coach/head AT) with the
    athlete on the active roster AND active ``at_symptom_scribe``
    consent on that specific team. Response includes the pass-through
    target_user_id + team_id so the frontend can route the Submit
    straight to /api/symptoms/scribe-log.
    """
    user_id = user["id"]

    # Scribe-mode gating (locked at sprint kickoff 2026-05-05)
    is_scribe = body.target_user_id is not None or body.team_id is not None
    if is_scribe:
        if body.target_user_id is None or body.team_id is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "target_user_id and team_id must be set together for "
                    "scribe-mode extraction"
                ),
            )
        # Role gate first (cheaper rejection path; no DB hits)
        if user.get("role") != "athletic_trainer":
            raise HTTPException(status_code=403, detail="Access denied")
        _assert_scribe_mode_allowed(
            caller_id=user_id,
            target_user_id=body.target_user_id,
            team_id=body.team_id,
        )

    upload: Optional[dict] = None
    if body.upload_id is not None:
        with get_connection() as db:
            upload = _load_upload_for_user(db, body.upload_id, user_id)
        if upload is None:
            raise HTTPException(status_code=404, detail="Upload not found")

    extraction = extract_symptoms_from_transcript(body.transcript)

    if body.upload_id is not None and upload is not None:
        with get_connection() as db:
            if extraction.error is None:
                # Success: drop the blob + mark deleted_at. Retention
                # decision: discard on success.
                db.execute(
                    f"UPDATE audio_uploads SET deleted_at = {sql.now_expr(get_backend())}, "
                    "extraction_error = NULL WHERE id = ? AND user_id = ?",
                    (body.upload_id, user_id),
                )
                db.commit()
                _delete_blob_safely(body.upload_id)
            else:
                # Failure: keep the blob, record the error. Athlete
                # may re-attempt; ops may inspect the audio later.
                db.execute(
                    "UPDATE audio_uploads SET extraction_error = ? "
                    "WHERE id = ? AND user_id = ?",
                    (extraction.error, body.upload_id, user_id),
                )
                db.commit()

    response = {
        "symptoms": [
            {"symptom": s.symptom, "severity": s.severity, "notes": s.notes}
            for s in extraction.symptoms
        ],
        "error": extraction.error,
        "model_version": extraction.model_version,
    }
    if is_scribe:
        # Pass-through so the frontend can route Submit straight to
        # /api/symptoms/scribe-log without re-deriving these fields
        response["target_user_id"] = body.target_user_id
        response["team_id"] = body.team_id
    return response