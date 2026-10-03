"""Audio-upload storage helpers (audit Phase 6 — break the router↔router
private import).

Shared by ``routers/voice_symptoms.py`` (athlete voice entry) and
``routers/teams.py`` (AT voice observation). ``teams.py`` previously
lazy-imported these as PRIVATE helpers from ``voice_symptoms``; relocating
them to a service lets both routers import them top-level and removes the
cross-router reach into another module's internals.

Pure storage/lookup: the ``audio_uploads`` row is the source of truth; the
blob on disk is transient and best-effort (deleted on extraction success
per the locked retention decision).
"""
import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def audio_tmp_dir() -> Path:
    """Where audio blobs live transiently. Filesystem in dev, overridable
    via ``VOICE_AUDIO_TMP_DIR`` (typical prod value: /var/lib/voice-audio).
    Created on demand; never relied upon for durability.

    Hardened to 0700 (security audit 2026-07-04, I1): these blobs are PHI
    (athlete symptom voice recordings), and the default lands in a shared
    ``/tmp`` where any local account could otherwise read them."""
    raw = os.environ.get("VOICE_AUDIO_TMP_DIR") or "/tmp/voice-audio"
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError as exc:  # pragma: no cover — non-owner on a pre-existing dir
        log.warning("could not chmod audio dir %s to 0700: %s", path, exc)
    return path


def blob_path(upload_id: int) -> Path:
    return audio_tmp_dir() / f"{upload_id}.bin"


def write_blob(upload_id: int, audio_bytes: bytes) -> Path:
    """Write a PHI voice blob with owner-only (0600) permissions.

    Single write path so no caller can create a world-readable blob via a
    bare ``write_bytes`` under the process umask (security audit I1). Opens
    with O_CREAT|O_EXCL at mode 0600 so the restrictive bits apply from
    creation, never a widen-then-narrow window."""
    path = blob_path(upload_id)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        # O_CREAT's mode only applies to a NEW file; fchmod also repairs a
        # pre-existing wider-mode blob (e.g. a legacy 0644 file from the old
        # write_bytes path) before we write PHI into it.
        os.fchmod(fd, 0o600)
        os.write(fd, audio_bytes)
    finally:
        os.close(fd)
    return path


def load_upload_for_user(db, upload_id: int, user_id: str) -> Optional[dict]:
    """Return the upload row IFF it belongs to ``user_id``, else None.

    Tenant boundary — cross-user access surfaces as 404 from the caller,
    not 403, so another user's upload_id existence doesn't leak."""
    row = db.execute(
        "SELECT id, user_id, mime_type, byte_size, sha256, transcript, "
        "extraction_error, created_at, transcribed_at, deleted_at "
        "FROM audio_uploads WHERE id = ? AND user_id = ?",
        (upload_id, user_id),
    ).fetchone()
    return dict(row) if row else None


def delete_blob_safely(upload_id: int) -> None:
    """Filesystem deletion that never raises — the audit row is the source
    of truth, blob storage is best-effort."""
    try:
        path = blob_path(upload_id)
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover — defensive
        log.warning("failed to delete voice blob %s: %s", upload_id, exc)
