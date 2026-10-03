-- Migration 049 — audio_uploads
--
-- Sprint: Voice-first symptom entry (2026-05-04 plan refresh of
-- 2026-05-02 sprint plan). Phase 1.
-- Plan: docs/superpowers/plans/2026-05-02-sprint-voice-symptom-entry.md
--
-- Tracks transient audio uploads for voice-driven symptom entry.
-- An athlete records audio in the browser via MediaRecorder, the
-- blob is POSTed to /api/symptoms/voice/upload, this row is created,
-- and the blob is stored in temporary blob storage (filesystem in
-- dev, S3 in prod via env). Transcription + structured-field
-- extraction happen via the AI inference adapter; on success the
-- blob is deleted (deleted_at populated) and `transcript` is stored
-- for audit. On extraction error the blob is preserved for debug
-- and the athlete may re-attempt.
--
-- Decisions locked at sprint kickoff (2026-05-04):
--   - Provider: Anthropic for both transcribe + extract (single BAA,
--     reuses existing app/api/ai/ adapter from the AI clinical-note
--     sprint). No second BAA to negotiate.
--   - Retention: discard blob immediately on extraction success;
--     keep ONLY when extraction errors. The `deleted_at` column tracks
--     when the underlying blob was removed; the row itself is kept
--     forever for audit (no PII in the row when transcript is null,
--     and transcripts get the same SaMD-respectful audit treatment as
--     clinical notes).
--   - SaMD positioning: voice extraction is a *suggestion* layer,
--     never a clinical decision. The athlete reviews + edits + clicks
--     Submit on the symptom-log form; that submit is the affirmative
--     act, not the audio upload.
--
-- Pattern A1 (athlete-only) for the eventual Postgres+RLS port —
-- every row carries `user_id` NOT NULL with FK to users(id). Reads
-- and writes are scoped by JWT-derived user_id at the router layer.
--
-- Schema:
--   audio_uploads (
--     id                INTEGER PRIMARY KEY AUTOINCREMENT,
--     user_id           TEXT NOT NULL REFERENCES users(id),
--     mime_type         TEXT NOT NULL,
--     byte_size         INTEGER NOT NULL,
--     sha256            TEXT NOT NULL,
--     transcript        TEXT,
--     extraction_error  TEXT,
--     created_at        TEXT NOT NULL DEFAULT (datetime('now')),
--     transcribed_at    TEXT,
--     deleted_at        TEXT
--   )
--
-- Indexes:
--   idx_audio_uploads_user_created — covers the per-user history
--     query used by the daily-quota gate (count uploads in last 24h)
--     and any future "recent voice attempts" UI.

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS audio_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id),
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    transcript TEXT,
    extraction_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    transcribed_at TEXT,
    deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_audio_uploads_user_created
    ON audio_uploads(user_id, created_at DESC);

COMMIT;
