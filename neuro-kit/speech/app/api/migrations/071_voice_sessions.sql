-- Migration 071 — voice_sessions: in-app voice acoustics biomarker (Pattern A1).
-- Spec: docs/superpowers/specs/2026-06-12-voice-biomarker-design.md
--
-- PRIVACY INVARIANT: derived acoustic features ONLY. There is NO column for
-- audio, waveform, PCM, or transcript — by design, forever. The frontend
-- processes audio on-device, in memory, and posts numbers.
--
-- Z-scores + voice_score are computed server-side at POST time against the
-- athlete's own prior usable sessions (>=5 usable) — mirrors typing_sessions.
--
-- Pattern A1 (athlete-only): user_id NOT NULL + FK users(id); reads are
-- JWT-scoped at the router; maps to Postgres RLS Pattern A1 later.
-- UNIQUE(user_id, client_session_id) makes POST idempotent — retry on network
-- failure is safe.

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS voice_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id),
    started_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL CHECK (duration_seconds > 0),
    task_type TEXT NOT NULL DEFAULT 'read_passage'
        CHECK (task_type IN ('read_passage', 'sustained_vowel', 'combined')),
    prompt_id TEXT,
    client_session_id TEXT NOT NULL,
    valid INTEGER NOT NULL DEFAULT 1,

    -- client-derived acoustic features (acousticFeatures.ts engine)
    speech_rate REAL,
    articulation_rate REAL,
    pause_count INTEGER,
    pause_ratio REAL,
    mean_pause_ms REAL,
    voice_onset_ms REAL,
    f0_mean_hz REAL,
    f0_sd_hz REAL,
    jitter_pct REAL,
    shimmer_pct REAL,
    snr_db REAL,
    voiced_seconds REAL,

    -- server-computed baseline comparison (voice.py engine)
    speech_rate_z REAL,
    pause_ratio_z REAL,
    f0_sd_z REAL,
    jitter_z REAL,
    shimmer_z REAL,
    voice_score INTEGER CHECK (voice_score IS NULL OR (voice_score >= 0 AND voice_score <= 100)),
    baseline_session_count INTEGER NOT NULL DEFAULT 0,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE (user_id, client_session_id)
);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_user_started
    ON voice_sessions(user_id, started_at DESC);

COMMIT;
