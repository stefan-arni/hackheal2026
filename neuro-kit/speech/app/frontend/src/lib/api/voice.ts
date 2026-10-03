/**
 * Voice biomarker API client.
 *
 * Sprint: Voice Biomarker Phase 3 (2026-06-12).
 * Plan: docs/superpowers/plans/2026-06-12-sprint-voice-biomarker-phase3.md
 *
 * Privacy invariant: no audio, no transcript, no waveform is ever sent.
 * Only derived acoustic numbers cross the network.
 *
 * SaMD positioning: the voice score is an *observational* signal.
 * Clinician remains the decision-maker at all times.
 *
 * Endpoints:
 *   POST /api/voice/sessions      → VoiceSessionResult
 *   GET  /api/voice/sessions      → VoiceSessionResult[]
 *   GET  /api/voice/baseline      → { count: number; metrics: Record<string, any> }
 *   GET  /api/voice/status        → VoiceStatus
 */
import { apiClient } from './client';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Acoustic features + session metadata sent to the backend.
 * All 12 acoustic fields are optional — the server accepts partial
 * feature sets and marks sessions whose coverage falls below the
 * quality gate as valid=false.
 */
export interface VoiceSessionPayload {
  /** ISO-8601 session start timestamp. */
  started_at: string;
  /** Recording duration in seconds. */
  duration_seconds: number;
  /** Task type: 'sustained_vowel' | 'read_passage' | 'combined'. */
  task_type: string;
  /** Prompt identifier (for read-passage tasks). */
  prompt_id?: string;
  /** Client-generated idempotency key. */
  client_session_id: string;

  // --- Acoustic features (all optional) ---
  /** Syllables per second (speaking rate). */
  speech_rate?: number;
  /** Syllables per second excluding pauses (articulation rate). */
  articulation_rate?: number;
  /** Number of inter-word pauses ≥ 250 ms. */
  pause_count?: number;
  /** Fraction of recording time spent in pauses. */
  pause_ratio?: number;
  /** Mean pause duration in milliseconds. */
  mean_pause_ms?: number;
  /** Voice onset time in milliseconds. */
  voice_onset_ms?: number;
  /** Mean fundamental frequency in Hz. */
  f0_mean_hz?: number;
  /** Standard deviation of fundamental frequency in Hz. */
  f0_sd_hz?: number;
  /** Jitter (cycle-to-cycle F0 variation) as a percentage. */
  jitter_pct?: number;
  /** Shimmer (cycle-to-cycle amplitude variation) as a percentage. */
  shimmer_pct?: number;
  /** Signal-to-noise ratio in dB. */
  snr_db?: number;
  /** Total voiced duration in seconds. */
  voiced_seconds?: number;
}

/** Response from POST /api/voice/sessions. */
export interface VoiceSessionResult {
  id: number;
  /** True when the session passed the server-side quality gate. */
  valid: boolean;
  /** 0–100 composite voice score; null when fewer than 5 usable sessions. */
  voice_score: number | null;
  /** How many usable (valid) sessions are in the athlete's baseline. */
  baseline_session_count?: number;
}

/** Response from GET /api/voice/status. */
export interface VoiceStatus {
  done_today: boolean;
  /** ISO-8601 date of the most recent valid session, if any. */
  date?: string;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export const voice = {
  /**
   * Submit an acoustic feature set for the authenticated athlete.
   * Returns the persisted session row including the computed voice score.
   */
  createSession(p: VoiceSessionPayload): Promise<VoiceSessionResult> {
    return apiClient.post<VoiceSessionResult>('/api/voice/sessions', p);
  },

  /** List the caller's voice sessions, newest first. */
  list(): Promise<VoiceSessionResult[]> {
    return apiClient.get<VoiceSessionResult[]>('/api/voice/sessions');
  },

  /**
   * Retrieve the caller's personal voice baseline statistics.
   * `count` is the number of usable sessions in the baseline window.
   * `metrics` maps feature key → { mean, std, n }.
   */
  baseline(): Promise<{ count: number; metrics: Record<string, any> }> {
    return apiClient.get<{ count: number; metrics: Record<string, any> }>('/api/voice/baseline');
  },

  /**
   * Check whether the athlete has already completed a voice session today.
   * Used by VoiceCheckCard to decide whether to show a completion badge
   * or the "Start" prompt.
   */
  status(): Promise<VoiceStatus> {
    return apiClient.get<VoiceStatus>('/api/voice/status');
  },

  /**
   * Record server-side consent for the voice biomarker (Spec §7).
   * Call once when the athlete accepts the consent gate in VoiceCheckTest.
   * Best-effort — callers must never block the flow on a rejection.
   *
   * @param consentVersion - optional version string (e.g. "v1") for future
   *   tracking of consent-copy changes.
   */
  consent(consentVersion?: string): Promise<{ ok: boolean }> {
    const body: Record<string, string> = {};
    if (consentVersion !== undefined) body.consent_version = consentVersion;
    return apiClient.post<{ ok: boolean }>('/api/voice/consent', body);
  },
};
