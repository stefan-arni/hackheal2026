/**
 * Voice symptom entry API client — native dictation edition.
 *
 * Sprint: Native on-device dictation (2026-05-21).
 * Plan: docs/superpowers/plans/2026-05-21-sprint-native-on-device-dictation.md
 *
 * The upload + transcribe endpoints from the 2026-05-04 MediaRecorder sprint
 * are soft-deprecated (backend still serves them with X-Deprecated headers
 * during the 4-week sunset window ending 2026-08-01). This client no longer
 * exports `upload` or `transcribe`.
 *
 * The active endpoint is:
 *
 *   extract({ transcript }) → ExtractionResult
 *     POST /api/symptoms/voice/extract
 *     Takes a plain-text transcript (from the native dictation plugin) and
 *     returns structured symptom suggestions. On success the athlete reviews
 *     the pre-filled form and clicks Submit — that Submit is the affirmative
 *     act per SaMD positioning.
 *
 * SaMD positioning: extraction is a *suggestion* layer. The athlete
 * reviews + edits the pre-filled form before submitting.
 */
import { ApiError, apiClient } from './client';

// Re-export ApiError so callers that only import from voiceLog don't
// need a second import.
export { ApiError };

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ExtractedSymptom {
  symptom: string;
  severity: number | null;
  notes: string | null;
}

export interface ExtractionResult {
  symptoms: ExtractedSymptom[];
  /**
   * `null` on success. On failure: a stable identifier the UI can
   * map to user-facing copy. Examples:
   *   - "empty_transcript"
   *   - "schema_invalid_json"
   *   - "schema_not_object"
   *   - "schema_missing_symptoms_list"
   *   - "inference_error: <provider message>"
   */
  error: string | null;
  model_version: string;
}

export interface ExtractInput {
  transcript: string;
  /** Optional — pass when the transcript came from a backend audio_uploads
   *  row (legacy/AT scribe flow). The server will clean up the blob on
   *  extraction success. Omit for native-dictation transcripts (no blob
   *  ever uploaded). */
  upload_id?: number;
  /** AT-scribe mode: the athlete being scribed for. */
  target_user_id?: string;
  /** AT-scribe mode: the team grounding the consent gate. */
  team_id?: number;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export const voiceLog = {
  /**
   * Run structured-field extraction on a transcript.
   *
   * The transcript arrives from the native SpeechRecognizer plugin
   * (or from the editable textarea in the reviewing state). The server
   * passes it through the AI extraction layer and returns
   * `{ symptoms, error, model_version }`.
   *
   * Error handling: throws `ApiError` on HTTP errors. The caller also
   * checks `result.error` for in-body extraction failures (schema
   * issues, empty transcript, etc.) which surface as HTTP 200 with
   * `error: "<stable_id>"`.
   */
  extract(input: ExtractInput): Promise<ExtractionResult> {
    return apiClient.post<ExtractionResult>('/api/symptoms/voice/extract', input);
  },
};
