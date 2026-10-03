"""Structured-field extraction from voice transcripts.

Sprint: Voice-first symptom entry (2026-05-04 plan refresh).

Takes a free-text transcript ("I have a headache about a 5 out of 10
since lunch, light hurts my eyes") and returns a structured suggestion
the athlete can review + edit + Submit on the existing symptom-log
form.

SaMD positioning: extraction is a *suggestion* layer, never a clinical
decision. The athlete's Submit on the form is the affirmative act that
enters the symptom log. The extraction output is captured in the
audit trail so any later review can reconstruct what the model
proposed vs. what the athlete actually entered.

Decisions locked at sprint kickoff (2026-05-04):
  - Provider: Anthropic for both transcribe + extract (single BAA,
    one round-trip per upload). Reuses the AI inference adapter from
    the AI clinical-note sprint.
  - Same model handles both — no separate transcription model.
  - Hard-coded against the existing ``VALID_SYMPTOMS`` enum from
    ``routers/symptoms.py`` so the extraction can never propose a
    symptom the form would reject. Mirrors the SOAP-note prompt
    pattern of constraining LLM output via a tight system message.
  - SYSTEM_MESSAGE explicitly instructs: "if severity not stated,
    leave null" — defensive against severity hallucination (R3 in
    AI_CLINICAL_NOTE_RISK_REGISTER style risk taxonomy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from api.ai.inference import (
    InferenceClient,
    InferenceError,
    get_inference_client,
)


# Valid symptom enum — duplicated from routers/symptoms.py so the
# extractor can constrain the LLM without a circular import. Keep in
# sync if symptoms.py changes (regression test in
# test_voice_extraction.py pins this).
VALID_SYMPTOMS = [
    "headache",
    "nausea",
    "dizziness",
    "fogginess",
    "light_sensitivity",
    "noise_sensitivity",
    "fatigue",
    "sleep_disturbance",
    "irritability",
    "difficulty_concentrating",
    "memory_problems",
    "visual_disturbance",
    "balance_problems",
    "neck_pain",
]


PROMPT_VERSION = "extract_v1"


SYSTEM_MESSAGE = """You are a structured-data extractor for a brain-health
app. The user has spoken about how they feel. Your only job is to
extract what they said into a strict JSON object. You are NOT making
clinical decisions, recommendations, or diagnoses — the athlete will
review what you produce and edit it before submitting.

Output rules (these are absolute):

1. Output ONLY a single JSON object. No prose, no preamble, no
   markdown, no code fences. Just the raw JSON.

2. Schema:
   {{
     "symptoms": [
       {{
         "symptom": "<one of: {valid_symptoms}>",
         "severity": <integer 0-10, or null if not stated>,
         "notes": "<short verbatim phrase from the user, or null>"
       }}
     ]
   }}

3. The "symptom" field MUST be one of the values listed above. If the
   user describes a symptom that does not map cleanly to one of these,
   omit that symptom — do NOT invent a new one.

4. The "severity" field MUST be an integer between 0 and 10, OR null.
   If the user did not state a number, set severity to null.
   NEVER invent or estimate a severity number from descriptive language.

5. The "notes" field is a short verbatim quote from what the user
   actually said about that symptom (not a paraphrase, not an
   interpretation). Set to null if there's nothing notable to record.

6. If the user mentioned multiple distinct symptoms, return one entry
   per symptom in the "symptoms" array.

7. If the user did not describe any symptom that maps to the allowed
   list, return {{"symptoms": []}}.

8. NEVER include patient identifiers, names, dates, locations, or any
   other PHI in the "notes" field — only symptom-relevant phrases.

You are a translator, not an interpreter. Stick to what was actually
said.""".format(valid_symptoms=", ".join(VALID_SYMPTOMS))


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------


@dataclass
class ExtractedSymptom:
    """One symptom suggestion. Mirrors the symptom-log form fields."""

    symptom: str
    severity: Optional[int]  # None when not stated; UI requires explicit entry
    notes: Optional[str]


@dataclass
class ExtractionResult:
    """Result of running extract_symptoms_from_transcript.

    On success: ``symptoms`` is a list (possibly empty); ``error`` is None.
    On failure: ``symptoms`` is empty; ``error`` is a stable identifier
    string the router stores in audio_uploads.extraction_error so the
    blob can be preserved for debug per the locked retention decision.
    """

    symptoms: list[ExtractedSymptom]
    error: Optional[str]
    raw_response: Optional[str] = None  # captured for audit + debug
    model_version: str = ""


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _coerce_symptom_entry(raw: Any) -> Optional[ExtractedSymptom]:
    """Parse one entry from the LLM's JSON.

    Returns None if the entry is malformed (missing/wrong-type symptom,
    out-of-range severity, etc.) so the caller can drop it without
    failing the whole extraction.
    """
    if not isinstance(raw, dict):
        return None
    symptom = raw.get("symptom")
    if not isinstance(symptom, str) or symptom not in VALID_SYMPTOMS:
        return None

    severity_raw = raw.get("severity")
    severity: Optional[int]
    if severity_raw is None:
        severity = None
    elif isinstance(severity_raw, bool):
        # Bool is a subclass of int — exclude explicitly so True/False
        # never sneak through as severity values.
        return None
    elif isinstance(severity_raw, int):
        if 0 <= severity_raw <= 10:
            severity = severity_raw
        else:
            return None
    else:
        return None

    notes_raw = raw.get("notes")
    notes: Optional[str]
    if notes_raw is None:
        notes = None
    elif isinstance(notes_raw, str):
        notes = notes_raw.strip() or None
    else:
        return None

    return ExtractedSymptom(symptom=symptom, severity=severity, notes=notes)


def extract_symptoms_from_transcript(
    transcript: str,
    *,
    client: Optional[InferenceClient] = None,
    max_tokens: int = 512,
) -> ExtractionResult:
    """Extract structured symptom suggestions from a transcript.

    The result is a *suggestion* surface — the athlete reviews + edits
    + clicks Submit on the existing symptom-log form. This function
    never writes anything to the symptom_logs table.

    On any failure (transcript empty, LLM invalid JSON, schema
    mismatch, transient inference error) the result carries an
    ``error`` string identifier so the calling endpoint can preserve
    the audio blob for debug per the locked retention decision.
    """
    text = (transcript or "").strip()
    if not text:
        return ExtractionResult(
            symptoms=[],
            error="empty_transcript",
            raw_response=None,
            model_version=PROMPT_VERSION,
        )

    if client is None:
        client = get_inference_client()

    try:
        result = client.generate(
            prompt=text,
            max_tokens=max_tokens,
            system=SYSTEM_MESSAGE,
        )
    except InferenceError as exc:
        return ExtractionResult(
            symptoms=[],
            error=f"inference_error: {exc}",
            raw_response=None,
            model_version=f"{PROMPT_VERSION}",
        )

    raw = (result.text or "").strip()
    model_version = f"{result.model}:{PROMPT_VERSION}"

    # Strip optional ```json fences in case the model ignored rule 1.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip()
        # Drop a leading "json\n" if present.
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ExtractionResult(
            symptoms=[],
            error="schema_invalid_json",
            raw_response=raw,
            model_version=model_version,
        )

    if not isinstance(parsed, dict):
        return ExtractionResult(
            symptoms=[],
            error="schema_not_object",
            raw_response=raw,
            model_version=model_version,
        )

    raw_list = parsed.get("symptoms")
    if not isinstance(raw_list, list):
        return ExtractionResult(
            symptoms=[],
            error="schema_missing_symptoms_list",
            raw_response=raw,
            model_version=model_version,
        )

    extracted: list[ExtractedSymptom] = []
    for entry in raw_list:
        coerced = _coerce_symptom_entry(entry)
        if coerced is not None:
            extracted.append(coerced)

    # Empty list with valid shape is a SUCCESS, not an error — the
    # transcript may have been about something other than a symptom.
    return ExtractionResult(
        symptoms=extracted,
        error=None,
        raw_response=raw,
        model_version=model_version,
    )


# ===========================================================================
# AT observation extraction (sprint 2026-05-05 — voice symptom entry: AT
# extension)
# ===========================================================================
#
# When an athletic trainer dictates a sideline observation about an
# athlete ("athlete looked wobbly after the third quarter hit, complained
# of light sensitivity"), the extractor returns a structured observation
# suggestion: a short paragraph + a list of observed-sign tags drawn from
# a closed allow-list. The trainer reviews + edits in
# ObservationComposer, then clicks Submit — the Submit (not the audio
# upload) is the affirmative act that creates the team_observations row.
#
# The structure intentionally diverges from the symptom extraction:
#   - observation_text: ONE editable paragraph (not a list of items)
#   - observed_signs: short kebab-case tags from a fixed allow-list
#                     (NOT severity numbers — observations are
#                     qualitative; the severity story lives in the
#                     scribe-symptom flow which reuses extract_v1)
#
# Same defensive-parsing pattern as the symptom extractor:
#   - empty / whitespace transcript → empty result with error tag
#   - invalid JSON → schema_invalid_json
#   - top-level non-object → schema_not_object
#   - missing observation_text key → schema_missing_observation_text
#   - tags outside the allow-list silently dropped (the prompt instructs
#     the model to OMIT them but defense-in-depth here too)


OBSERVATION_PROMPT_VERSION = "observation_v1"


# Closed allow-list of observed-sign tags. Hard-coded here so the
# extractor can constrain the LLM without a circular import. Keep in
# sync with ObservationComposer.svelte's tag chip set.
OBSERVED_SIGN_TAGS = (
    "wobbly",
    "ataxia",
    "light-sensitivity",
    "sound-sensitivity",
    "reports-headache",
    "reports-nausea",
    "reports-dizziness",
    "slow-to-respond",
    "glassy-eyes",
    "confused-questions",
    "slurred-speech",
    "lost-consciousness",
    "vomiting",
    "visible-impact",
    "removed-from-play",
    "returned-after-evaluation",
)


OBSERVATION_SYSTEM_MESSAGE = """You are a structured-data extractor for a
brain-health app. An athletic trainer has just spoken about an athlete's
sideline appearance or recent behavior. Your only job is to extract what
they said into a strict JSON object. You are NOT making clinical
decisions, recommendations, or diagnoses — the trainer will review what
you produce, edit it, and click Submit before any record is saved.

Output rules (these are absolute):

1. Output ONLY a single JSON object. No prose, no preamble, no
   markdown, no code fences. Just the raw JSON.

2. Schema:
   {{
     "observation_text": "<a single concise paragraph in the trainer's
                           voice, no more than 280 characters>",
     "observed_signs": ["<short tag>", ...]
   }}

3. observation_text MUST be a faithful summary of what the trainer
   said. Past tense. Athlete-anonymous (use "the athlete" not their
   name). NEVER add detail the trainer did not state.

4. observed_signs MUST be drawn ONLY from this allow-list:
   {valid_tags}.
   If the trainer described a sign that does not map cleanly to one of
   these, OMIT that tag — do NOT invent new ones.

5. NEVER include athlete names, dates, locations, or any other PHI in
   observation_text or in any tag.

6. If the trainer described nothing actionable, return
   {{"observation_text": "", "observed_signs": []}}.

You are a translator, not an interpreter. Stick to what was actually
said.""".format(valid_tags=", ".join(OBSERVED_SIGN_TAGS))


@dataclass
class ExtractedObservation:
    """One observation suggestion. Mirrors ObservationComposer fields."""

    observation_text: str
    observed_signs: list[str]


@dataclass
class ObservationExtractionResult:
    """Result of running ``extract_observation_from_transcript``.

    On success: ``observation`` carries the parsed result and
    ``error`` is None. On failure: ``observation`` is None, ``error``
    is a stable identifier the router stores in
    ``audio_uploads.extraction_error`` so the blob can be preserved
    for debug per the locked retention decision.
    """

    observation: Optional[ExtractedObservation]
    error: Optional[str]
    raw_response: Optional[str] = None
    model_version: str = ""


def _coerce_observation(raw: Any) -> Optional[ExtractedObservation]:
    """Parse the LLM's JSON object into ExtractedObservation.

    Returns None if the shape is wrong; the caller turns this into a
    ``schema_*`` error code so the blob is preserved.
    """
    if not isinstance(raw, dict):
        return None
    text = raw.get("observation_text")
    if not isinstance(text, str):
        return None
    signs_raw = raw.get("observed_signs")
    if not isinstance(signs_raw, list):
        return None
    # Defense-in-depth: drop any tag outside the allow-list (or
    # non-string entries). The prompt forbids them, but trust nothing
    # the model outputs.
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in signs_raw:
        if not isinstance(tag, str):
            continue
        if tag not in OBSERVED_SIGN_TAGS:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return ExtractedObservation(
        observation_text=text.strip(),
        observed_signs=cleaned,
    )


def extract_observation_from_transcript(
    transcript: str,
    *,
    client: Optional[InferenceClient] = None,
    max_tokens: int = 512,
) -> ObservationExtractionResult:
    """Extract a structured observation suggestion from a trainer's
    transcript.

    The result is a *suggestion* surface — the trainer reviews + edits
    + clicks Submit on the existing ObservationComposer. This function
    NEVER writes anything to ``team_observations``.

    On any failure (transcript empty, LLM invalid JSON, schema mismatch,
    transient inference error) the result carries an ``error`` string
    identifier so the calling endpoint can preserve the audio blob for
    debug per the locked retention decision.
    """
    text = (transcript or "").strip()
    if not text:
        return ObservationExtractionResult(
            observation=None,
            error="empty_transcript",
            raw_response=None,
            model_version=OBSERVATION_PROMPT_VERSION,
        )

    if client is None:
        client = get_inference_client()

    try:
        result = client.generate(
            prompt=text,
            max_tokens=max_tokens,
            system=OBSERVATION_SYSTEM_MESSAGE,
        )
    except InferenceError as exc:
        return ObservationExtractionResult(
            observation=None,
            error=f"inference_error: {exc}",
            raw_response=None,
            model_version=f"{OBSERVATION_PROMPT_VERSION}",
        )

    raw = (result.text or "").strip()
    model_version = f"{result.model}:{OBSERVATION_PROMPT_VERSION}"

    # Strip optional ```json fences in case the model ignored rule 1.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip()
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ObservationExtractionResult(
            observation=None,
            error="schema_invalid_json",
            raw_response=raw,
            model_version=model_version,
        )

    if not isinstance(parsed, dict):
        return ObservationExtractionResult(
            observation=None,
            error="schema_not_object",
            raw_response=raw,
            model_version=model_version,
        )

    if "observation_text" not in parsed:
        return ObservationExtractionResult(
            observation=None,
            error="schema_missing_observation_text",
            raw_response=raw,
            model_version=model_version,
        )

    observation = _coerce_observation(parsed)
    if observation is None:
        return ObservationExtractionResult(
            observation=None,
            error="schema_invalid_observation_shape",
            raw_response=raw,
            model_version=model_version,
        )

    return ObservationExtractionResult(
        observation=observation,
        error=None,
        raw_response=raw,
        model_version=model_version,
    )


__all__ = [
    "VALID_SYMPTOMS",
    "PROMPT_VERSION",
    "SYSTEM_MESSAGE",
    "ExtractedSymptom",
    "ExtractionResult",
    "extract_symptoms_from_transcript",
    # Observation extraction (AT extension sprint 2026-05-05)
    "OBSERVED_SIGN_TAGS",
    "OBSERVATION_PROMPT_VERSION",
    "OBSERVATION_SYSTEM_MESSAGE",
    "ExtractedObservation",
    "ObservationExtractionResult",
    "extract_observation_from_transcript",
]
