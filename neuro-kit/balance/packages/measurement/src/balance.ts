/**
 * Instrumented standing-balance module for three cards that share every derived metric and every
 * core quality/invalidity rule: BFP-BAL-FT-EO-001 (feet together, eyes open), BFP-BAL-FT-EC-001
 * (feet together, eyes closed), and BFP-BAL-TAN-EO-001 (tandem stance, eyes open). One evaluator,
 * `evaluateBalanceSession`, takes the card id as a rule variant rather than being copy-pasted
 * three times; cards 10 and 11 each add exactly the extra rule their card text authors (eyes-open
 * cumulative seconds; heel-toe contact loss and lead foot) on top of the shared base.
 *
 * Pure function from two recorded trials (repetitions: "One ... familiarization; two recorded
 * trials" -- the familiarization is never scored and this module never sees it) to derived
 * metrics, quality flags, and invalidity reasons. The clinical thresholds are the cards', cited
 * inline. Each recorded trial is evaluated independently for its own validity, the same
 * per-trial/session split BFP-MOTOR-TAP-001 established: an invalid trial's own metrics are
 * null ("retain the stop/event record but do not fabricate sway metrics"), but the session as a
 * whole stays valid as long as at least one trial is usable, exactly matching how
 * packages/task-flow/src/index.ts already counts "valid trials of this card in this session" for
 * downstream prerequisites (BFP-BAL-TAN-EO-001 needs >=1 valid FT-EO trial in the same session;
 * BFP-BAL-FT-EC-001 needs both). The session is invalid only when BOTH trials are.
 *
 * Consumes the proven per-channel statistics from @platform/capture (`completenessPercent`,
 * `largestGapMs`) rather than recomputing them: the cards threshold on those numbers directly.
 * Everything else the capture layer has not already proven -- orientation, device shift, event
 * flags, and the two card-specific signals -- is proven here the same way DEF-017 requires:
 * literal booleans only, refuse rather than read falsy, refuse rather than throw.
 *
 * ENGINEERING INTERPRETATIONS the cards leave open, made once here, documented, and
 * fixture-pinned (flagged for the clinical owner's veto in state.md):
 *
 *  1. This module consumes ALREADY-RESOLVED anterior-posterior (AP) and medio-lateral (ML)
 *     acceleration channels plus an already-resolved angular-velocity channel. It does not
 *     project raw device-frame motion onto anatomical axes itself: that projection needs the
 *     belt-mounting geometry and the "validated orientation" reference, which are not specified
 *     in the card, the engineering overlay, or the capture boundary. The projection is assumed to
 *     happen upstream, exactly as this module consumes derived completeness/gap statistics rather
 *     than raw sample arrays for those.
 *  2. resultant_acceleration_rms is the RMS of the PLANAR resultant sqrt(ap^2 + ml^2) at each
 *     paired sample, excluding the vertical axis: the card authors no vertical channel or metric,
 *     and postural-sway "resultant" conventionally refers to the horizontal AP-ML plane. AP and
 *     ML must report IDENTICAL sample times to be paired; a mismatch cannot be resolved here and
 *     invalidates the trial (`acceleration_channels_misaligned`) rather than pairing mismatched
 *     instants.
 *  3. Jerk is the derivative of acceleration (card silent on the statistical definition): a
 *     forward finite difference per consecutive sample pair, divided by THAT pair's own elapsed
 *     time in seconds, so one gap's larger interval does not corrupt a neighboring tightly-spaced
 *     one. A zero-duration pair (two samples sharing a timestamp, which @platform/capture treats
 *     as a plausible sensor batch) contributes no term rather than an infinite or NaN rate.
 *  4. RMS over a gappy stream is the RMS over only the samples actually present (never a
 *     fabricated zero for a missing one): the card's own completeness/gap rules already gate
 *     whether the stream is trustworthy enough to reach this computation at all.
 *  5. dominant_frequency_by_axis (per AP and ML only; angular velocity and the acceleration
 *     resultant have no "by axis" split authored) is the frequency of the largest-magnitude bin
 *     of a direct discrete Fourier transform. The bin-to-Hz mapping uses the AVERAGE OBSERVED rate
 *     derived from the channel's own proven first/last timestamps and sample count, not the
 *     caller-declared `requestedRateHz`: an independent security review found that trusting the
 *     declared rate silently mis-scales the reported frequency whenever true sample density
 *     differs from it, which the card's own completeness rule cannot catch (it authors only a
 *     LOWER bound). DC (bin 0) is excluded: it is the signal's mean offset, not an oscillation. A
 *     tie keeps the smaller bin. A channel with no variation at all has no oscillation to report
 *     and is null, never the arbitrary lowest bin. A channel larger than MAX_DFT_SAMPLES (10,000;
 *     the card's own window and rate never legitimately produce this many) also reports null: the
 *     direct DFT is O(n^2), and an unbounded one is an event-loop-blocking payload, not a clinical
 *     judgement.
 *  6. Sample completeness and timestamp gap are each SINGLE per-trial numbers combined across the
 *     trial's three channels: completeness is the MINIMUM completenessPercent (the worst-served
 *     channel bounds trust in the whole trial) and gap is the MAXIMUM largestGapMs (a null gap,
 *     from a channel with fewer than two samples, contributes nothing to the maximum).
 *  7. Initial device orientation deviation and device shift are consumed as ALREADY-COMPUTED
 *     per-trial degree values, not derived here from raw orientation samples: resolving "degrees
 *     from the validated orientation" needs the same belt/device geometry as (1) and is not this
 *     module's job, mirroring how BFP-MOTOR-TAP-001 takes `deviceMovedMm` as a supplied number
 *     rather than integrating raw accelerometer displacement.
 *  8. "The two valid trial values differ by more than 30% for a primary RMS metric" is read as
 *     ALL SIX metrics whose id ends `_rms` (both acceleration and jerk, plus angular velocity),
 *     because the card does not name a narrower "primary" subset and nothing distinguishes one
 *     RMS metric from another as more "primary". The alternative reading -- only the three
 *     acceleration RMS metrics -- is equally defensible and is flagged as an open question.
 *  9. The percent difference for that comparison is the SYMMETRIC "percentage difference"
 *     |trial1 - trial2| / mean(trial1, trial2) * 100, not a signed "percentage change" relative
 *     to either trial specifically: trial 1 and trial 2 are two repetitions of the same task, not
 *     a before/after pair, so no trial is privileged as the reference.
 * 10. BFP-BAL-TAN-EO-001's `leadFoot` is typed "correct" | "incorrect" (a verdict against the
 *     card's own positioning rule), not the actual foot ("left" | "right"). This correctly
 *     implements the authored "the wrong lead foot is used" invalidity rule for a SINGLE trial,
 *     and an independent security review confirmed it does not silently break the card's OWN
 *     "same lead foot across both trials" requirement (repetitions: "two recorded trials using the
 *     same lead foot"): if the mapping from "correct" to a physical foot is fixed for the session
 *     (true absent an "approved protocol configuration" override this module never sees), two
 *     "correct" trials necessarily used the same physical foot. What it CANNOT support is
 *     comparing WHICH foot was used across separate SESSIONS for the card's own
 *     serial_within_patient_trajectory comparator (positioning: "held constant across
 *     comparisons"), because the actual foot is never recorded. No consumer of that cross-session
 *     comparison exists yet in this codebase (reference_comparator has no implementation anywhere),
 *     so this is carried rather than fixed: recording the physical foot would need a
 *     patient-dominant-side input this module does not have, mirroring TAP's session-level
 *     `handedness` field. Flagged for the clinical/architecture owner alongside interpretation 8.
 *
 * center_of_pressure_metric is reportable:false and has no authored formula anywhere on any of the
 * three cards. This module will not invent one: the value is always null, exactly like
 * composite_inhibition_score in gng.ts.
 */
import type { ProvenChannel } from "@platform/capture";

export type BalanceCardId = "BFP-BAL-FT-EO-001" | "BFP-BAL-FT-EC-001" | "BFP-BAL-TAN-EO-001";

export interface BalanceTrial {
  /** Already-resolved anterior-posterior linear acceleration (see header, interpretation 1). */
  readonly apAcceleration: ProvenChannel;
  /** Already-resolved medio-lateral linear acceleration (see header, interpretation 1). */
  readonly mlAcceleration: ProvenChannel;
  /** Already-resolved angular velocity (see header, interpretation 1). */
  readonly angularVelocity: ProvenChannel;
  readonly startedAtMs: number;
  readonly endedAtMs: number;
  /** Card: "initial device orientation differs ... from the validated orientation" (degrees). */
  readonly initialOrientationDeviationDegrees: number;
  /** Card: "the device shifts ... during the window" (degrees; see header, interpretation 7). */
  readonly deviceShiftDegrees: number;
  /** Card: "a step ... occurs". */
  readonly stepOccurred: boolean;
  /** Card: "external support ... occurs". */
  readonly externalSupportUsed: boolean;
  /** Card: "hands-off-hips balance correction ... occurs". */
  readonly handsOffHipsCorrection: boolean;
  /** Card: "observer contact ... occurs". */
  readonly observerContact: boolean;
  /** Card: "fall/near-fall occurs". */
  readonly fallOrNearFall: boolean;
  /** BFP-BAL-FT-EC-001 only. Card: "eyes are open for a cumulative ... second(s)"; null is the
   * card's own explicit "eye-state confirmation is unavailable", not a proving failure. */
  readonly eyesOpenCumulativeSeconds?: number | null;
  /** BFP-BAL-TAN-EO-001 only. Card: "heel-to-toe contact is lost for ... second(s)". */
  readonly heelToeContactLossSeconds?: number;
  /** BFP-BAL-TAN-EO-001 only. Card: "foot position cannot be verified". */
  readonly footPositionVerified?: boolean;
  /** BFP-BAL-TAN-EO-001 only. Card: "the wrong lead foot is used". */
  readonly leadFoot?: "correct" | "incorrect";
}

export interface BalanceSession {
  readonly cardId: BalanceCardId;
  /** Exactly the two recorded trials; the familiarization is not one of them. */
  readonly trials: readonly BalanceTrial[];
}

export interface BalanceResult {
  readonly valid: boolean;
  readonly invalidReasons: readonly string[];
  readonly qualityFlags: readonly string[];
  readonly metrics: Readonly<Record<string, unknown>> | null;
}

/** Card: "repetitions: One ... familiarization; two recorded trials." */
export const RECORDED_TRIALS_PER_SESSION = 2;
/** Card: "the two valid trial values differ by more than 30% for a primary RMS metric". */
export const TRIAL_VARIABILITY_OVER_PERCENT = 30;

const RMS_METRIC_IDS = {
  apAccelerationRms: "ap_acceleration_rms",
  mlAccelerationRms: "ml_acceleration_rms",
  resultantAccelerationRms: "resultant_acceleration_rms",
  apJerkRms: "ap_jerk_rms",
  mlJerkRms: "ml_jerk_rms",
  angularVelocityRms: "angular_velocity_rms",
} as const;
type RmsMetricKey = keyof typeof RMS_METRIC_IDS;

interface TrialMetrics {
  readonly apAccelerationRms: number | null;
  readonly mlAccelerationRms: number | null;
  readonly resultantAccelerationRms: number | null;
  readonly apJerkRms: number | null;
  readonly mlJerkRms: number | null;
  readonly angularVelocityRms: number | null;
  readonly dominantFrequencyAp: number | null;
  readonly dominantFrequencyMl: number | null;
  readonly completedDurationSeconds: number | null;
}

interface TrialOutcome {
  readonly invalidTrialReasons: readonly string[];
  readonly bandFlags: readonly string[];
  /** Null exactly when the trial is invalid: no fabricated sway metrics for a stopped trial. */
  readonly metrics: TrialMetrics | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * A sparse array (a hole from `new Array(n)`, `delete arr[i]`, or elision) is transport
 * corruption, not a value, and `structuredClone` (the capture boundary's own defense) preserves
 * holes rather than closing them. This matters because `Array.prototype.every`/`reduce` SILENTLY
 * SKIP holes rather than failing on them (a hole is not "visited" at all) -- exactly the shape
 * that would silently dilute an RMS with the missing samples' true magnitude never included in
 * the sum, while `.length` still counts them in the denominator. Independently verified before
 * writing this function: `[1,2,3]` with a hole added at a later index reports
 * `.every(Number.isFinite) === true`, and `.reduce` invokes its callback only for the real
 * elements. The fix is the explicit indexed loop below, NOT an extra `Object.hasOwn` check:
 * `value[i]` on a hole reads as `undefined` via ordinary property access (confirmed independently:
 * unlike `.every`, indexing does not skip it), which already fails `typeof v !== "number"`. A
 * seeded mutation adding `Object.hasOwn` back as a first check is therefore an EQUIVALENT
 * MUTATION under this loop shape and was removed rather than kept as untested code (matching the
 * DEF-011/DEF-017 standing rule that dead code does not get to look like a control).
 */
function isDenseFiniteNumberArray(value: unknown): value is number[] {
  if (!Array.isArray(value)) return false;
  // for...of (unlike .every/.reduce) visits every index including holes, reading a hole as
  // `undefined` -- confirmed independently -- which is exactly what the typeof check below needs.
  for (const v of value as unknown[]) {
    if (typeof v !== "number" || !Number.isFinite(v)) return false;
  }
  return true;
}

/**
 * The shape this module trusts after re-proving a channel: narrower than @platform/capture's own
 * `ProvenChannel`, whose `completenessPercent` is `number | null` to allow an event-driven channel
 * (a discrete-event stream with no sampling rate, signalled by `isEventDriven`) to report "not a
 * fact about this channel" rather than a fabricated percentage. This module's three channels are
 * always continuous streams, never event-driven, so `completenessPercent` must be a genuine number
 * here; an event-driven channel is refused by the same `typeof !== "number"` check that refuses
 * any other unestablished completeness reading, with no separate `isEventDriven` check needed.
 */
interface UsableChannel {
  readonly values: readonly number[];
  readonly sampleTimesMs: readonly number[];
  readonly completenessPercent: number;
  readonly largestGapMs: number | null;
  readonly requestedRateHz: number;
}

/**
 * The minimal re-proving this module needs of a channel it did not itself prove: it must be able
 * to trust the arithmetic it reads directly (`values`, `sampleTimesMs`, `completenessPercent`,
 * `largestGapMs`, `requestedRateHz`). This is deliberately NOT a re-implementation of
 * @platform/capture's own proving (sampleCount, expectedSampleCount): those are not read here.
 * Monotonicity of `sampleTimesMs` IS re-proven here (@platform/capture already proves it too):
 * jerk and the dominant-frequency rate estimate below both assume non-decreasing time.
 */
function isUsableChannel(value: unknown): value is UsableChannel {
  if (!isRecord(value)) return false;
  if (!isDenseFiniteNumberArray(value.values)) return false;
  const times: unknown = value.sampleTimesMs;
  if (!isDenseFiniteNumberArray(times) || times.length !== value.values.length) return false;
  for (let i = 1; i < times.length; i++) {
    const prev = times[i - 1];
    const here = times[i];
    if (prev === undefined || here === undefined || here < prev) return false;
  }
  if (
    typeof value.completenessPercent !== "number" ||
    !Number.isFinite(value.completenessPercent)
  ) {
    return false;
  }
  const gap: unknown = value.largestGapMs;
  if (gap !== null && (typeof gap !== "number" || !Number.isFinite(gap) || gap < 0)) return false;
  const rate: unknown = value.requestedRateHz;
  if (typeof rate !== "number" || !Number.isFinite(rate) || rate <= 0) return false;
  return true;
}

function rmsOrNull(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const meanSquare = values.reduce((acc, v) => acc + v * v, 0) / values.length;
  return Math.sqrt(meanSquare);
}

/** See header, interpretation 3. */
function jerkSeries(values: readonly number[], timesMs: readonly number[]): number[] {
  const jerks: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const v0 = values[i - 1];
    const v1 = values[i];
    const t0 = timesMs[i - 1];
    const t1 = timesMs[i];
    if (v0 === undefined || v1 === undefined || t0 === undefined || t1 === undefined) continue;
    const dtSeconds = (t1 - t0) / 1000;
    if (dtSeconds > 0) jerks.push((v1 - v0) / dtSeconds);
  }
  return jerks;
}

/**
 * A direct DFT is O(n^2); an independent security review measured 55.8 s at n=60,000 on a single
 * channel, an event-loop-blocking payload the card's own physical parameters (a 20 s window) never
 * produce legitimately. Refusing beyond a generous multiple of the card's target ~2,000 samples
 * (100 Hz x 20 s) is a computational bound, not a clinical threshold: it never fires for the
 * authored device rate, and an oversized channel already carries no established frequency answer
 * more than an absent one would.
 */
const MAX_DFT_SAMPLES = 10_000;

/** See header, interpretation 5. */
function dominantFrequencyHz(values: readonly number[], timesMs: readonly number[]): number | null {
  const n = values.length;
  if (n < 2 || n > MAX_DFT_SAMPLES) return null;
  const first = values[0];
  if (first === undefined) return null;
  if (values.every((v) => v === first)) return null;

  // The bin-to-Hz mapping needs the TRUE average sample rate. Deriving it from the channel's own
  // proven timestamps (already checked finite and non-decreasing by isUsableChannel), rather than
  // trusting the caller-declared requestedRateHz, closes a real gap an independent security review
  // found: a channel whose actual sample density differs from its declared rate (the completeness
  // rule authors only a LOWER bound, so denser-than-declared passes clean) previously reported a
  // frequency scaled by that same ratio, silently wrong on a reportable:true metric.
  const firstTimeMs = timesMs[0];
  const lastTimeMs = timesMs[n - 1];
  if (firstTimeMs === undefined || lastTimeMs === undefined) return null;
  const elapsedSeconds = (lastTimeMs - firstTimeMs) / 1000;
  if (elapsedSeconds <= 0) return null;
  const observedRateHz = (n - 1) / elapsedSeconds;

  const maxBin = Math.floor(n / 2);
  let bestBin = 1;
  let bestMagnitudeSquared = -1;
  for (let k = 1; k <= maxBin; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const value = values[t];
      if (value === undefined) continue;
      const angle = (2 * Math.PI * k * t) / n;
      re += value * Math.cos(angle);
      im -= value * Math.sin(angle);
    }
    const magnitudeSquared = re * re + im * im;
    if (magnitudeSquared > bestMagnitudeSquared) {
      bestMagnitudeSquared = magnitudeSquared;
      bestBin = k;
    }
  }
  return (bestBin * observedRateHz) / n;
}

function evaluateTrial(trial: BalanceTrial, cardId: BalanceCardId): TrialOutcome {
  const reasons: string[] = [];
  const bandFlags: string[] = [];

  // Duration. Card: "usable duration is below 18.0 seconds" invalid; "18.0-19.9 seconds" quality.
  const startedAtMs: unknown = trial.startedAtMs;
  const endedAtMs: unknown = trial.endedAtMs;
  const durationEstablished =
    typeof startedAtMs === "number" &&
    Number.isFinite(startedAtMs) &&
    typeof endedAtMs === "number" &&
    Number.isFinite(endedAtMs) &&
    endedAtMs >= startedAtMs;
  if (!durationEstablished) reasons.push("duration_unestablished");
  const durationSeconds =
    durationEstablished && typeof startedAtMs === "number" && typeof endedAtMs === "number"
      ? (endedAtMs - startedAtMs) / 1000
      : null;
  if (durationSeconds !== null) {
    if (durationSeconds < 18.0) reasons.push("usable_duration_below_18s");
    if (durationSeconds >= 18.0 && durationSeconds <= 19.9) {
      bandFlags.push("usable_duration_18_0_to_19_9");
    }
  }

  // Channels. Card thresholds directly on @platform/capture's own completeness/gap statistics.
  const apUnknown: unknown = trial.apAcceleration;
  const mlUnknown: unknown = trial.mlAcceleration;
  const avUnknown: unknown = trial.angularVelocity;
  const apOk = isUsableChannel(apUnknown);
  const mlOk = isUsableChannel(mlUnknown);
  const avOk = isUsableChannel(avUnknown);
  if (!apOk) reasons.push("ap_acceleration_channel_unestablished");
  if (!mlOk) reasons.push("ml_acceleration_channel_unestablished");
  if (!avOk) reasons.push("angular_velocity_channel_unestablished");

  let aligned = false;
  if (apOk && mlOk && avOk) {
    const ap = apUnknown;
    const ml = mlUnknown;
    // Card: "sample completeness is below 90%" invalid; "90-94.9%" quality (see header, 6).
    // No null-handling here: `isUsableChannel` already proves completenessPercent is a finite
    // number, so a channel whose expectation the capture boundary could not establish (a
    // degenerate window, or an event-driven channel with no sampling rate) has already been
    // refused above as unusable. Re-checking here would be dead code dressed as a control.
    const worstCompleteness = Math.min(
      ap.completenessPercent,
      ml.completenessPercent,
      avUnknown.completenessPercent,
    );
    if (worstCompleteness < 90) reasons.push("sample_completeness_below_90pct");
    if (worstCompleteness >= 90 && worstCompleteness <= 94.9) {
      bandFlags.push("sample_completeness_90_to_94_9pct");
    }
    // Card: "any timestamp gap exceeds 100 ms" invalid; "50-100 ms" quality (see header, 6).
    let worstGapMs: number | null = null;
    for (const c of [ap, ml, avUnknown]) {
      if (c.largestGapMs !== null && (worstGapMs === null || c.largestGapMs > worstGapMs)) {
        worstGapMs = c.largestGapMs;
      }
    }
    if (worstGapMs !== null) {
      if (worstGapMs > 100) reasons.push("timestamp_gap_over_100ms");
      if (worstGapMs >= 50 && worstGapMs <= 100) bandFlags.push("timestamp_gap_50_to_100ms");
    }

    // See header, interpretation 2: AP and ML must agree on when their samples were taken.
    aligned =
      ap.sampleTimesMs.length === ml.sampleTimesMs.length &&
      ap.sampleTimesMs.every((t, i) => t === ml.sampleTimesMs[i]);
    if (!aligned) reasons.push("acceleration_channels_misaligned");
  }

  // Orientation. Card: "differs by more than 20 degrees" invalid; "10-20 degrees" quality.
  const orientation: unknown = trial.initialOrientationDeviationDegrees;
  const orientationOk =
    typeof orientation === "number" && Number.isFinite(orientation) && orientation >= 0;
  if (!orientationOk) reasons.push("initial_orientation_unestablished");
  else {
    if (orientation > 20) reasons.push("initial_orientation_over_20deg");
    if (orientation >= 10 && orientation <= 20) bandFlags.push("initial_orientation_10_to_20deg");
  }

  // Device shift. Card: "the device shifts by more than 10 degrees during the window" invalid.
  // No quality band is authored for shift, only for initial orientation.
  const shift: unknown = trial.deviceShiftDegrees;
  const shiftOk = typeof shift === "number" && Number.isFinite(shift) && shift >= 0;
  if (!shiftOk) reasons.push("device_shift_unestablished");
  else if (shift > 10) reasons.push("device_shift_over_10deg");

  // Event flags. Card: "a step, external support, hands-off-hips balance correction, observer
  // contact, or fall/near-fall occurs". Every reported flag is a claim, not a fact (DEF-017): read
  // as unknown, proven a literal boolean, refuse rather than read falsy.
  const events: readonly { readonly value: unknown; readonly reason: string }[] = [
    { value: trial.stepOccurred, reason: "step_occurred" },
    { value: trial.externalSupportUsed, reason: "external_support_used" },
    { value: trial.handsOffHipsCorrection, reason: "hands_off_hips_correction" },
    { value: trial.observerContact, reason: "observer_contact" },
    { value: trial.fallOrNearFall, reason: "fall_or_near_fall" },
  ];
  if (events.some((e) => typeof e.value !== "boolean")) reasons.push("signals_unestablished");
  for (const event of events) {
    if (event.value === true) reasons.push(event.reason);
  }

  // BFP-BAL-FT-EC-001 only. Card: "more than 1.0 cumulative second or eye-state confirmation is
  // unavailable" invalid; "0.5-1.0 seconds" quality. Null is the card's own explicit observed
  // value ("unavailable"), never a proving failure.
  if (cardId === "BFP-BAL-FT-EC-001") {
    const eyesOpen: unknown = trial.eyesOpenCumulativeSeconds;
    if (eyesOpen === null) {
      reasons.push("eye_state_unconfirmed");
    } else if (typeof eyesOpen !== "number" || !Number.isFinite(eyesOpen) || eyesOpen < 0) {
      reasons.push("eyes_open_unestablished");
    } else {
      if (eyesOpen > 1.0) reasons.push("eyes_open_over_1s");
      if (eyesOpen >= 0.5 && eyesOpen <= 1.0) bandFlags.push("eyes_open_0_5_to_1_0s");
    }
  }

  // BFP-BAL-TAN-EO-001 only. Card: "the wrong lead foot is used, heel-to-toe contact is lost for
  // more than 1.0 second, or foot position cannot be verified" invalid; "lost for up to 1.0 second
  // without a step" quality.
  if (cardId === "BFP-BAL-TAN-EO-001") {
    const loss: unknown = trial.heelToeContactLossSeconds;
    const lossOk = typeof loss === "number" && Number.isFinite(loss) && loss >= 0;
    if (!lossOk) reasons.push("heel_toe_contact_loss_unestablished");
    else {
      if (loss > 1.0) reasons.push("heel_toe_contact_lost_over_1s");
      // Strict: an unestablished step signal must not be read as "definitely no step" (DEF-017),
      // so this reads stepOccurred as unknown again rather than trusting the declared boolean.
      const noStep: unknown = trial.stepOccurred;
      if (loss > 0 && loss <= 1.0 && noStep === false) {
        bandFlags.push("heel_toe_contact_lost_up_to_1s");
      }
    }
    const verified: unknown = trial.footPositionVerified;
    if (typeof verified !== "boolean") reasons.push("foot_position_verification_unestablished");
    else if (!verified) reasons.push("foot_position_unverifiable");
    const leadFoot: unknown = trial.leadFoot;
    if (leadFoot !== "correct" && leadFoot !== "incorrect") reasons.push("lead_foot_unestablished");
    else if (leadFoot === "incorrect") reasons.push("wrong_lead_foot");
  }

  if (reasons.length > 0) {
    return { invalidTrialReasons: reasons, bandFlags, metrics: null };
  }

  // Every guard above passed, so the channels are established and aligned, and duration is a
  // finite number: safe to compute every metric now.
  const ap = apUnknown as UsableChannel;
  const ml = mlUnknown as UsableChannel;
  const av = avUnknown as UsableChannel;

  let resultantRms: number | null = null;
  if (aligned) {
    // resultantOk can only become false if ap.values and ml.values disagree in length, which
    // cannot happen here: isUsableChannel proves each channel's own values array is DENSE (no
    // holes; see isDenseFiniteNumberArray) and equal in length to its own sampleTimesMs, and
    // `aligned` proves the two sampleTimesMs arrays are the same length and elementwise equal. An
    // independent security review confirmed this reasoning FAILED before isUsableChannel proved
    // density: a sparse ap.values (a hole `Array.prototype.every` silently skips) let `aligned`
    // pass while ap.values[i] was undefined for a real i, so this is an ACCEPTED EQUIVALENT
    // MUTATION only as of the density fix above, not by construction. Kept as a per-element check
    // rather than a non-null assertion because noUncheckedIndexedAccess still requires it, and a
    // non-null assertion would remove the type checker's own backstop against a future regression
    // in either proof.
    const resultantValues: number[] = [];
    let resultantOk = true;
    for (let i = 0; i < ap.values.length; i++) {
      const a = ap.values[i];
      const m = ml.values[i];
      if (a === undefined || m === undefined) {
        resultantOk = false;
        break;
      }
      resultantValues.push(Math.sqrt(a * a + m * m));
    }
    resultantRms = resultantOk ? rmsOrNull(resultantValues) : null;
  }

  const metrics: TrialMetrics = {
    apAccelerationRms: rmsOrNull(ap.values),
    mlAccelerationRms: rmsOrNull(ml.values),
    resultantAccelerationRms: resultantRms,
    apJerkRms: rmsOrNull(jerkSeries(ap.values, ap.sampleTimesMs)),
    mlJerkRms: rmsOrNull(jerkSeries(ml.values, ml.sampleTimesMs)),
    angularVelocityRms: rmsOrNull(av.values),
    dominantFrequencyAp: dominantFrequencyHz(ap.values, ap.sampleTimesMs),
    dominantFrequencyMl: dominantFrequencyHz(ml.values, ml.sampleTimesMs),
    completedDurationSeconds: durationSeconds,
  };
  return { invalidTrialReasons: [], bandFlags, metrics };
}

/**
 * Symmetric percentage difference (see header, interpretation 9). Equal values never "differ",
 * which also sidesteps 0/0: an RMS value is never negative, so mean=0 implies both are 0.
 *
 * ACCEPTED EQUIVALENT MUTATION (seeded and reviewed): deleting the `a === b` guard is
 * unobservable through the only caller, which compares this function's result against
 * TRIAL_VARIABILITY_OVER_PERCENT with `>`. Without the guard, a===b===0 returns NaN instead of 0,
 * and `NaN > 30` and `0 > 30` are both false, so the flag is absent either way. The guard is kept
 * anyway: a silent NaN is a worse internal state than 0 regardless of whether today's one caller
 * happens not to expose it, and a future caller that reports this percentage directly (rather
 * than only thresholding it) would otherwise inherit a live NaN-fabrication bug.
 */
function percentDifference(a: number, b: number): number {
  if (a === b) return 0;
  return (Math.abs(a - b) / ((a + b) / 2)) * 100;
}

export function evaluateBalanceSession(session: BalanceSession): BalanceResult {
  const sessionObject: unknown = session;
  if (
    typeof sessionObject !== "object" ||
    sessionObject === null ||
    !Array.isArray((sessionObject as { trials?: unknown }).trials)
  ) {
    return {
      valid: false,
      invalidReasons: ["session_unestablished"],
      qualityFlags: [],
      metrics: null,
    };
  }
  // The card id selects the rule variant; anything else means the recording cannot be trusted.
  const cardId: unknown = session.cardId;
  if (
    cardId !== "BFP-BAL-FT-EO-001" &&
    cardId !== "BFP-BAL-FT-EC-001" &&
    cardId !== "BFP-BAL-TAN-EO-001"
  ) {
    return {
      valid: false,
      invalidReasons: ["session_unestablished"],
      qualityFlags: [],
      metrics: null,
    };
  }
  // Card: "repetitions: One ... familiarization; two recorded trials." A truncated or padded
  // transport is visible here rather than silently scored against the wrong trial count.
  if (session.trials.length !== RECORDED_TRIALS_PER_SESSION) {
    return {
      valid: false,
      invalidReasons: ["session_unestablished"],
      qualityFlags: [],
      metrics: null,
    };
  }
  const trial1Raw = session.trials[0];
  const trial2Raw = session.trials[1];
  if (trial1Raw === undefined || trial2Raw === undefined) {
    return {
      valid: false,
      invalidReasons: ["session_unestablished"],
      qualityFlags: [],
      metrics: null,
    };
  }
  const trial1Unknown: unknown = trial1Raw;
  const trial2Unknown: unknown = trial2Raw;
  if (
    typeof trial1Unknown !== "object" ||
    trial1Unknown === null ||
    typeof trial2Unknown !== "object" ||
    trial2Unknown === null
  ) {
    return {
      valid: false,
      invalidReasons: ["session_unestablished"],
      qualityFlags: [],
      metrics: null,
    };
  }

  const outcome1 = evaluateTrial(trial1Raw, cardId);
  const outcome2 = evaluateTrial(trial2Raw, cardId);

  const qualityFlags: string[] = [];
  for (const band of outcome1.bandFlags) qualityFlags.push(`${band}:trial1`);
  for (const reason of outcome1.invalidTrialReasons)
    qualityFlags.push(`trial_invalid:trial1:${reason}`);
  for (const band of outcome2.bandFlags) qualityFlags.push(`${band}:trial2`);
  for (const reason of outcome2.invalidTrialReasons)
    qualityFlags.push(`trial_invalid:trial2:${reason}`);

  const t1 = outcome1.metrics;
  const t2 = outcome2.metrics;
  if (t1 === null && t2 === null) {
    return {
      valid: false,
      invalidReasons: ["no_valid_trial:trial1", "no_valid_trial:trial2"],
      qualityFlags,
      metrics: null,
    };
  }
  if (t1 === null) qualityFlags.push("no_valid_trial:trial1");
  if (t2 === null) qualityFlags.push("no_valid_trial:trial2");

  // Cross-trial reliability (see header, interpretations 8 and 9). Only computable when BOTH
  // trials are valid; an invalid trial contributes no comparator value, never a fabricated one.
  if (t1 !== null && t2 !== null) {
    for (const key of Object.keys(RMS_METRIC_IDS) as RmsMetricKey[]) {
      const v1 = t1[key];
      const v2 = t2[key];
      if (v1 === null || v2 === null) continue;
      if (percentDifference(v1, v2) > TRIAL_VARIABILITY_OVER_PERCENT) {
        qualityFlags.push(`trial_variability_over_30pct:${RMS_METRIC_IDS[key]}`);
      }
    }
  }

  // Metric ids are the cards' derived_metrics ids, verbatim. Every metric "shows both trials"
  // rather than collapsing them, per the card's own "do not suppress the variability".
  const metrics: Record<string, unknown> = {
    ap_acceleration_rms: {
      trial1: t1 === null ? null : t1.apAccelerationRms,
      trial2: t2 === null ? null : t2.apAccelerationRms,
    },
    ml_acceleration_rms: {
      trial1: t1 === null ? null : t1.mlAccelerationRms,
      trial2: t2 === null ? null : t2.mlAccelerationRms,
    },
    resultant_acceleration_rms: {
      trial1: t1 === null ? null : t1.resultantAccelerationRms,
      trial2: t2 === null ? null : t2.resultantAccelerationRms,
    },
    ap_jerk_rms: {
      trial1: t1 === null ? null : t1.apJerkRms,
      trial2: t2 === null ? null : t2.apJerkRms,
    },
    ml_jerk_rms: {
      trial1: t1 === null ? null : t1.mlJerkRms,
      trial2: t2 === null ? null : t2.mlJerkRms,
    },
    angular_velocity_rms: {
      trial1: t1 === null ? null : t1.angularVelocityRms,
      trial2: t2 === null ? null : t2.angularVelocityRms,
    },
    dominant_frequency_by_axis: {
      trial1: t1 === null ? null : { ap: t1.dominantFrequencyAp, ml: t1.dominantFrequencyMl },
      trial2: t2 === null ? null : { ap: t2.dominantFrequencyAp, ml: t2.dominantFrequencyMl },
    },
    completed_duration_seconds: {
      trial1: t1 === null ? null : t1.completedDurationSeconds,
      trial2: t2 === null ? null : t2.completedDurationSeconds,
    },
    // reportable:false, no authored formula anywhere: permanently null (see header; matches
    // composite_inhibition_score in gng.ts).
    center_of_pressure_metric: null,
  };

  return { valid: true, invalidReasons: [], qualityFlags, metrics };
}
