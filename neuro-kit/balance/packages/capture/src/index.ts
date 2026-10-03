/**
 * The capture trust boundary: the one place device-reported data is proven, before any
 * measurement module sees it.
 *
 * This exists because of DEF-017. Five measurement modules each grew their own inline proving of
 * the same wire shape, three of them fail-open in the same way, and the fix was applied five
 * times. A boundary is the structural version of that fix: prove once, hand downstream a value
 * that is already true.
 *
 * The contract, in three rules:
 *
 *  1. ABSENT is transport corruption and refuses. An explicit null is an observed value and is
 *     accepted only where the wire shape defines one (today: `incompleteness`). A serializer that
 *     omits nil optionals therefore breaks the contract loudly instead of silently reporting a
 *     clean capture -- which is exactly the bug DEF-017 item 3 found in Go/No-Go.
 *  2. Prove the CLONE, not the caller's object. `structuredClone` drops inherited, non-enumerable
 *     and getter-backed properties (DEF-013 finding 4) and makes prove-then-use immune to a value
 *     that changes between reads.
 *  3. Refuse, never throw. A boundary that throws turns into "no validation ran" inside a
 *     caller's try/catch, which is the fail-open shape (DEF-017 item 4).
 *
 * Nothing here computes a derived metric or reads a clinical rule. It reports what was captured,
 * plus the three arithmetic facts every card thresholds on (sample count, completeness, largest
 * gap) so five modules stop recomputing them five ways.
 */

/** The only record kind a device may produce. A client never computes a derived metric. */
export const CAPTURE_KIND = "raw_measurement" as const;

/** Mirrors `CaptureIncompleteness` in PlatformKit. An unknown reason refuses. */
export const INCOMPLETENESS_REASONS = [
  "interrupted",
  "permission_revoked",
  "sensor_unavailable",
  "stopped_by_safety_rule",
  "abandoned_by_patient",
] as const;
export type IncompletenessReason = (typeof INCOMPLETENESS_REASONS)[number];

export interface ProvenChannel {
  readonly name: string;
  readonly units: string;
  readonly coordinateFrame: string;
  readonly requestedRateHz: number;
  readonly sampleTimesMs: readonly number[];
  readonly values: readonly number[];
  /** Observed samples. The card's completeness rule thresholds on this against the expectation. */
  readonly sampleCount: number;
  /**
   * True when the channel carries discrete EVENTS rather than a sampled stream, signalled by a
   * requested rate of exactly zero. Events have no sampling rate, so completeness is not a fact
   * about them.
   */
  readonly isEventDriven: boolean;
  /** Window duration x requested rate, rounded. Null for an event channel or a degenerate window. */
  readonly expectedSampleCount: number | null;
  /**
   * sampleCount / expectedSampleCount as a percent. NULL when expectation is unknowable, never a
   * fabricated 0 or 100: "we could not tell" is a different statement from "nothing arrived".
   */
  readonly completenessPercent: number | null;
  /** Largest interval between consecutive samples; null when there is no interval to measure. */
  readonly largestGapMs: number | null;
}

export interface ProvenAttempt {
  readonly attemptId: string;
  readonly protocolCardId: string;
  readonly protocolCardVersion: number;
  readonly kind: typeof CAPTURE_KIND;
  readonly startedAtMs: number;
  readonly endedAtMs: number;
  readonly channels: readonly ProvenChannel[];
  readonly deviceModel: string;
  readonly osVersion: string;
  readonly appVersion: string;
  /** Null means the device observed nothing wrong. Absent on the wire refuses. */
  readonly incompleteness: IncompletenessReason | null;
}

export type ProveResult =
  | { readonly refused: false; readonly attempt: ProvenAttempt }
  | { readonly refused: true; readonly reasons: readonly string[] };

export const REQUIRED_ATTEMPT_FIELDS = [
  "attemptId",
  "protocolCardId",
  "protocolCardVersion",
  "kind",
  "startedAtMs",
  "endedAtMs",
  "channels",
  "deviceModel",
  "osVersion",
  "appVersion",
  "incompleteness",
] as const;

export const REQUIRED_CHANNEL_FIELDS = [
  "name",
  "units",
  "coordinateFrame",
  "requestedRateHz",
  "observedSampleTimesMs",
  "values",
] as const;

const STRING_ATTEMPT_FIELDS = [
  "attemptId",
  "protocolCardId",
  "deviceModel",
  "osVersion",
  "appVersion",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Own-property presence only: a value reached through the prototype is not a reported field.
 *
 * HONEST REDUNDANCY, recorded rather than counted as coverage. Rule 2 clones first, and
 * `structuredClone` always returns a plain object, so replacing this with `field in value` is an
 * EQUIVALENT mutation today: the only keys `in` would additionally find live on Object.prototype,
 * and no field name collides with one. It is kept because it is correct by construction rather
 * than by luck, and because it stops being equivalent the moment either precondition changes --
 * so the field lists are pinned against Object.prototype by their own test, which fails loudly if
 * a future field is ever named `constructor`, `toString`, or similar.
 */
function hasOwn(value: Record<string, unknown>, field: string): boolean {
  return Object.hasOwn(value, field);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function proveChannel(input: unknown, index: number, reasons: string[]): ProvenChannel | null {
  if (!isRecord(input)) {
    reasons.push(`channel_unestablished:#${String(index)}`);
    return null;
  }
  const label = isNonEmptyString(input.name) ? input.name : `#${String(index)}`;
  let ok = true;
  for (const field of REQUIRED_CHANNEL_FIELDS) {
    if (!hasOwn(input, field)) {
      reasons.push(`channel_field_absent:${label}:${field}`);
      ok = false;
    }
  }
  if (!ok) return null;

  for (const field of ["name", "units", "coordinateFrame"] as const) {
    if (!isNonEmptyString(input[field])) {
      reasons.push(`channel_field_unestablished:${label}:${field}`);
      ok = false;
    }
  }
  const rate = input.requestedRateHz;
  // Exactly zero is legal and means EVENT-DRIVEN (discrete events, not a sampled stream). A
  // negative or non-finite rate is still nonsense and refuses. Found by review: refusing zero
  // outright made the declared `stimulus_events` channel unrepresentable, which forced the facts
  // it carries to be self-attested instead.
  if (typeof rate !== "number" || !Number.isFinite(rate) || rate < 0) {
    reasons.push(`channel_rate_unestablished:${label}`);
    ok = false;
  }

  const times: unknown = input.observedSampleTimesMs;
  const values: unknown = input.values;
  if (!Array.isArray(times) || !Array.isArray(values)) {
    reasons.push(`channel_stream_unestablished:${label}`);
    return null;
  }
  if (times.length !== values.length) {
    // The iOS `isConsistent` property advises this; here it refuses. A channel that cannot
    // account for its own gaps is not evidence of anything.
    reasons.push(`channel_times_values_length_mismatch:${label}`);
    ok = false;
  }
  // `for...of`, NOT `.every`: Array.prototype.every SKIPS holes in a sparse array rather than
  // failing on them, so `[1,2,,4].every(Number.isFinite)` is true while index 2 carries nothing.
  // sampleCount would then count a sample that does not exist and every downstream mean or RMS
  // would divide by a length it has no data for. `for...of` visits holes and reads them as
  // undefined.
  const allValuesFinite = (list: readonly unknown[]): boolean => {
    for (const v of list) {
      if (typeof v !== "number" || !Number.isFinite(v)) return false;
    }
    return true;
  };
  if (!allValuesFinite(values)) {
    reasons.push(`channel_value_unestablished:${label}`);
    ok = false;
  }
  if (!allValuesFinite(times)) {
    reasons.push(`channel_time_unestablished:${label}`);
    ok = false;
  } else {
    let previous: number | undefined;
    for (const time of times as number[]) {
      if (previous !== undefined && time < previous) {
        reasons.push(`channel_times_nonmonotonic:${label}`);
        ok = false;
        break;
      }
      previous = time;
    }
  }
  if (!ok) return null;

  const timeList = times as number[];
  const sampleCount = timeList.length;
  let largestGapMs: number | null = null;
  let previousTime: number | undefined;
  for (const time of timeList) {
    if (previousTime !== undefined) {
      const gap = time - previousTime;
      if (largestGapMs === null || gap > largestGapMs) largestGapMs = gap;
    }
    previousTime = time;
  }
  return {
    name: input.name as string,
    units: input.units as string,
    coordinateFrame: input.coordinateFrame as string,
    requestedRateHz: rate as number,
    sampleTimesMs: timeList,
    values: values as number[],
    sampleCount,
    isEventDriven: rate === 0,
    // expectedSampleCount is filled by the caller, which alone knows the attempt window.
    expectedSampleCount: null,
    completenessPercent: null,
    largestGapMs,
  };
}

/**
 * Proves a device-reported capture attempt.
 *
 * Returns every reason it found, not just the first: a device team fixing a payload should see
 * the whole list in one round trip.
 */
export function proveAttempt(input: unknown): ProveResult {
  const reasons: string[] = [];
  let clone: unknown;
  try {
    // Clone FIRST. Everything proven below is proven about this copy, which is what the caller
    // gets back, so a getter cannot answer differently on the second read.
    clone = structuredClone(input);
  } catch {
    // structuredClone throws on functions, symbols, and class instances it cannot carry --
    // exactly the shapes that have no business crossing a wire boundary.
    return { refused: true, reasons: ["attempt_unreadable"] };
  }

  if (!isRecord(clone)) return { refused: true, reasons: ["attempt_unestablished"] };

  let missing = false;
  for (const field of REQUIRED_ATTEMPT_FIELDS) {
    if (!hasOwn(clone, field)) {
      reasons.push(`field_absent:${field}`);
      missing = true;
    }
  }
  if (missing) return { refused: true, reasons };

  for (const field of STRING_ATTEMPT_FIELDS) {
    if (!isNonEmptyString(clone[field])) reasons.push(`field_unestablished:${field}`);
  }
  if (clone.kind !== CAPTURE_KIND) reasons.push("kind_not_raw_measurement");

  const version = clone.protocolCardVersion;
  if (typeof version !== "number" || !Number.isInteger(version) || version < 0) {
    reasons.push("card_version_unestablished");
  }

  const startedAtMs = clone.startedAtMs;
  const endedAtMs = clone.endedAtMs;
  const startOk = typeof startedAtMs === "number" && Number.isFinite(startedAtMs);
  const endOk = typeof endedAtMs === "number" && Number.isFinite(endedAtMs);
  if (!startOk) reasons.push("field_unestablished:startedAtMs");
  if (!endOk) reasons.push("field_unestablished:endedAtMs");
  if (startOk && endOk && endedAtMs < startedAtMs) reasons.push("attempt_ends_before_it_starts");

  const incompleteness = clone.incompleteness;
  if (incompleteness !== null) {
    if (
      typeof incompleteness !== "string" ||
      !(INCOMPLETENESS_REASONS as readonly string[]).includes(incompleteness)
    ) {
      reasons.push("incompleteness_unknown");
    }
  }

  const rawChannels = clone.channels;
  if (!Array.isArray(rawChannels)) {
    reasons.push("channels_unestablished");
    return { refused: true, reasons };
  }
  const channels: ProvenChannel[] = [];
  const seen = new Set<string>();
  rawChannels.forEach((raw, index) => {
    const channel = proveChannel(raw, index, reasons);
    if (channel === null) return;
    if (seen.has(channel.name)) {
      // A duplicate name means a consumer looking a channel up gets an arbitrary answer.
      reasons.push(`channel_name_duplicated:${channel.name}`);
      return;
    }
    seen.add(channel.name);
    channels.push(channel);
  });

  if (reasons.length > 0) return { refused: true, reasons };

  // Only now, with the window proven, can expectation be stated. A degenerate window expects
  // nothing, and says so with null rather than a fabricated count.
  const windowMs = (endedAtMs as number) - (startedAtMs as number);
  const finished: ProvenChannel[] = channels.map((channel) => {
    // An event channel has no expectation to compare against, and neither does a degenerate
    // window. Both report null rather than a number that would read as a measurement.
    const expected =
      !channel.isEventDriven && windowMs > 0
        ? Math.round((windowMs / 1_000) * channel.requestedRateHz)
        : null;
    const completenessPercent =
      expected !== null && expected > 0 ? (channel.sampleCount / expected) * 100 : null;
    return Object.freeze({
      ...channel,
      sampleTimesMs: Object.freeze(channel.sampleTimesMs),
      values: Object.freeze(channel.values),
      expectedSampleCount: expected,
      completenessPercent,
    });
  });

  return {
    refused: false,
    attempt: Object.freeze({
      attemptId: clone.attemptId as string,
      protocolCardId: clone.protocolCardId as string,
      protocolCardVersion: version as number,
      kind: CAPTURE_KIND,
      startedAtMs: startedAtMs as number,
      endedAtMs: endedAtMs as number,
      channels: Object.freeze(finished),
      deviceModel: clone.deviceModel as string,
      osVersion: clone.osVersion as string,
      appVersion: clone.appVersion as string,
      incompleteness: incompleteness as IncompletenessReason | null,
    }),
  };
}

/** Looks a proven channel up by name. Null when absent: the caller decides what that means. */
export function findChannel(attempt: ProvenAttempt, name: string): ProvenChannel | null {
  return attempt.channels.find((channel) => channel.name === name) ?? null;
}
