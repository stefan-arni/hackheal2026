/**
 * Instrumented standing-balance module (BFP-BAL-FT-EO-001, BFP-BAL-FT-EC-001, BFP-BAL-TAN-EO-001)
 * against each card's golden fixtures, plus the boundary cases fixtures cannot isolate. Same
 * discipline as the SRT/TAP suites: fixtures are the oracle, the digest pin forces a module review
 * when the clinical rules change, and every capture-boundary refusal is proven directly.
 *
 * Fixture channels are expanded into wire-shaped capture payloads and run through the REAL
 * @platform/capture `proveAttempt`, so completeness/gap/sampleCount in every expectation are the
 * boundary's own arithmetic, never a parallel reimplementation in this test.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { cardDigest } from "@platform/protocol-card";
import { CAPTURE_KIND, findChannel, proveAttempt, type ProvenChannel } from "@platform/capture";
import {
  evaluateBalanceSession,
  RECORDED_TRIALS_PER_SESSION,
  TRIAL_VARIABILITY_OVER_PERCENT,
  type BalanceCardId,
  type BalanceSession,
  type BalanceTrial,
} from "../src/balance.js";

function loadCard(id: string): Record<string, unknown> {
  const url = new URL(`../../../protocols/v1-battery/${id}.yaml`, import.meta.url);
  return parse(readFileSync(fileURLToPath(url), "utf8")) as Record<string, unknown>;
}

// --- Channel construction: raw samples -> a real @platform/capture ProvenChannel ------------

let attemptCounter = 0;

function provenChannelFromRaw(
  values: readonly number[],
  timesMs: readonly number[],
  requestedRateHz: number,
  name: string,
  startedAtMs: number,
  endedAtMs: number,
): ProvenChannel {
  attemptCounter += 1;
  const wireAttempt = {
    attemptId: `fixture-attempt-${String(attemptCounter)}`,
    protocolCardId: "BFP-BAL-TEST",
    protocolCardVersion: 1,
    kind: CAPTURE_KIND,
    startedAtMs,
    endedAtMs,
    channels: [
      {
        name,
        units: "unit",
        coordinateFrame: "device",
        requestedRateHz,
        observedSampleTimesMs: timesMs,
        values,
      },
    ],
    deviceModel: "iPhone17,1",
    osVersion: "26.0",
    appVersion: "0.1.0",
    incompleteness: null,
  };
  const result = proveAttempt(wireAttempt);
  if (result.refused) {
    throw new Error(`test fixture channel "${name}" failed to prove: ${result.reasons.join(", ")}`);
  }
  const channel = findChannel(result.attempt, name);
  if (channel === null) throw new Error(`test fixture channel "${name}" not found after proving`);
  return channel;
}

// --- Fixture-file channel encoding: {kind: constant|sine, ...} -> raw values/times -----------

interface ConstantChannelSpec {
  readonly kind: "constant";
  readonly value: number;
  readonly count: number;
  readonly requested_rate_hz: number;
}
interface SineChannelSpec {
  readonly kind: "sine";
  readonly amplitude: number;
  readonly cycles: number;
  readonly count: number;
  readonly requested_rate_hz: number;
}
type ChannelSpec = ConstantChannelSpec | SineChannelSpec;

function expandSpecValues(spec: ChannelSpec): number[] {
  if (spec.kind === "constant") return Array.from({ length: spec.count }, () => spec.value);
  return Array.from(
    { length: spec.count },
    (_, i) => spec.amplitude * Math.sin((2 * Math.PI * spec.cycles * i) / spec.count),
  );
}

function expandSpecTimes(spec: ChannelSpec): number[] {
  const dtMs = 1000 / spec.requested_rate_hz;
  return Array.from({ length: spec.count }, (_, i) => i * dtMs);
}

function provenChannelFromSpec(
  spec: ChannelSpec,
  name: string,
  startedAtMs: number,
  endedAtMs: number,
): ProvenChannel {
  return provenChannelFromRaw(
    expandSpecValues(spec),
    expandSpecTimes(spec),
    spec.requested_rate_hz,
    name,
    startedAtMs,
    endedAtMs,
  );
}

// --- Fixture file shape ------------------------------------------------------------------

interface FixtureTrialInput {
  readonly started_at_ms: number;
  readonly ended_at_ms: number;
  readonly ap_acceleration: ChannelSpec;
  readonly ml_acceleration: ChannelSpec;
  readonly angular_velocity: ChannelSpec;
  readonly initial_orientation_deviation_degrees: number;
  readonly device_shift_degrees: number;
  readonly step_occurred: boolean;
  readonly external_support_used: boolean;
  readonly hands_off_hips_correction: boolean;
  readonly observer_contact: boolean;
  readonly fall_or_near_fall: boolean;
  readonly eyes_open_cumulative_seconds?: number | null;
  readonly heel_toe_contact_loss_seconds?: number;
  readonly foot_position_verified?: boolean;
  readonly lead_foot?: "correct" | "incorrect";
}

interface Fixture {
  readonly fixture_id: string;
  readonly input: { readonly trials: readonly FixtureTrialInput[] };
  readonly expected: {
    readonly valid: boolean;
    readonly invalid_reasons: readonly string[];
    readonly quality_flags: readonly string[];
    readonly metrics: Record<string, unknown> | null;
  };
  readonly tolerance: number;
}

function fixtureTrialToBalanceTrial(input: FixtureTrialInput): BalanceTrial {
  return {
    apAcceleration: provenChannelFromSpec(
      input.ap_acceleration,
      "ap_acceleration",
      input.started_at_ms,
      input.ended_at_ms,
    ),
    mlAcceleration: provenChannelFromSpec(
      input.ml_acceleration,
      "ml_acceleration",
      input.started_at_ms,
      input.ended_at_ms,
    ),
    angularVelocity: provenChannelFromSpec(
      input.angular_velocity,
      "angular_velocity",
      input.started_at_ms,
      input.ended_at_ms,
    ),
    startedAtMs: input.started_at_ms,
    endedAtMs: input.ended_at_ms,
    initialOrientationDeviationDegrees: input.initial_orientation_deviation_degrees,
    deviceShiftDegrees: input.device_shift_degrees,
    stepOccurred: input.step_occurred,
    externalSupportUsed: input.external_support_used,
    handsOffHipsCorrection: input.hands_off_hips_correction,
    observerContact: input.observer_contact,
    fallOrNearFall: input.fall_or_near_fall,
    ...(input.eyes_open_cumulative_seconds !== undefined
      ? { eyesOpenCumulativeSeconds: input.eyes_open_cumulative_seconds }
      : {}),
    ...(input.heel_toe_contact_loss_seconds !== undefined
      ? { heelToeContactLossSeconds: input.heel_toe_contact_loss_seconds }
      : {}),
    ...(input.foot_position_verified !== undefined
      ? { footPositionVerified: input.foot_position_verified }
      : {}),
    ...(input.lead_foot !== undefined ? { leadFoot: input.lead_foot } : {}),
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Recursively compares the metrics object, tolerating floating-point noise at every leaf. */
function assertMetricsMatch(
  actual: unknown,
  expected: unknown,
  tolerance: number,
  path: string,
): void {
  if (expected === null) {
    expect(actual, path).toBeNull();
    return;
  }
  if (typeof expected === "number") {
    expect(typeof actual, path).toBe("number");
    if (tolerance > 0) {
      expect(Math.abs((actual as number) - expected), path).toBeLessThanOrEqual(tolerance);
    } else {
      expect(actual, path).toBe(expected);
    }
    return;
  }
  if (isPlainObject(expected)) {
    expect(isPlainObject(actual), path).toBe(true);
    const actualObj = actual as Record<string, unknown>;
    expect(Object.keys(actualObj).sort(), path).toEqual(Object.keys(expected).sort());
    for (const key of Object.keys(expected)) {
      assertMetricsMatch(actualObj[key], expected[key], tolerance, `${path}.${key}`);
    }
    return;
  }
  expect(actual, path).toEqual(expected);
}

function runFixture(cardId: BalanceCardId, fixture: Fixture): void {
  const session: BalanceSession = {
    cardId,
    trials: fixture.input.trials.map(fixtureTrialToBalanceTrial),
  };
  const result = evaluateBalanceSession(session);
  expect(result.valid, fixture.fixture_id).toBe(fixture.expected.valid);
  expect([...result.invalidReasons].sort(), fixture.fixture_id).toEqual(
    [...fixture.expected.invalid_reasons].sort(),
  );
  expect([...result.qualityFlags].sort(), fixture.fixture_id).toEqual(
    [...fixture.expected.quality_flags].sort(),
  );
  if (fixture.expected.metrics === null) {
    expect(result.metrics, fixture.fixture_id).toBeNull();
    return;
  }
  expect(result.metrics, fixture.fixture_id).not.toBeNull();
  if (result.metrics === null) return;
  assertMetricsMatch(
    result.metrics,
    fixture.expected.metrics,
    fixture.tolerance,
    fixture.fixture_id,
  );
}

describe("BFP-BAL-FT-EO-001 golden fixtures", () => {
  const card = loadCard("BFP-BAL-FT-EO-001");
  const fixtures = card.golden_fixtures as Fixture[];

  it("the card carries the full fixture set", () => {
    expect(fixtures.length).toBe(5);
    expect(new Set(fixtures.map((f) => f.fixture_id)).size).toBe(5);
  });

  for (const fixture of fixtures) {
    it(`fixture ${fixture.fixture_id} computes exactly the expected result`, () => {
      runFixture("BFP-BAL-FT-EO-001", fixture);
    });
  }
});

describe("BFP-BAL-FT-EC-001 golden fixtures", () => {
  const card = loadCard("BFP-BAL-FT-EC-001");
  const fixtures = card.golden_fixtures as Fixture[];

  it("the card carries the full fixture set", () => {
    expect(fixtures.length).toBe(6);
    expect(new Set(fixtures.map((f) => f.fixture_id)).size).toBe(6);
  });

  for (const fixture of fixtures) {
    it(`fixture ${fixture.fixture_id} computes exactly the expected result`, () => {
      runFixture("BFP-BAL-FT-EC-001", fixture);
    });
  }
});

describe("BFP-BAL-TAN-EO-001 golden fixtures", () => {
  const card = loadCard("BFP-BAL-TAN-EO-001");
  const fixtures = card.golden_fixtures as Fixture[];

  it("the card carries the full fixture set", () => {
    expect(fixtures.length).toBe(7);
    expect(new Set(fixtures.map((f) => f.fixture_id)).size).toBe(7);
  });

  for (const fixture of fixtures) {
    it(`fixture ${fixture.fixture_id} computes exactly the expected result`, () => {
      runFixture("BFP-BAL-TAN-EO-001", fixture);
    });
  }
});

// --- Boundary cases: every threshold's exact edge, constructed directly (fixtures cannot
// isolate a single axis as cleanly as a dedicated boundary case can). ---------------------------

const DEFAULT_RATE_HZ = 100;
const DEFAULT_STARTED_MS = 0;
const DEFAULT_ENDED_MS = 20_000;

interface TrialOverrides {
  readonly startedAtMs?: number;
  readonly endedAtMs?: number;
  readonly rateHz?: number;
  readonly count?: number;
  /** Overrides just the AP channel's sample count, leaving ML/angular velocity at `count`; lets a
   * test give one channel worse completeness than its siblings. */
  readonly apCount?: number;
  readonly apValue?: number;
  readonly mlValue?: number;
  readonly avValue?: number;
  readonly apTimes?: readonly number[];
  readonly mlTimes?: readonly number[];
  readonly initialOrientationDeviationDegrees?: number;
  readonly deviceShiftDegrees?: number;
  readonly stepOccurred?: boolean;
  readonly externalSupportUsed?: boolean;
  readonly handsOffHipsCorrection?: boolean;
  readonly observerContact?: boolean;
  readonly fallOrNearFall?: boolean;
  readonly eyesOpenCumulativeSeconds?: number | null;
  readonly heelToeContactLossSeconds?: number;
  readonly footPositionVerified?: boolean;
  readonly leadFoot?: "correct" | "incorrect";
}

/** A trial that passes every invalidity gate unless an override deliberately breaks one axis. */
function cleanTrial(over: TrialOverrides = {}): BalanceTrial {
  const startedAtMs = over.startedAtMs ?? DEFAULT_STARTED_MS;
  const endedAtMs = over.endedAtMs ?? DEFAULT_ENDED_MS;
  const rateHz = over.rateHz ?? DEFAULT_RATE_HZ;
  const count = over.count ?? Math.round(((endedAtMs - startedAtMs) / 1000) * rateHz);
  const apCount = over.apCount ?? count;
  const dtMs = 1000 / rateHz;
  const nominalTimes = Array.from({ length: count }, (_, i) => i * dtMs);
  const apTimesDefault = Array.from({ length: apCount }, (_, i) => i * dtMs);
  const apValues = Array.from({ length: apCount }, () => over.apValue ?? 0.01);
  const mlValues = Array.from({ length: count }, () => over.mlValue ?? 0.01);
  const avValues = Array.from({ length: count }, () => over.avValue ?? 0.02);

  return {
    apAcceleration: provenChannelFromRaw(
      apValues,
      over.apTimes ?? apTimesDefault,
      rateHz,
      "ap",
      startedAtMs,
      endedAtMs,
    ),
    mlAcceleration: provenChannelFromRaw(
      mlValues,
      over.mlTimes ?? nominalTimes,
      rateHz,
      "ml",
      startedAtMs,
      endedAtMs,
    ),
    angularVelocity: provenChannelFromRaw(
      avValues,
      nominalTimes,
      rateHz,
      "av",
      startedAtMs,
      endedAtMs,
    ),
    startedAtMs,
    endedAtMs,
    initialOrientationDeviationDegrees: over.initialOrientationDeviationDegrees ?? 3,
    deviceShiftDegrees: over.deviceShiftDegrees ?? 1,
    stepOccurred: over.stepOccurred ?? false,
    externalSupportUsed: over.externalSupportUsed ?? false,
    handsOffHipsCorrection: over.handsOffHipsCorrection ?? false,
    observerContact: over.observerContact ?? false,
    fallOrNearFall: over.fallOrNearFall ?? false,
    ...(over.eyesOpenCumulativeSeconds !== undefined
      ? { eyesOpenCumulativeSeconds: over.eyesOpenCumulativeSeconds }
      : {}),
    ...(over.heelToeContactLossSeconds !== undefined
      ? { heelToeContactLossSeconds: over.heelToeContactLossSeconds }
      : {}),
    ...(over.footPositionVerified !== undefined
      ? { footPositionVerified: over.footPositionVerified }
      : {}),
    ...(over.leadFoot !== undefined ? { leadFoot: over.leadFoot } : {}),
  };
}

/** A gap of `gapMs` between sample `atIndex` and `atIndex + 1`; every other step is nominal. */
function timesWithOneGap(count: number, dtMs: number, atIndex: number, gapMs: number): number[] {
  const times: number[] = [];
  let t = 0;
  for (let i = 0; i < count; i++) {
    if (i > 0) t += i === atIndex + 1 ? gapMs : dtMs;
    times.push(t);
  }
  return times;
}

function eoSession(trial1: BalanceTrial, trial2: BalanceTrial): BalanceSession {
  return { cardId: "BFP-BAL-FT-EO-001", trials: [trial1, trial2] };
}
function ecSession(trial1: BalanceTrial, trial2: BalanceTrial): BalanceSession {
  return { cardId: "BFP-BAL-FT-EC-001", trials: [trial1, trial2] };
}
function tanSession(trial1: BalanceTrial, trial2: BalanceTrial): BalanceSession {
  return { cardId: "BFP-BAL-TAN-EO-001", trials: [trial1, trial2] };
}

describe("duration boundary: below 18.0 invalid, 18.0-19.9 quality, above clean", () => {
  it("17.9 s is invalid", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ endedAtMs: 17_900 }), cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:usable_duration_below_18s");
  });
  it("exactly 18.0 s is low quality, not invalid", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ endedAtMs: 18_000 }), cleanTrial()));
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:usable_duration_below_18s");
    expect(r.qualityFlags).toContain("usable_duration_18_0_to_19_9:trial1");
  });
  it("exactly 19.9 s is still the low-quality band", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ endedAtMs: 19_900 }), cleanTrial()));
    expect(r.qualityFlags).toContain("usable_duration_18_0_to_19_9:trial1");
  });
  it("20.0 s (the full window) carries no duration flag at all", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial(), cleanTrial()));
    expect(r.qualityFlags.some((f) => f.includes("usable_duration"))).toBe(false);
  });
});

describe("sample completeness boundary: below 90 invalid, 90-94.9 quality, above clean", () => {
  it("89.9% (1798 of 2000) is invalid", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ count: 1798 }), cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:sample_completeness_below_90pct");
  });
  it("exactly 90% (1800 of 2000) is low quality, not invalid", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ count: 1800 }), cleanTrial()));
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:sample_completeness_below_90pct");
    expect(r.qualityFlags).toContain("sample_completeness_90_to_94_9pct:trial1");
  });
  it("94.9% (1898 of 2000) is still the low-quality band", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ count: 1898 }), cleanTrial()));
    expect(r.qualityFlags).toContain("sample_completeness_90_to_94_9pct:trial1");
  });
  it("95% (1900 of 2000) carries no completeness flag at all", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ count: 1900 }), cleanTrial()));
    expect(r.qualityFlags.some((f) => f.includes("sample_completeness"))).toBe(false);
  });
  it("the WORST channel governs: one channel at 80% invalidates even though its siblings are at 100%", () => {
    // ap alone is short (1600 of 2000 = 80%); ml and angular_velocity are fully complete. The
    // card's "sample completeness" is a single per-trial number, and interpretation 6 (module
    // header) takes the minimum across the trial's three channels, not the best one: a single
    // badly-served channel must not be hidden behind two healthy ones.
    const r = evaluateBalanceSession(eoSession(cleanTrial({ apCount: 1600 }), cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:sample_completeness_below_90pct");
  });
});

describe("timestamp gap boundary: over 100 ms invalid, 50-100 ms quality, below clean", () => {
  it("101 ms is invalid", () => {
    const times = timesWithOneGap(2000, 10, 500, 101);
    const r = evaluateBalanceSession(eoSession(cleanTrial({ apTimes: times }), cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:timestamp_gap_over_100ms");
  });
  it("exactly 100 ms is low quality, not invalid", () => {
    const times = timesWithOneGap(2000, 10, 500, 100);
    const r = evaluateBalanceSession(eoSession(cleanTrial({ apTimes: times }), cleanTrial()));
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:timestamp_gap_over_100ms");
    expect(r.qualityFlags).toContain("timestamp_gap_50_to_100ms:trial1");
  });
  it("exactly 50 ms is still the low-quality band", () => {
    const times = timesWithOneGap(2000, 10, 500, 50);
    const r = evaluateBalanceSession(eoSession(cleanTrial({ apTimes: times }), cleanTrial()));
    expect(r.qualityFlags).toContain("timestamp_gap_50_to_100ms:trial1");
  });
  it("49 ms carries no gap flag at all", () => {
    const times = timesWithOneGap(2000, 10, 500, 49);
    const r = evaluateBalanceSession(eoSession(cleanTrial({ apTimes: times }), cleanTrial()));
    expect(r.qualityFlags.some((f) => f.includes("timestamp_gap"))).toBe(false);
  });
});

describe("initial orientation boundary: over 20deg invalid, 10-20deg quality, below clean", () => {
  it("20.1 degrees is invalid", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ initialOrientationDeviationDegrees: 20.1 }), cleanTrial()),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:initial_orientation_over_20deg");
  });
  it("exactly 20 degrees is low quality, not invalid", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ initialOrientationDeviationDegrees: 20 }), cleanTrial()),
    );
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:initial_orientation_over_20deg");
    expect(r.qualityFlags).toContain("initial_orientation_10_to_20deg:trial1");
  });
  it("exactly 10 degrees is still the low-quality band", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ initialOrientationDeviationDegrees: 10 }), cleanTrial()),
    );
    expect(r.qualityFlags).toContain("initial_orientation_10_to_20deg:trial1");
  });
  it("9.9 degrees carries no orientation flag at all", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ initialOrientationDeviationDegrees: 9.9 }), cleanTrial()),
    );
    expect(r.qualityFlags.some((f) => f.includes("orientation"))).toBe(false);
  });
});

describe("device shift boundary: over 10deg invalid, no quality band authored", () => {
  it("10.1 degrees is invalid", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ deviceShiftDegrees: 10.1 }), cleanTrial()),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:device_shift_over_10deg");
  });
  it("exactly 10 degrees is clean: the card's shift threshold is 'more than 10'", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ deviceShiftDegrees: 10 }), cleanTrial()),
    );
    expect(r.qualityFlags.some((f) => f.includes("device_shift"))).toBe(false);
  });
});

describe("cross-trial variability: exceed 30% flags, exactly 30% does not", () => {
  it("exactly 30% (17 vs 23) does not flag: the comparator is strictly greater-than", () => {
    // 17 and 23 are chosen so BOTH the RMS and the percent-difference are bit-exact: RMS of a
    // single constant sample is sqrt(v*v), and 17 and 23 are perfect-square roots, so this hits
    // literally 30.0, not merely "very close" -- which matters, because a nearby-but-inexact value
    // cannot distinguish a strict `>` comparator from a weakened `>=` one. (Confirmed independently:
    // |17-23|/((17+23)/2)*100 === 30 exactly in IEEE 754 double, verified by direct computation
    // before writing this test.) A single sample per channel also sidesteps the floating-point
    // summation noise a 2000-sample RMS would add on top (empirically, the visually-similar
    // 0.85/1.15 pair is NOT bit-exact and lands on different sides of 30 at different sample
    // counts). rate=0.05 Hz over the 20 s window asks for exactly one expected sample
    // (round(20*0.05)=1), so completeness is still 100% and duration is still the full 20 s: a
    // fully valid trial, just not one built from 2000 summed samples.
    const r = evaluateBalanceSession(
      eoSession(
        cleanTrial({ apValue: 17, count: 1, rateHz: 0.05 }),
        cleanTrial({ apValue: 23, count: 1, rateHz: 0.05 }),
      ),
    );
    expect(r.qualityFlags.some((f) => f.startsWith("trial_variability_over_30pct"))).toBe(false);
  });
  it("just over 30% (0.849 vs 1.151) flags ap_acceleration_rms", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ apValue: 0.849 }), cleanTrial({ apValue: 1.151 })),
    );
    expect(r.qualityFlags).toContain("trial_variability_over_30pct:ap_acceleration_rms");
  });
  it("order does not matter: trial2 lower than trial1 also flags", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ apValue: 1.151 }), cleanTrial({ apValue: 0.849 })),
    );
    expect(r.qualityFlags).toContain("trial_variability_over_30pct:ap_acceleration_rms");
  });
  it("identical values across trials never flag, including both exactly zero", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ apValue: 0 }), cleanTrial({ apValue: 0 })),
    );
    expect(r.qualityFlags.some((f) => f.startsWith("trial_variability_over_30pct"))).toBe(false);
  });
  it(`the exported threshold constant is ${String(TRIAL_VARIABILITY_OVER_PERCENT)}`, () => {
    expect(TRIAL_VARIABILITY_OVER_PERCENT).toBe(30);
  });
});

describe("BFP-BAL-FT-EC-001: eyes-open boundary (0.5-1.0 quality, over 1.0 or null invalid)", () => {
  it("1.1 s is invalid", () => {
    const r = evaluateBalanceSession(
      ecSession(
        cleanTrial({ eyesOpenCumulativeSeconds: 1.1 }),
        cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
      ),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:eyes_open_over_1s");
  });
  it("exactly 1.0 s is low quality, not invalid", () => {
    const r = evaluateBalanceSession(
      ecSession(
        cleanTrial({ eyesOpenCumulativeSeconds: 1.0 }),
        cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
      ),
    );
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:eyes_open_over_1s");
    expect(r.qualityFlags).toContain("eyes_open_0_5_to_1_0s:trial1");
  });
  it("exactly 0.5 s is still the low-quality band", () => {
    const r = evaluateBalanceSession(
      ecSession(
        cleanTrial({ eyesOpenCumulativeSeconds: 0.5 }),
        cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
      ),
    );
    expect(r.qualityFlags).toContain("eyes_open_0_5_to_1_0s:trial1");
  });
  it("0.4 s carries no eyes-open flag at all", () => {
    const r = evaluateBalanceSession(
      ecSession(
        cleanTrial({ eyesOpenCumulativeSeconds: 0.4 }),
        cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
      ),
    );
    expect(r.qualityFlags.some((f) => f.includes("eyes_open"))).toBe(false);
  });
  it("null (eye-state confirmation unavailable) is invalid, not a proving failure", () => {
    const r = evaluateBalanceSession(
      ecSession(
        cleanTrial({ eyesOpenCumulativeSeconds: null }),
        cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
      ),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:eye_state_unconfirmed");
  });
});

describe("BFP-BAL-TAN-EO-001: heel-toe contact loss boundary (up to 1.0 s without a step is quality)", () => {
  it("1.1 s is invalid", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          heelToeContactLossSeconds: 1.1,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
      ),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:heel_toe_contact_lost_over_1s");
  });
  it("exactly 1.0 s (no step) is low quality, not invalid", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          heelToeContactLossSeconds: 1.0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
      ),
    );
    expect(r.qualityFlags).not.toContain("trial_invalid:trial1:heel_toe_contact_lost_over_1s");
    expect(r.qualityFlags).toContain("heel_toe_contact_lost_up_to_1s:trial1");
  });
  it("a small loss just above zero (no step) is still the low-quality band", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          heelToeContactLossSeconds: 0.1,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
      ),
    );
    expect(r.qualityFlags).toContain("heel_toe_contact_lost_up_to_1s:trial1");
  });
  it("zero loss carries no heel-toe flag at all", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
      ),
    );
    expect(r.qualityFlags.some((f) => f.includes("heel_toe"))).toBe(false);
  });
  it("the quality band requires 'without a step': a step voids the band flag (and invalidates anyway)", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          heelToeContactLossSeconds: 0.5,
          stepOccurred: true,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
        cleanTrial({
          heelToeContactLossSeconds: 0,
          footPositionVerified: true,
          leadFoot: "correct",
        }),
      ),
    );
    expect(r.qualityFlags).not.toContain("heel_toe_contact_lost_up_to_1s:trial1");
    expect(r.qualityFlags).toContain("trial_invalid:trial1:step_occurred");
  });
  it("the wrong lead foot invalidates regardless of contact loss", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          leadFoot: "incorrect",
          footPositionVerified: true,
          heelToeContactLossSeconds: 0,
        }),
        cleanTrial({
          leadFoot: "correct",
          footPositionVerified: true,
          heelToeContactLossSeconds: 0,
        }),
      ),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:wrong_lead_foot");
  });
  it("an unverifiable foot position invalidates", () => {
    const r = evaluateBalanceSession(
      tanSession(
        cleanTrial({
          footPositionVerified: false,
          leadFoot: "correct",
          heelToeContactLossSeconds: 0,
        }),
        cleanTrial({
          footPositionVerified: true,
          leadFoot: "correct",
          heelToeContactLossSeconds: 0,
        }),
      ),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:foot_position_unverifiable");
  });
});

describe("capture trust boundary: proven, never assumed", () => {
  it("a non-object session refuses instead of throwing", () => {
    for (const junk of [null, undefined, 7, "session", [], true]) {
      const r = evaluateBalanceSession(junk as unknown as BalanceSession);
      expect(r.valid, String(junk)).toBe(false);
      expect(r.invalidReasons).toEqual(["session_unestablished"]);
      expect(r.metrics).toBeNull();
    }
  });

  it("a cardId outside the three known cards refuses", () => {
    const session = {
      cardId: "BFP-BAL-NOT-A-CARD",
      trials: [cleanTrial(), cleanTrial()],
    } as unknown as BalanceSession;
    const r = evaluateBalanceSession(session);
    expect(r.valid).toBe(false);
    expect(r.invalidReasons).toEqual(["session_unestablished"]);
  });

  it(`the trial count must equal exactly ${String(RECORDED_TRIALS_PER_SESSION)}`, () => {
    for (const trials of [[], [cleanTrial()], [cleanTrial(), cleanTrial(), cleanTrial()]]) {
      const r = evaluateBalanceSession({ cardId: "BFP-BAL-FT-EO-001", trials });
      expect(r.valid, String(trials.length)).toBe(false);
      expect(r.invalidReasons).toEqual(["session_unestablished"]);
    }
  });

  it("a null or non-object trial element refuses instead of throwing", () => {
    for (const junk of [null, undefined, 7, "trial"]) {
      const r = evaluateBalanceSession({
        cardId: "BFP-BAL-FT-EO-001",
        trials: [junk as unknown as BalanceTrial, cleanTrial()],
      });
      expect(r.valid, String(junk)).toBe(false);
      expect(r.invalidReasons).toEqual(["session_unestablished"]);
    }
  });

  it("a channel that is not a proven object invalidates just that trial", () => {
    const badTrial = {
      ...cleanTrial(),
      apAcceleration: "not-a-channel",
    } as unknown as BalanceTrial;
    const r = evaluateBalanceSession(eoSession(badTrial, cleanTrial()));
    expect(r.valid).toBe(true);
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  it("a channel with a non-finite value refuses that trial rather than poisoning an RMS", () => {
    const trial = cleanTrial();
    const poisoned = {
      ...trial,
      apAcceleration: { ...trial.apAcceleration, values: [...trial.apAcceleration.values] },
    };
    poisoned.apAcceleration.values[5] = Number.NaN;
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  /**
   * A real hole (a slot with no own property at all), built the way an actual producer would:
   * pre-size the array and assign only the indices that were "received", leaving one genuinely
   * unset. This is deliberately NOT an explicit `undefined` at that index (which own-property
   * checks would still see) and deliberately NOT the `delete` operator on an existing array
   * (readable but an unusual thing to do to real data): it is the shape a native bridge or a
   * fixed-size buffer filled by received index would produce for a dropped sample.
   */
  function withHoleAt(source: readonly number[], holeIndex: number): number[] {
    const sparse: number[] = new Array<number>(source.length);
    for (let i = 0; i < source.length; i++) {
      if (i !== holeIndex) {
        const v = source[i];
        if (v !== undefined) sparse[i] = v;
      }
    }
    return sparse;
  }

  it("a sparse (holey) values array refuses rather than silently diluting the RMS", () => {
    // Array.prototype.every/reduce SILENTLY SKIP holes (confirmed independently: [1,2,3] with a
    // hole added at index 10 still reports every(Number.isFinite) === true), and structuredClone
    // preserves holes rather than closing them, so a naive `.every` proof is bypassable.
    const trial = cleanTrial();
    const values = withHoleAt(trial.apAcceleration.values, 5);
    const poisoned = { ...trial, apAcceleration: { ...trial.apAcceleration, values } };
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  it("a sparse (holey) sampleTimesMs array refuses the same way", () => {
    const trial = cleanTrial();
    const sampleTimesMs = withHoleAt(trial.apAcceleration.sampleTimesMs, 5);
    const poisoned = { ...trial, apAcceleration: { ...trial.apAcceleration, sampleTimesMs } };
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  it("non-monotonic sampleTimesMs on an already-proven channel is re-proven and refused", () => {
    const trial = cleanTrial();
    const sampleTimesMs = [...trial.apAcceleration.sampleTimesMs];
    const swap = sampleTimesMs[10];
    const swapped = sampleTimesMs[11];
    if (swap === undefined || swapped === undefined) throw new Error("fixture too short");
    sampleTimesMs[10] = swapped;
    sampleTimesMs[11] = swap;
    const poisoned = { ...trial, apAcceleration: { ...trial.apAcceleration, sampleTimesMs } };
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  it("EQUAL consecutive sample times remain allowed: a sensor batch, not corruption", () => {
    // Monotonicity is non-decreasing, not strictly increasing (matching @platform/capture's own
    // "two samples sharing a timestamp is a plausible sensor batch" convention): a mutation
    // widening the refusal to `<=` must be caught here, since it would refuse this legitimate
    // reading.
    const trial = cleanTrial();
    const sampleTimesMs = [...trial.apAcceleration.sampleTimesMs];
    const tenth = sampleTimesMs[10];
    if (tenth === undefined) throw new Error("fixture too short");
    sampleTimesMs[11] = tenth;
    const poisoned = { ...trial, apAcceleration: { ...trial.apAcceleration, sampleTimesMs } };
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).not.toContain(
      "trial_invalid:trial1:ap_acceleration_channel_unestablished",
    );
  });

  it("all-equal sample times (zero elapsed duration) report a null dominant frequency, not Infinity", () => {
    // A degenerate channel where every sample shares one timestamp passes the monotonicity check
    // (constant is non-decreasing) but has no elapsed time to derive a rate from. Dividing by that
    // zero would fabricate Infinity on a reportable:true metric instead of refusing.
    const n = 2000;
    const constantTimes = Array.from({ length: n }, () => 5000);
    const apValues = Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 0.01 : -0.01));
    const trial: BalanceTrial = {
      apAcceleration: provenChannelFromRaw(apValues, constantTimes, 100, "ap", 0, 20_000),
      mlAcceleration: provenChannelFromRaw(
        Array.from({ length: n }, () => 0.01),
        constantTimes,
        100,
        "ml",
        0,
        20_000,
      ),
      angularVelocity: provenChannelFromRaw(
        Array.from({ length: n }, () => 0.02),
        Array.from({ length: n }, (_, i) => i * 10),
        100,
        "av",
        0,
        20_000,
      ),
      startedAtMs: 0,
      endedAtMs: 20_000,
      initialOrientationDeviationDegrees: 3,
      deviceShiftDegrees: 1,
      stepOccurred: false,
      externalSupportUsed: false,
      handsOffHipsCorrection: false,
      observerContact: false,
      fallOrNearFall: false,
    };
    const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
    expect(r.valid).toBe(true);
    if (r.metrics === null) throw new Error("expected metrics");
    const dominant = r.metrics.dominant_frequency_by_axis as { trial1: { ap: unknown } | null };
    if (dominant.trial1 === null) throw new Error("expected trial1 dominant frequency object");
    expect(dominant.trial1.ap).toBeNull();
  });

  it("a negative largestGapMs refuses rather than being treated as a valid (impossible) gap", () => {
    const trial = cleanTrial();
    const poisoned = { ...trial, apAcceleration: { ...trial.apAcceleration, largestGapMs: -5 } };
    const r = evaluateBalanceSession(eoSession(poisoned, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:ap_acceleration_channel_unestablished");
  });

  it("mismatched AP/ML sample times invalidate the trial rather than pairing wrong instants", () => {
    const shiftedTimes = Array.from({ length: 2000 }, (_, i) => i * 10 + 1);
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ mlTimes: shiftedTimes }), cleanTrial()),
    );
    expect(r.qualityFlags).toContain("trial_invalid:trial1:acceleration_channels_misaligned");
  });

  it("every event flag must be a literal boolean: junk refuses the trial, never reads as falsy", () => {
    for (const junk of [0, 1, "true", "false", null, undefined]) {
      for (const field of [
        "stepOccurred",
        "externalSupportUsed",
        "handsOffHipsCorrection",
        "observerContact",
        "fallOrNearFall",
      ] as const) {
        const trial = { ...cleanTrial(), [field]: junk };
        const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
        expect(r.qualityFlags, `${field}=${String(junk)}`).toContain(
          "trial_invalid:trial1:signals_unestablished",
        );
      }
    }
  });

  it("a genuinely true event flag is still reported alongside an unestablished sibling", () => {
    const trial = {
      ...cleanTrial({ fallOrNearFall: true }),
      stepOccurred: "junk",
    } as unknown as BalanceTrial;
    const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
    expect(r.qualityFlags).toContain("trial_invalid:trial1:signals_unestablished");
    expect(r.qualityFlags).toContain("trial_invalid:trial1:fall_or_near_fall");
  });

  it("orientation and shift must be finite non-negative numbers, not merely truthy", () => {
    for (const junk of [NaN, Infinity, -1, "3", null, undefined]) {
      const r1 = evaluateBalanceSession(
        eoSession(
          { ...cleanTrial(), initialOrientationDeviationDegrees: junk } as unknown as BalanceTrial,
          cleanTrial(),
        ),
      );
      expect(r1.qualityFlags, String(junk)).toContain(
        "trial_invalid:trial1:initial_orientation_unestablished",
      );
      const r2 = evaluateBalanceSession(
        eoSession(
          { ...cleanTrial(), deviceShiftDegrees: junk } as unknown as BalanceTrial,
          cleanTrial(),
        ),
      );
      expect(r2.qualityFlags, String(junk)).toContain(
        "trial_invalid:trial1:device_shift_unestablished",
      );
    }
  });

  it("duration is unestablished when timestamps are junk or the trial ends before it starts", () => {
    for (const bad of [
      { startedAtMs: Number.NaN, endedAtMs: 20_000 },
      { startedAtMs: 0, endedAtMs: Number.POSITIVE_INFINITY },
      { startedAtMs: 5000, endedAtMs: 4000 },
    ]) {
      const trial = { ...cleanTrial(), ...bad } as unknown as BalanceTrial;
      const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
      expect(r.qualityFlags, JSON.stringify(bad)).toContain(
        "trial_invalid:trial1:duration_unestablished",
      );
    }
  });

  it("BFP-BAL-FT-EC-001: a non-number, non-null eyes-open value is unestablished, not falsy", () => {
    for (const junk of ["0.5", true, {}]) {
      const r = evaluateBalanceSession(
        ecSession(
          { ...cleanTrial(), eyesOpenCumulativeSeconds: junk } as unknown as BalanceTrial,
          cleanTrial({ eyesOpenCumulativeSeconds: 0 }),
        ),
      );
      expect(r.qualityFlags, JSON.stringify(junk)).toContain(
        "trial_invalid:trial1:eyes_open_unestablished",
      );
    }
  });

  it("BFP-BAL-TAN-EO-001: leadFoot outside the closed enum is unestablished", () => {
    for (const junk of ["left", "", null, undefined, 1]) {
      const r = evaluateBalanceSession(
        tanSession(
          {
            ...cleanTrial(),
            leadFoot: junk,
            footPositionVerified: true,
            heelToeContactLossSeconds: 0,
          } as unknown as BalanceTrial,
          cleanTrial({ footPositionVerified: true, heelToeContactLossSeconds: 0 }),
        ),
      );
      expect(r.qualityFlags, String(junk)).toContain(
        "trial_invalid:trial1:lead_foot_unestablished",
      );
    }
  });

  it("BFP-BAL-TAN-EO-001: footPositionVerified must be a literal boolean", () => {
    for (const junk of [0, 1, "true", null, undefined]) {
      const r = evaluateBalanceSession(
        tanSession(
          {
            ...cleanTrial(),
            footPositionVerified: junk,
            leadFoot: "correct",
            heelToeContactLossSeconds: 0,
          } as unknown as BalanceTrial,
          cleanTrial({
            footPositionVerified: true,
            leadFoot: "correct",
            heelToeContactLossSeconds: 0,
          }),
        ),
      );
      expect(r.qualityFlags, String(junk)).toContain(
        "trial_invalid:trial1:foot_position_verification_unestablished",
      );
    }
  });
});

describe("both-trials-invalid and single-survivor session outcomes", () => {
  it("both trials invalid makes the session invalid with a reason naming each", () => {
    const r = evaluateBalanceSession(
      eoSession(cleanTrial({ fallOrNearFall: true }), cleanTrial({ stepOccurred: true })),
    );
    expect(r.valid).toBe(false);
    expect([...r.invalidReasons].sort()).toEqual([
      "no_valid_trial:trial1",
      "no_valid_trial:trial2",
    ]);
    expect(r.metrics).toBeNull();
  });

  it("one valid trial keeps the session valid and reports only that trial's metrics", () => {
    const r = evaluateBalanceSession(eoSession(cleanTrial({ fallOrNearFall: true }), cleanTrial()));
    expect(r.valid).toBe(true);
    expect(r.qualityFlags).toContain("no_valid_trial:trial1");
    if (r.metrics === null) throw new Error("expected metrics");
    const durations = r.metrics.completed_duration_seconds as { trial1: unknown; trial2: unknown };
    expect(durations.trial1).toBeNull();
    expect(durations.trial2).toBe(20);
  });
});

describe("dominant frequency: derived from OBSERVED sample spacing, not the declared rate", () => {
  it("a channel whose true density differs from its declared requestedRateHz reports the TRUE frequency", () => {
    // True spacing: dt=10ms (a genuine 100 Hz stream), 2000 samples, a clean 20-cycle sinusoid ->
    // true frequency = 20 * 100 / 2000 = 1.0 Hz. The channel DECLARES requestedRateHz=50 (half the
    // true rate): the pre-fix code multiplied the bin by the declared rate and would have reported
    // 0.5 Hz. Confirms the fix derives the scale from sampleTimesMs, not from the caller's label.
    const n = 2000;
    const trueDtMs = 10;
    const times = Array.from({ length: n }, (_, i) => i * trueDtMs);
    const apValues = Array.from(
      { length: n },
      (_, i) => 0.05 * Math.sin((2 * Math.PI * 20 * i) / n),
    );
    const mlValues = Array.from({ length: n }, () => 0.01);
    const avValues = Array.from({ length: n }, () => 0.02);
    const trial: BalanceTrial = {
      apAcceleration: provenChannelFromRaw(apValues, times, 50, "ap", 0, 20_000),
      mlAcceleration: provenChannelFromRaw(mlValues, times, 50, "ml", 0, 20_000),
      angularVelocity: provenChannelFromRaw(avValues, times, 50, "av", 0, 20_000),
      startedAtMs: 0,
      endedAtMs: 20_000,
      initialOrientationDeviationDegrees: 3,
      deviceShiftDegrees: 1,
      stepOccurred: false,
      externalSupportUsed: false,
      handsOffHipsCorrection: false,
      observerContact: false,
      fallOrNearFall: false,
    };
    const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
    expect(r.valid).toBe(true);
    if (r.metrics === null) throw new Error("expected metrics");
    const dominant = r.metrics.dominant_frequency_by_axis as {
      trial1: { ap: number } | null;
    };
    if (dominant.trial1 === null) throw new Error("expected trial1 dominant frequency");
    expect(dominant.trial1.ap).toBeCloseTo(1.0, 6);
  });
});

describe("the DFT is bounded: an oversized channel reports null rather than blocking the event loop", () => {
  it("a channel over MAX_DFT_SAMPLES (10,000) reports a null dominant frequency, not a hang", () => {
    const n = 10_001;
    const times = Array.from({ length: n }, (_, i) => i * 2); // 2 ms apart, well under any gap flag
    const apValues = Array.from(
      { length: n },
      (_, i) => 0.05 * Math.sin((2 * Math.PI * 40 * i) / n),
    );
    // AP and ML share the SAME times array (required for alignment); ML and angular velocity stay
    // small and simple since only AP's size is under test.
    const trial: BalanceTrial = {
      apAcceleration: provenChannelFromRaw(apValues, times, 100, "ap", 0, 20_000),
      mlAcceleration: provenChannelFromRaw(
        Array.from({ length: n }, () => 0.01),
        times,
        100,
        "ml",
        0,
        20_000,
      ),
      angularVelocity: provenChannelFromRaw(
        Array.from({ length: 2000 }, () => 0.02),
        Array.from({ length: 2000 }, (_, i) => i * 10),
        100,
        "av",
        0,
        20_000,
      ),
      startedAtMs: 0,
      endedAtMs: 20_000,
      initialOrientationDeviationDegrees: 3,
      deviceShiftDegrees: 1,
      stepOccurred: false,
      externalSupportUsed: false,
      handsOffHipsCorrection: false,
      observerContact: false,
      fallOrNearFall: false,
    };
    const startedAt = Date.now();
    const r = evaluateBalanceSession(eoSession(trial, cleanTrial()));
    const elapsedMs = Date.now() - startedAt;
    expect(elapsedMs, "must short-circuit, not run the O(n^2) DFT").toBeLessThan(2000);
    expect(r.valid).toBe(true);
    if (r.metrics === null) throw new Error("expected metrics");
    const dominant = r.metrics.dominant_frequency_by_axis as { trial1: { ap: unknown } | null };
    if (dominant.trial1 === null) throw new Error("expected trial1 dominant frequency object");
    expect(dominant.trial1.ap).toBeNull();
  });
});

describe("module-to-card digest pin", () => {
  // Extends the srt.test.ts/tap.test.ts convention (quality_rules, invalidity_rules,
  // derived_metrics, timing, repetitions, golden_fixtures, instructions_onscreen,
  // reference_comparator) with `positioning`: an independent security review found it is the SOLE
  // authored source of BFP-BAL-TAN-EO-001's "correct" lead foot ("Nondominant foot behind dominant
  // foot..."), which this module implements (`wrong_lead_foot`) but the standard field list does
  // not cover. Pinned on all three cards for one uniform field set across the shared evaluator,
  // matching DEF-015 F4's precedent of extending a pin to whatever a module actually implements.
  function relevantFields(card: Record<string, unknown>): Record<string, unknown> {
    return {
      quality_rules: card.quality_rules,
      invalidity_rules: card.invalidity_rules,
      derived_metrics: card.derived_metrics,
      timing: card.timing,
      repetitions: card.repetitions,
      golden_fixtures: card.golden_fixtures,
      instructions_onscreen: card.instructions_onscreen,
      reference_comparator: card.reference_comparator,
      positioning: card.positioning,
    };
  }

  it("BFP-BAL-FT-EO-001's clinical rules have not changed underneath this module", () => {
    const card = loadCard("BFP-BAL-FT-EO-001");
    expect(cardDigest(relevantFields(card))).toBe(
      "81f5e55dd145b146a5310a796b3743d1373b98e0e33c4272f4d5603c0aba76ad",
    );
  });

  it("BFP-BAL-FT-EC-001's clinical rules have not changed underneath this module", () => {
    const card = loadCard("BFP-BAL-FT-EC-001");
    expect(cardDigest(relevantFields(card))).toBe(
      "b6695331ba5da48feb38e5c8e6fc3b31c93fb5388d6d0024f104ef16d298b92b",
    );
  });

  it("BFP-BAL-TAN-EO-001's clinical rules have not changed underneath this module", () => {
    const card = loadCard("BFP-BAL-TAN-EO-001");
    expect(cardDigest(relevantFields(card))).toBe(
      "e233e06a3081902c9b4e9fb098e5bcc51922a802c5569123772d96a15c8415a0",
    );
  });
});
