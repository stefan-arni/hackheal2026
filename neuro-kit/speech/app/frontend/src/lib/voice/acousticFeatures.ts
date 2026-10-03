// app/frontend/src/lib/voice/acousticFeatures.ts
//
// On-device acoustic feature extraction for the voice biomarker.
//
// Pure, dependency-free DSP over a Float32Array of mono PCM. The raw audio
// NEVER leaves the device — only the 12 numeric features computed here are
// transmitted. Every feature is guaranteed to be a finite real number on
// every input (including pure silence and degenerate frames); any division
// that could produce NaN/Infinity is guarded and falls back to 0.
//
// Pipeline (per the sprint plan):
//   - Frame the signal (25 ms window, 10 ms hop).
//   - VAD: per-frame short-time energy vs a noise-floor threshold.
//   - Pauses: runs of consecutive unvoiced frames inside the voiced span.
//   - Rates: voiced-segment count over voiced / total time (syllable proxy).
//   - Pitch: autocorrelation peak in the 70–400 Hz lag band on voiced frames.
//   - Jitter: period-to-period variation of the glottal period (1/F0).
//   - Shimmer: frame-to-frame variation of the per-frame peak amplitude.

export interface AcousticFeatures {
  speech_rate: number;        // voiced segments / total seconds
  articulation_rate: number;  // voiced segments / voiced seconds
  pause_count: number;        // interior unvoiced runs (>= MIN_PAUSE_FRAMES)
  pause_ratio: number;        // pause time / total time
  mean_pause_ms: number;      // mean interior pause length, ms
  voice_onset_ms: number;     // leading silence before first voiced frame, ms
  f0_mean_hz: number;         // mean fundamental over voiced frames
  f0_sd_hz: number;           // sd of fundamental over voiced frames
  jitter_pct: number;         // mean |ΔT| / mean T * 100, T = 1/F0
  shimmer_pct: number;        // mean |ΔA| / mean A * 100, A = frame peak
  snr_db: number;             // 10·log10(voiced energy / unvoiced energy)
  voiced_seconds: number;     // voiced frames × hop
}

const FRAME_MS = 25;
const HOP_MS = 10;

// Pitch search band: 70–400 Hz is the conventional human-voice F0 range and
// also comfortably brackets the synthetic oracle tones (180 Hz, 220 Hz).
const F0_MIN_HZ = 70;
const F0_MAX_HZ = 400;

// A noise-floor multiplier: a frame counts as voiced when its energy exceeds
// the 10th-percentile (noise-floor) frame energy by this factor. Pure silence
// has uniformly tiny energy, so the floor ≈ the mean and (almost) nothing
// clears the bar — exactly the "low quality on silence" behaviour we want.
const VOICED_ENERGY_FACTOR = 4.0;
// Fraction of the loudest frame's energy used as a fallback voiced gate for
// steady signals (where the noise-floor-relative test degenerates).
const PEAK_ENERGY_FRACTION = 0.25;
// Absolute energy gate so true digital silence never registers as voiced even
// if both relative tests degenerate (floor ≈ 0, peak ≈ 0).
const MIN_VOICED_ENERGY = 1e-6;

// An interior unvoiced run must be at least this many frames to count as a
// deliberate pause (≈ 50 ms) rather than a single-frame voicing dropout.
const MIN_PAUSE_FRAMES = 5;

function zero(x: number): number {
  return Number.isFinite(x) ? x : 0;
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function stddev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / xs.length);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

/**
 * Estimate the fundamental frequency of one frame via autocorrelation.
 * Returns 0 when no usable periodic peak is found in the F0 band.
 */
function frameF0(frame: Float32Array, sampleRate: number): number {
  const n = frame.length;
  if (n < 8) return 0;

  // Remove DC so the autocorrelation reflects periodicity, not offset.
  let dc = 0;
  for (let i = 0; i < n; i++) dc += frame[i];
  dc /= n;

  const minLag = Math.max(1, Math.floor(sampleRate / F0_MAX_HZ));
  const maxLag = Math.min(n - 1, Math.ceil(sampleRate / F0_MIN_HZ));
  if (maxLag <= minLag) return 0;

  // r[0] (zero-lag energy) for normalisation / a periodicity sanity check.
  let r0 = 0;
  for (let i = 0; i < n; i++) {
    const v = frame[i] - dc;
    r0 += v * v;
  }
  if (r0 <= 0) return 0;

  let bestLag = -1;
  let bestVal = -Infinity;
  // Track the autocorrelation immediately around the peak for parabolic
  // sub-sample interpolation (needed to hit the ±10 Hz oracle tolerance).
  let rPrev = 0;
  let rPeak = 0;
  let rNext = 0;

  let rm1 = 0; // r[lag-1] carried across the loop
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = lag; i < n; i++) {
      s += (frame[i] - dc) * (frame[i - lag] - dc);
    }
    if (s > bestVal) {
      bestVal = s;
      bestLag = lag;
      rPrev = rm1;
      rPeak = s;
      rNext = 0; // will be filled on the next iteration
    } else if (lag === bestLag + 1) {
      rNext = s; // the sample just past the current best peak
    }
    rm1 = s;
  }

  if (bestLag < 0) return 0;
  // Require a real periodic peak (≥ 30% of zero-lag energy). Aperiodic noise
  // and silence fall below this and report no pitch.
  if (bestVal < 0.3 * r0) return 0;

  // Parabolic interpolation around the integer peak for sub-sample accuracy.
  let lag = bestLag;
  const denom = rPrev - 2 * rPeak + rNext;
  if (rNext !== 0 && Number.isFinite(denom) && denom !== 0) {
    const delta = (0.5 * (rPrev - rNext)) / denom;
    if (Number.isFinite(delta) && Math.abs(delta) < 1) lag = bestLag + delta;
  }

  if (lag <= 0) return 0;
  const f0 = sampleRate / lag;
  if (!Number.isFinite(f0) || f0 < F0_MIN_HZ || f0 > F0_MAX_HZ) return 0;
  return f0;
}

export function extractFeatures(pcm: Float32Array, sampleRate: number): AcousticFeatures {
  const empty: AcousticFeatures = {
    speech_rate: 0, articulation_rate: 0, pause_count: 0, pause_ratio: 0,
    mean_pause_ms: 0, voice_onset_ms: 0, f0_mean_hz: 0, f0_sd_hz: 0,
    jitter_pct: 0, shimmer_pct: 0, snr_db: 0, voiced_seconds: 0,
  };

  if (!pcm || pcm.length === 0 || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    return empty;
  }

  const frameLen = Math.max(8, Math.round((FRAME_MS / 1000) * sampleRate));
  const hopLen = Math.max(1, Math.round((HOP_MS / 1000) * sampleRate));
  const hopSeconds = hopLen / sampleRate;

  // ---- Frame and compute per-frame energy + peak amplitude. ----
  const frameStarts: number[] = [];
  const energies: number[] = [];
  const peaks: number[] = [];
  for (let start = 0; start + frameLen <= pcm.length; start += hopLen) {
    let e = 0;
    let pk = 0;
    for (let i = 0; i < frameLen; i++) {
      const v = pcm[start + i];
      e += v * v;
      const a = Math.abs(v);
      if (a > pk) pk = a;
    }
    frameStarts.push(start);
    energies.push(e / frameLen); // mean-square energy
    peaks.push(pk);
  }

  const numFrames = frameStarts.length;
  if (numFrames === 0) return empty;

  // ---- VAD: noise floor (10th percentile energy) × factor. ----
  // A steady tone has near-uniform frame energy, so the noise-floor-relative
  // test alone would mark *nothing* voiced (floor ≈ signal). Anchor a second
  // gate to a fraction of the loudest frame's energy: for a tone this clears
  // every frame; for pure silence the peak energy is ~0 so nothing clears.
  const sortedE = [...energies].sort((a, b) => a - b);
  const noiseFloor = percentile(sortedE, 0.10);
  const peakEnergy = sortedE[sortedE.length - 1];
  const relThreshold = noiseFloor * VOICED_ENERGY_FACTOR;
  const peakRelThreshold = peakEnergy * PEAK_ENERGY_FRACTION;
  // Voiced if it clears EITHER the noise-floor gate (handles speech with a
  // quiet background) OR the peak-relative gate (handles steady signals where
  // the floor degenerates). Both sit above the absolute silence guard.
  const threshold = Math.max(
    Math.min(relThreshold, peakRelThreshold),
    MIN_VOICED_ENERGY,
  );

  const voicedMask: boolean[] = energies.map((e) => e > threshold);

  let voicedCount = 0;
  let voicedEnergySum = 0;
  let unvoicedEnergySum = 0;
  let unvoicedCount = 0;
  for (let i = 0; i < numFrames; i++) {
    if (voicedMask[i]) {
      voicedCount++;
      voicedEnergySum += energies[i];
    } else {
      unvoicedCount++;
      unvoicedEnergySum += energies[i];
    }
  }

  const voiced_seconds = zero(voicedCount * hopSeconds);
  const total_seconds = zero(numFrames * hopSeconds);

  // ---- SNR: voiced vs unvoiced mean energy. ----
  let snr_db = 0;
  if (voicedCount > 0 && unvoicedCount > 0) {
    const meanVoiced = voicedEnergySum / voicedCount;
    const meanUnvoiced = unvoicedEnergySum / unvoicedCount;
    if (meanUnvoiced > 0 && meanVoiced > 0) {
      snr_db = zero(10 * Math.log10(meanVoiced / meanUnvoiced));
    }
  }
  // No voiced frames at all (e.g. pure silence) → no usable speech signal.
  // Leave snr_db at 0, which is < the test's 10 dB low-quality bar.

  // ---- Locate the voiced span (first..last voiced frame). ----
  let firstVoiced = -1;
  let lastVoiced = -1;
  for (let i = 0; i < numFrames; i++) {
    if (voicedMask[i]) {
      if (firstVoiced < 0) firstVoiced = i;
      lastVoiced = i;
    }
  }

  // ---- Pauses: interior unvoiced runs within the voiced span. ----
  let pause_count = 0;
  const pauseLengthsFrames: number[] = [];
  if (firstVoiced >= 0 && lastVoiced > firstVoiced) {
    let run = 0;
    for (let i = firstVoiced; i <= lastVoiced; i++) {
      if (!voicedMask[i]) {
        run++;
      } else {
        if (run >= MIN_PAUSE_FRAMES) {
          pause_count++;
          pauseLengthsFrames.push(run);
        }
        run = 0;
      }
    }
    // A trailing run inside the span is impossible (lastVoiced is voiced),
    // so no flush needed here.
  }

  const totalPauseFrames = pauseLengthsFrames.reduce((a, b) => a + b, 0);
  const pause_ratio = numFrames > 0 ? zero((totalPauseFrames / numFrames)) : 0;
  const mean_pause_ms = pauseLengthsFrames.length > 0
    ? zero(mean(pauseLengthsFrames) * hopSeconds * 1000)
    : 0;

  // ---- Voice onset: leading silence before first voiced frame. ----
  const voice_onset_ms = firstVoiced > 0 ? zero(firstVoiced * hopSeconds * 1000) : 0;

  // ---- Rates: count contiguous voiced segments (syllable proxy). ----
  let voicedSegments = 0;
  {
    let prev = false;
    for (let i = 0; i < numFrames; i++) {
      const cur = voicedMask[i];
      if (cur && !prev) voicedSegments++;
      prev = cur;
    }
  }
  const articulation_rate = voiced_seconds > 0 ? zero(voicedSegments / voiced_seconds) : 0;
  const speech_rate = total_seconds > 0 ? zero(voicedSegments / total_seconds) : 0;

  // ---- Pitch over voiced frames. ----
  const f0s: number[] = [];
  const periods: number[] = []; // glottal periods in seconds (1/F0)
  const voicedPeaks: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    if (!voicedMask[i]) continue;
    const start = frameStarts[i];
    const frame = pcm.subarray(start, start + frameLen);
    const f0 = frameF0(frame, sampleRate);
    if (f0 > 0) {
      f0s.push(f0);
      periods.push(1 / f0);
    }
    voicedPeaks.push(peaks[i]);
  }

  const f0_mean_hz = f0s.length > 0 ? zero(mean(f0s)) : 0;
  const f0_sd_hz = f0s.length > 1 ? zero(stddev(f0s)) : 0;

  // ---- Jitter: mean |ΔT| of consecutive periods / mean period × 100. ----
  let jitter_pct = 0;
  if (periods.length >= 2) {
    let absDiffSum = 0;
    for (let i = 1; i < periods.length; i++) {
      absDiffSum += Math.abs(periods[i] - periods[i - 1]);
    }
    const meanAbsDiff = absDiffSum / (periods.length - 1);
    const meanPeriod = mean(periods);
    if (meanPeriod > 0) jitter_pct = zero((meanAbsDiff / meanPeriod) * 100);
  }

  // ---- Shimmer: mean |ΔA| of consecutive voiced-frame peaks / mean A × 100. ----
  let shimmer_pct = 0;
  if (voicedPeaks.length >= 2) {
    let absDiffSum = 0;
    for (let i = 1; i < voicedPeaks.length; i++) {
      absDiffSum += Math.abs(voicedPeaks[i] - voicedPeaks[i - 1]);
    }
    const meanAbsDiff = absDiffSum / (voicedPeaks.length - 1);
    const meanPeak = mean(voicedPeaks);
    if (meanPeak > 0) shimmer_pct = zero((meanAbsDiff / meanPeak) * 100);
  }

  return {
    speech_rate: zero(speech_rate),
    articulation_rate: zero(articulation_rate),
    pause_count: zero(pause_count),
    pause_ratio: zero(pause_ratio),
    mean_pause_ms: zero(mean_pause_ms),
    voice_onset_ms: zero(voice_onset_ms),
    f0_mean_hz: zero(f0_mean_hz),
    f0_sd_hz: zero(f0_sd_hz),
    jitter_pct: zero(jitter_pct),
    shimmer_pct: zero(shimmer_pct),
    snr_db: zero(snr_db),
    voiced_seconds: zero(voiced_seconds),
  };
}
