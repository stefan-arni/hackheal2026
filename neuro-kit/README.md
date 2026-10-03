# Neuro Kit — speech + balance measurement modules (hackathon extract)

Extracted from a larger private project for a hackathon. **Not a medical device.** Outputs are
observational measurements for research/demo use only; they do not diagnose concussion or clear
anyone to return to play. Use synthetic or consenting-volunteer data only; no real patient data.

## TL;DR: what to use for a live telemedicine demo

| Piece | Path | Runs standalone? |
|---|---|---|
| **Voice features (start here)** | `speech/app/frontend/src/lib/voice/acousticFeatures.ts` | ✅ Yes, zero dependencies |
| **Mic capture** | `speech/app/frontend/src/lib/voice/recorder.ts` | ✅ Yes, browser only (getUserMedia + AudioWorklet) |
| Reading passages | `speech/app/frontend/src/lib/voice/promptPool.ts` | ✅ Yes |
| **Balance evaluator** | `balance/packages/measurement/src/balance.ts` | ✅ With `balance/packages/capture/src/index.ts` (included) |
| Personal-baseline scoring | `speech/app/api/clinical/voice.py` | ✅ Pure Python functions, but see caveat below |
| Backend API routes, DB migrations, AI symptom extraction | `speech/app/api/...` | ❌ Reference only, depend on the original app's auth/DB/billing/AI layers |
| Native speech-to-text plugins | `speech/app/frontend/*-template/`, `lib/native/` | ⚠️ Need Capacitor (mobile app framework) |

## Speech: how to use it

```ts
import { record } from "./recorder";
import { extractFeatures } from "./acousticFeatures";

const { pcm, sampleRate } = await record(15);        // record up to 15 s from the mic
const features = extractFeatures(pcm, sampleRate);    // 12 numbers, always finite
```

`extractFeatures(pcm: Float32Array, sampleRate: number)` returns:
`speech_rate`, `articulation_rate`, `pause_count`, `pause_ratio`, `mean_pause_ms`,
`voice_onset_ms`, `f0_mean_hz`, `f0_sd_hz`, `jitter_pct`, `shimmer_pct`, `snr_db`, `voiced_seconds`.

Design notes:
- Audio stays on the device; only the 12 numbers need to leave it.
- 25 ms frames / 10 ms hop, energy-based voice activity detection, autocorrelation pitch (70–400 Hz).
- Call audio over a video call is compressed and noisy. Check `snr_db` and `voiced_seconds`
  before trusting jitter/shimmer; pause and rate features are the most robust over telehealth.

**Baseline caveat:** `voice.py` scores a session against the person's *own* prior sessions and needs
**5 usable sessions** before it produces a real score (until then it returns a placeholder of 75).
For a one-shot live demo, show the raw features, not the composite score.

## Balance: how to use it

`evaluateBalanceSession({ cardId, trials })` → `{ valid, invalidReasons, qualityFlags, metrics }`

- `cardId`: `"BFP-BAL-FT-EO-001"` (feet together, eyes open), `"BFP-BAL-FT-EC-001"` (eyes closed),
  or `"BFP-BAL-TAN-EO-001"` (tandem stance).
- `trials`: exactly **2 recorded trials** (a practice trial is done first and not submitted).
- Each trial needs **motion-sensor data** from a body-worn phone/IMU: anterior-posterior and
  medio-lateral acceleration plus angular velocity (already resolved into those axes), and observer
  flags (step taken, support used, fall, etc.). **A webcam alone cannot feed this module.**
- Fix-up needed: `balance.ts` imports `@platform/capture`. Point that at
  `../../capture/src/index.ts` (or add a tsconfig path alias).
- `balance.test.ts` is included for reference but needs two internal packages that are not in this
  kit, so it will not run as-is.

## Not included on purpose

Proprietary scoring weights, credentials, databases, and the rest of the original application.
