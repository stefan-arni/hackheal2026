/**
 * Speech recognizer bridge — TypeScript surface for native on-device
 * dictation. Mirrors `healthkit.ts` / `healthConnect.ts` in shape so
 * call sites stay platform-agnostic.
 *
 * On the web (and in dev) `isAvailable()` returns
 * `{ available: false, reason: "web platform — use text form" }` and
 * every method is a no-op that resolves with a sensible default.
 *
 * On iOS the methods are backed by a Swift Capacitor plugin under
 * `ios/App/App/Speech/SpeechRecognitionPlugin.swift`. On Android by
 * a Kotlin plugin under
 * `android/app/src/main/java/com/brainhealth/cowork/SpeechRecognitionPlugin.kt`.
 * See `frontend/MOBILE_BUILD.md § 10` for the integration runbook.
 *
 * Plugin contract:
 *
 *   isAvailable(): Promise<{ available: boolean; reason?: string }>
 *   requestPermission(): Promise<{ granted: boolean; status: string }>
 *   start(): Promise<{ ok: boolean; reason?: string }>
 *   stop(): Promise<{ ok: boolean; reason?: string }>
 *   cancel(): Promise<{ ok: boolean; reason?: string }>
 *
 * Events emitted by the native side (subscribe with `on()`):
 *
 *   "partialResult" -> { text: string; isFinal: false }
 *   "finalResult"   -> { text: string; isFinal: true }
 *   "error"         -> { code: string; message: string }
 *
 * The transcript that arrives via `finalResult` is feed-forward into
 * the existing `voiceLog.extract({ transcript })` flow from the
 * 2026-05-04 sprint (transcript -> structured symptom fields). Audio
 * never leaves the device — this bridge only ever sees text.
 *
 * v1 is English-only because the symptom-extraction prompt
 * (`OBSERVED_SIGN_TAGS` in `app/api/ai/voice_extraction.py`) is
 * English-only. The native plugins both initialize with locale en-US;
 * future localization work will widen this surface.
 */
import { Capacitor } from '@capacitor/core';
import type { PluginListenerHandle } from '@capacitor/core';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SpeechRecognitionAvailability {
  available: boolean;
  /** Human-readable reason populated when `available === false`. */
  reason?: string;
}

export interface SpeechRecognitionPermission {
  granted: boolean;
  /**
   * Stable status string. One of: `authorized`, `microphone_denied`,
   * `speech_denied`, `speech_restricted`, `speech_notDetermined`,
   * `speech_unknown`. Code that needs to branch should switch on
   * `granted` first; `status` is for debugging + telemetry only.
   */
  status: string;
}

export interface SpeechRecognitionOk {
  ok: boolean;
  reason?: string;
}

export interface SpeechPartialResult {
  text: string;
  isFinal: false;
}

export interface SpeechFinalResult {
  text: string;
  isFinal: true;
}

export interface SpeechRecognitionError {
  /**
   * Stable code string. Known values include:
   *   `no_speech` — recognizer fired no-match / timeout
   *   `no_on_device` — offline model unavailable / refused
   *   `permission_denied` — mic permission revoked mid-session
   *   `audio_engine_failed` — audio capture pipeline failure
   *   `audio_session_failed` — iOS-only; AVAudioSession config error
   *   `recognizer_busy` — Android-only; another recognizer is active
   *   `recognition_failed` — catch-all
   */
  code: string;
  message: string;
}

export type SpeechRecognitionEvent =
  | 'partialResult'
  | 'finalResult'
  | 'error';

export type SpeechRecognitionEventData<E extends SpeechRecognitionEvent> =
  E extends 'partialResult' ? SpeechPartialResult :
  E extends 'finalResult' ? SpeechFinalResult :
  E extends 'error' ? SpeechRecognitionError :
  never;

export interface SpeechRecognitionPlugin {
  isAvailable(): Promise<SpeechRecognitionAvailability>;
  requestPermission(): Promise<SpeechRecognitionPermission>;
  start(): Promise<SpeechRecognitionOk>;
  stop(): Promise<SpeechRecognitionOk>;
  cancel(): Promise<SpeechRecognitionOk>;
  addListener(
    eventName: SpeechRecognitionEvent,
    listenerFunc: (data: unknown) => void
  ): Promise<PluginListenerHandle> | PluginListenerHandle;
}

// ---------------------------------------------------------------------------
// Stub used on web + when the native plugin is not registered.
// ---------------------------------------------------------------------------

const WEB_REASON = 'web platform — use text form';

const stub: SpeechRecognitionPlugin = {
  async isAvailable() {
    return { available: false, reason: WEB_REASON };
  },
  async requestPermission() {
    return { granted: false, status: 'unsupported' };
  },
  async start() {
    return { ok: false, reason: WEB_REASON };
  },
  async stop() {
    return { ok: false, reason: WEB_REASON };
  },
  async cancel() {
    return { ok: false, reason: WEB_REASON };
  },
  addListener() {
    // Returns a noop handle so consumers can always call .remove()
    // without a runtime check.
    const handle: PluginListenerHandle = {
      remove: async () => {
        /* noop */
      }
    };
    return handle;
  }
};

// ---------------------------------------------------------------------------
// Native plugin lookup. Mirrors the pattern in healthkit.ts:
//   - SSR/web -> stub
//   - iOS / Android with plugin registered -> the registered plugin
//   - iOS / Android without plugin (cap sync not run) -> stub
// ---------------------------------------------------------------------------

function plugin(): SpeechRecognitionPlugin {
  if (typeof window === 'undefined') return stub;
  const platform = Capacitor.getPlatform();
  if (platform !== 'ios' && platform !== 'android') return stub;
  // The Swift @objc(SpeechRecognitionPlugin) and Kotlin
  // @CapacitorPlugin(name = "SpeechRecognition") both register under
  // the bridge name 'SpeechRecognition'.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const p = (Capacitor as any).Plugins?.SpeechRecognition as
    | SpeechRecognitionPlugin
    | undefined;
  return p ?? stub;
}

// ---------------------------------------------------------------------------
// Capability cache.
//
// `isAvailable()` is called on every VoiceLogModal mount AND on every
// page-mount of `/symptoms` for capability detection. The native call
// is fast (~ms) but caching keeps the symptom screen from re-querying
// on each Svelte re-render, and matches the pattern requested in
// Phase 4 of the sprint plan.
//
// TTL = 60s. Long enough to avoid repeat queries within a session,
// short enough that installing the Android offline speech model
// mid-session is picked up the next time the modal opens.
// ---------------------------------------------------------------------------

const AVAILABILITY_TTL_MS = 60_000;
let availabilityCache:
  | { value: SpeechRecognitionAvailability; expiresAt: number }
  | null = null;

/** Internal — exported only so tests can reset the cache between cases. */
export function _resetAvailabilityCache(): void {
  availabilityCache = null;
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

export const speechRecognizer = {
  /**
   * Cheap synchronous platform check. Returns true on iOS + Android
   * where the native plugin *may* be available; the authoritative
   * answer comes from `isAvailable()` below.
   *
   * Mirrors `healthkit.isAvailable()` so call sites that just need a
   * platform gate don't need to await a promise.
   */
  isNativePlatform(): boolean {
    if (typeof window === 'undefined') return false;
    const platform = Capacitor.getPlatform();
    return platform === 'ios' || platform === 'android';
  },

  /**
   * Definitive availability check — covers the device-capability
   * checks the native side performs (A12+ chip on iOS,
   * `isOnDeviceRecognitionAvailable` + offline-model gate on Android).
   *
   * Result is cached for 60s; pass `{ force: true }` to bypass.
   */
  async isAvailable(opts: { force?: boolean } = {}): Promise<
    SpeechRecognitionAvailability
  > {
    const now = Date.now();
    if (
      !opts.force &&
      availabilityCache &&
      availabilityCache.expiresAt > now
    ) {
      return availabilityCache.value;
    }
    let value: SpeechRecognitionAvailability;
    try {
      value = await plugin().isAvailable();
    } catch (err) {
      // Defensive — native side should never throw, but a Capacitor
      // bridge crash shouldn't blow up the symptom page. Treat as
      // unavailable + record the reason in dev for debugging.
      value = {
        available: false,
        reason: 'isAvailable threw: ' +
          (err instanceof Error ? err.message : 'unknown')
      };
    }
    availabilityCache = {
      value,
      expiresAt: now + AVAILABILITY_TTL_MS
    };
    return value;
  },

  /**
   * Requests both microphone + speech-recognition permissions on iOS
   * (combined into a single grant from the JS side; native code
   * sequences the two iOS prompts). On Android only RECORD_AUDIO is
   * required — `granted` reflects that single grant.
   */
  async requestPermission(): Promise<SpeechRecognitionPermission> {
    return plugin().requestPermission();
  },

  /**
   * Begin streaming dictation. Partial + final results are delivered
   * via the `partialResult` / `finalResult` event listeners — attach
   * those BEFORE calling `start()` to avoid missing the first partial.
   */
  async start(): Promise<SpeechRecognitionOk> {
    return plugin().start();
  },

  /**
   * Gracefully stop the session. The recognizer flushes any buffered
   * audio and emits a `finalResult` event. Resolves once the stop
   * request is acknowledged — the actual finalResult may arrive
   * milliseconds later.
   */
  async stop(): Promise<SpeechRecognitionOk> {
    return plugin().stop();
  },

  /**
   * Abort the session WITHOUT emitting a finalResult. Use this when
   * the user explicitly cancels (e.g., closes the modal mid-listen).
   */
  async cancel(): Promise<SpeechRecognitionOk> {
    return plugin().cancel();
  },

  /**
   * Subscribe to a recognition event. Returns a handle whose
   * `.remove()` method tears down the subscription.
   *
   * Pattern matches the Capacitor `addListener` convention used by
   * `@capacitor/push-notifications` etc., so callers familiar with
   * that surface should feel at home.
   *
   * Example:
   *
   *   const handle = await speechRecognizer.on('partialResult', (e) => {
   *     transcript = e.text;
   *   });
   *   // ... later, in onDestroy:
   *   await handle.remove();
   */
  async on<E extends SpeechRecognitionEvent>(
    event: E,
    handler: (data: SpeechRecognitionEventData<E>) => void
  ): Promise<PluginListenerHandle> {
    const result = plugin().addListener(event, (data) =>
      handler(data as SpeechRecognitionEventData<E>)
    );
    // `addListener` in Capacitor 5+ returns a Promise; in older
    // builds it was synchronous. Normalize both shapes.
    return await Promise.resolve(result);
  },

  /**
   * Convenience: clear all listeners for an event. Implemented as a
   * thin wrapper over the handle pattern — call sites that need to
   * detach listeners on component teardown should keep their own
   * handles and call `.remove()`. This helper exists for the rare
   * "blow it all away" path during error recovery.
   */
  async removeAllListeners(): Promise<void> {
    // The native plugins expose `removeAllListeners` only as part of
    // the standard Capacitor Plugin base class. We surface it
    // defensively — if the plugin doesn't expose it (web stub), we
    // just no-op.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const p = plugin() as any;
    if (typeof p.removeAllListeners === 'function') {
      await p.removeAllListeners();
    }
  }
};
