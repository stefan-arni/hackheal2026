/**
 * Web stub for the SpeechRecognizer Capacitor plugin.
 *
 * This file is imported in web (browser) environments where the native
 * Capacitor plugin is not available. All methods return sensible
 * defaults so call sites never need to null-check.
 *
 * On a real device (iOS / Android) the bridge in speechRecognizer.ts
 * resolves to the registered native plugin instead of this stub.
 *
 * Exported for completeness; consumers should import from
 * `$lib/native/speechRecognizer` — not from this file directly.
 */
import type { PluginListenerHandle } from '@capacitor/core';
import type { SpeechRecognitionPlugin } from './speechRecognizer';

const WEB_REASON = 'web platform — use text form';

export const webStub: SpeechRecognitionPlugin = {
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
  addListener(): PluginListenerHandle {
    return { remove: async () => { /* noop */ } };
  },
};
