/*
 * SpeechRecognition Capacitor plugin — template (Phase: iOS bridge).
 *
 * Drop this into your Xcode project after running
 * `npx cap add ios`. Final path:
 *   ios/App/App/Speech/SpeechRecognitionPlugin.swift
 *
 * Wire it up by adding the file to the App target in Xcode (drag into
 * the project navigator and check "App" in target membership).
 * Capacitor's plugin auto-discovery (introspecting
 * @objc(SpeechRecognitionPlugin)) picks it up after `npx cap sync ios`.
 *
 * Required Xcode setup:
 *   - In Info.plist, add:
 *       NSSpeechRecognitionUsageDescription
 *         "Brain Health uses on-device speech recognition so you can
 *          log symptoms by voice. Audio is processed entirely on your
 *          device and never leaves it."
 *       NSMicrophoneUsageDescription
 *         "Brain Health needs the microphone to capture your voice
 *          when you log symptoms by speaking. Audio never leaves
 *          your device."
 *   - No special capability required — speech recognition is part of
 *     the standard iOS SDK from iOS 10+.
 *
 * Device requirements:
 *   - iOS 13+ (we gate on `supportsOnDeviceRecognition`, available iOS 13+).
 *   - A12 Bionic chip or newer (iPhone XS / XR / 11 / 12 / 13 / 14 / 15 / 16,
 *     iPad Pro 11" / iPad Air 4+, iPad mini 6+). Older devices return
 *     `isAvailable() -> { available: false, reason: "device does not
 *     support on-device recognition" }`.
 *   - We DELIBERATELY set `requiresOnDeviceRecognition = true` so audio
 *     never leaves the device. No PHI-in-transit, no BAA needed.
 *
 * The plugin contract matches the TS surface in
 * `frontend/src/lib/native/speechRecognizer.ts`:
 *
 *   isAvailable()        -> { available: bool, reason?: string }
 *   requestPermission()  -> { granted: bool, status: string }
 *   start()              -> { ok: bool }     [streams events]
 *   stop()               -> { ok: bool }     [emits finalResult]
 *   cancel()             -> { ok: bool }     [no finalResult]
 *
 * Events emitted via `notifyListeners`:
 *   "partialResult" -> { text: string, isFinal: false }
 *   "finalResult"   -> { text: string, isFinal: true }
 *   "error"         -> { code: string, message: string }
 *
 * Internal type vocabulary (kept in sync with speechRecognizer.ts +
 * SpeechRecognitionPlugin.kt): there is no metric-key vocabulary here
 * — the plugin returns plain text. The extraction step that turns
 * "I have a headache about a 5 since lunch" into structured
 * `{symptom, severity, notes}` happens server-side via the existing
 * `voiceLog.extract()` flow (unchanged from the 2026-05-04 sprint).
 *
 * Device verification of this plugin runs on a real iPhone or iPad —
 * the simulator's speech recognition is unreliable and does not
 * exercise the on-device path. See `MOBILE_BUILD.md § 10`.
 */
import AVFoundation
import Capacitor
import Foundation
import Speech

@objc(SpeechRecognitionPlugin)
public class SpeechRecognitionPlugin: CAPPlugin {

    // MARK: - Speech recognition state
    //
    // We keep a single in-flight recognition session per plugin
    // instance. start() is rejected if a session is already running;
    // stop() / cancel() / a final result tears everything down.

    private let audioEngine = AVAudioEngine()
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    // MARK: - isAvailable

    /// Resolves true only when the device supports on-device speech
    /// recognition. We never fall back to Apple's server-side
    /// recognition (would violate the "audio never leaves the device"
    /// guarantee).
    @objc func isAvailable(_ call: CAPPluginCall) {
        // Recognizer for the device's preferred locale, falling back
        // to en-US. (English-only is a deliberate v1 limitation —
        // OBSERVED_SIGN_TAGS and the symptom enum are English-only.)
        let locale = Locale(identifier: "en-US")
        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            call.resolve([
                "available": false,
                "reason": "locale not supported"
            ])
            return
        }

        if !recognizer.isAvailable {
            call.resolve([
                "available": false,
                "reason": "speech recognizer is currently unavailable"
            ])
            return
        }

        // The A12+ chip gate. supportsOnDeviceRecognition is the only
        // reliable signal — older devices route through Apple's
        // servers, which we refuse.
        if !recognizer.supportsOnDeviceRecognition {
            call.resolve([
                "available": false,
                "reason": "device does not support on-device recognition"
            ])
            return
        }

        call.resolve(["available": true])
    }

    // MARK: - requestPermission

    /// Requests BOTH permissions iOS needs for dictation:
    ///   1. Speech recognition authorization (NSSpeechRecognitionUsageDescription).
    ///   2. Microphone access (NSMicrophoneUsageDescription).
    /// Resolves with the combined result. Either denial blocks dictation.
    @objc func requestPermission(_ call: CAPPluginCall) {
        SFSpeechRecognizer.requestAuthorization { speechStatus in
            // SFSpeechRecognizer.requestAuthorization invokes its
            // completion on an arbitrary queue — bounce to main for
            // the audio session check (and so the JS bridge sees a
            // consistent thread).
            DispatchQueue.main.async {
                let speechStr = self.statusString(speechStatus)
                let speechGranted = speechStatus == .authorized

                AVAudioSession.sharedInstance().requestRecordPermission { micGranted in
                    DispatchQueue.main.async {
                        let granted = speechGranted && micGranted
                        let status: String
                        if !speechGranted {
                            status = "speech_\(speechStr)"
                        } else if !micGranted {
                            status = "microphone_denied"
                        } else {
                            status = "authorized"
                        }
                        call.resolve([
                            "granted": granted,
                            "status": status
                        ])
                    }
                }
            }
        }
    }

    private func statusString(_ status: SFSpeechRecognizerAuthorizationStatus) -> String {
        switch status {
        case .notDetermined: return "notDetermined"
        case .denied:        return "denied"
        case .restricted:    return "restricted"
        case .authorized:    return "authorized"
        @unknown default:    return "unknown"
        }
    }

    // MARK: - start

    /// Begins streaming on-device recognition. Partial results are
    /// emitted via the `partialResult` listener; the final transcript
    /// arrives via `finalResult` once the user calls stop() or the
    /// recognizer auto-finalizes after silence.
    @objc func start(_ call: CAPPluginCall) {
        // Guard against double-start. We resolve rather than reject
        // since the caller may have lost track of state during a
        // re-render — the existing session keeps running, no harm done.
        if recognitionTask != nil {
            call.resolve([
                "ok": false,
                "reason": "session already running"
            ])
            return
        }

        let locale = Locale(identifier: "en-US")
        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            emitError(code: "unsupported_locale",
                      message: "locale not supported")
            call.resolve(["ok": false, "reason": "locale not supported"])
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            emitError(code: "no_on_device",
                      message: "device does not support on-device recognition")
            call.resolve(["ok": false,
                          "reason": "device does not support on-device recognition"])
            return
        }
        self.speechRecognizer = recognizer

        // Configure the audio session. .measurement gives the speech
        // recognizer the most stable input; .duckOthers politely
        // lowers other audio (music, etc.) while we listen.
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record,
                                    mode: .measurement,
                                    options: [.duckOthers])
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            emitError(code: "audio_session_failed",
                      message: error.localizedDescription)
            call.resolve([
                "ok": false,
                "reason": "audio_session_failed"
            ])
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        // CRITICAL: this is what keeps audio on-device. If iOS can't
        // satisfy the requirement (e.g., model not downloaded yet), the
        // task will fail — we surface that via the error listener.
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        self.recognitionRequest = request

        // Hook up the audio engine's input node into the request.
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        // Defensive: removing any tap installed by a prior session
        // that didn't tear down cleanly (e.g., killed mid-listen).
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(
            onBus: 0,
            bufferSize: 1024,
            format: recordingFormat
        ) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            tearDown()
            emitError(code: "audio_engine_failed",
                      message: error.localizedDescription)
            call.resolve([
                "ok": false,
                "reason": "audio_engine_failed"
            ])
            return
        }

        // Start the recognition task. The handler is invoked on each
        // partial result and once on the final result (or on error).
        self.recognitionTask = recognizer.recognitionTask(with: request) {
            [weak self] result, error in
            guard let self = self else { return }

            if let result = result {
                let text = result.bestTranscription.formattedString
                if result.isFinal {
                    self.notifyListeners("finalResult", data: [
                        "text": text,
                        "isFinal": true
                    ])
                    self.tearDown()
                } else {
                    self.notifyListeners("partialResult", data: [
                        "text": text,
                        "isFinal": false
                    ])
                }
            }

            if let error = error {
                // SFSpeechRecognizer reports a no-speech-detected
                // "error" with code 203 when stop() is called before
                // any audio arrives. We translate that into a clean
                // empty finalResult rather than surfacing as an
                // error to the UI.
                let nsError = error as NSError
                if nsError.domain == "kAFAssistantErrorDomain"
                    && nsError.code == 203 {
                    self.notifyListeners("finalResult", data: [
                        "text": "",
                        "isFinal": true
                    ])
                } else {
                    self.emitError(code: "recognition_failed",
                                   message: nsError.localizedDescription)
                }
                self.tearDown()
            }
        }

        call.resolve(["ok": true])
    }

    // MARK: - stop

    /// Ends the audio capture gracefully. The recognizer flushes any
    /// buffered audio and emits a final result via the listener.
    @objc func stop(_ call: CAPPluginCall) {
        guard recognitionTask != nil else {
            call.resolve(["ok": false, "reason": "no active session"])
            return
        }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        // Don't tearDown() here — we want the recognition task's
        // completion handler to fire one more time with isFinal=true,
        // which then calls tearDown() itself.
        call.resolve(["ok": true])
    }

    // MARK: - cancel

    /// Aborts the session without emitting a final result.
    @objc func cancel(_ call: CAPPluginCall) {
        guard recognitionTask != nil else {
            call.resolve(["ok": false, "reason": "no active session"])
            return
        }
        recognitionTask?.cancel()
        tearDown()
        call.resolve(["ok": true])
    }

    // MARK: - helpers

    private func emitError(code: String, message: String) {
        notifyListeners("error", data: [
            "code": code,
            "message": message
        ])
    }

    /// Releases every piece of audio + recognizer state. Idempotent —
    /// safe to call from multiple completion paths.
    private func tearDown() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        recognitionRequest = nil
        recognitionTask = nil
        // Release the audio session so other apps' audio can resume.
        // .notifyOthersOnDeactivation lets backgrounded music apps
        // (Spotify, Apple Music) auto-resume.
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }
}
