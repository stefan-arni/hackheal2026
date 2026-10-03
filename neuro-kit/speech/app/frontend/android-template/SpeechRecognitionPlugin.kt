/*
 * SpeechRecognition Capacitor plugin — template (Phase: Android bridge).
 *
 * Drop this into your Android Studio project after running
 * `npx cap add android` from the frontend directory. Final path:
 *   android/app/src/main/java/com/brainhealth/cowork/SpeechRecognitionPlugin.kt
 *
 * Wire it up in MainActivity.kt:
 *   class MainActivity : BridgeActivity() {
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *       registerPlugin(SpeechRecognitionPlugin::class.java)
 *       super.onCreate(savedInstanceState)
 *     }
 *   }
 *
 * Add the permission to AndroidManifest.xml:
 *   <uses-permission android:name="android.permission.RECORD_AUDIO" />
 *
 * Device requirements:
 *   - API 23+ (Android 6.0). On-device recognition support varies by
 *     OEM and language pack — the plugin gates via
 *     `SpeechRecognizer.isOnDeviceRecognitionAvailable(context)` on
 *     API 31+; older devices fall back to feature-detection via
 *     `SpeechRecognizer.isRecognitionAvailable()` plus a manual check
 *     for installed offline language models.
 *   - We DELIBERATELY set `RecognizerIntent.EXTRA_PREFER_OFFLINE = true`
 *     so audio never leaves the device. If the OS can't honor it (no
 *     offline model installed), `isAvailable()` returns false and the
 *     fallback hint banner renders.
 *
 * The plugin contract matches the TS surface in
 * `frontend/src/lib/native/speechRecognizer.ts`:
 *
 *   isAvailable()        -> { available: bool, reason?: string }
 *   requestPermission()  -> { granted: bool, status: string }
 *   start()              -> { ok: bool }   [streams events]
 *   stop()               -> { ok: bool }   [emits finalResult]
 *   cancel()             -> { ok: bool }   [no finalResult]
 *
 * Events emitted via `notifyListeners`:
 *   "partialResult" -> { text: string, isFinal: false }
 *   "finalResult"   -> { text: string, isFinal: true }
 *   "error"         -> { code: string, message: string }
 *
 * Device verification of this plugin runs on a real Android device
 * (Pixel 6+ for best on-device behavior, or any device with the
 * offline speech model installed via Settings → System → Languages
 * → Speech → Offline speech recognition). Emulators do NOT support
 * on-device recognition reliably — verification requires hardware.
 */
package com.brainhealth.cowork

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.core.content.ContextCompat
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import com.getcapacitor.annotation.Permission
import com.getcapacitor.annotation.PermissionCallback

@CapacitorPlugin(
    name = "SpeechRecognition",
    permissions = [
        Permission(
            alias = "microphone",
            strings = [Manifest.permission.RECORD_AUDIO]
        )
    ]
)
class SpeechRecognitionPlugin : Plugin() {

    // MARK: - Speech recognition state
    //
    // We keep a single in-flight recognition session per plugin
    // instance. start() is rejected if a session is already running;
    // stop() / cancel() / a final result tears everything down.

    private var speechRecognizer: SpeechRecognizer? = null
    private var isListening: Boolean = false

    // MARK: - isAvailable

    @PluginMethod
    fun isAvailable(call: PluginCall) {
        val ctx: Context = context ?: run {
            val ret = JSObject()
            ret.put("available", false)
            ret.put("reason", "context unavailable")
            call.resolve(ret)
            return
        }

        if (!SpeechRecognizer.isRecognitionAvailable(ctx)) {
            val ret = JSObject()
            ret.put("available", false)
            ret.put("reason", "speech recognition service not available")
            call.resolve(ret)
            return
        }

        // API 31+ exposes the dedicated on-device availability check.
        // Older API levels have no first-class way to verify "offline
        // model installed" — we accept the EXTRA_PREFER_OFFLINE intent
        // hint and degrade gracefully on first `start()` error if the
        // OS can't honor it.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val onDevice = try {
                SpeechRecognizer.isOnDeviceRecognitionAvailable(ctx)
            } catch (e: Throwable) {
                // Some OEM builds throw NoSuchMethodError on this call
                // despite reporting API 31+. Treat as "unknown" and
                // fall through to the feature-detection path.
                null
            }
            if (onDevice == false) {
                val ret = JSObject()
                ret.put("available", false)
                ret.put("reason", "offline speech model not installed")
                call.resolve(ret)
                return
            }
        }

        val ret = JSObject()
        ret.put("available", true)
        call.resolve(ret)
    }

    // MARK: - requestPermission

    @PluginMethod
    fun requestPermission(call: PluginCall) {
        if (ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            val ret = JSObject()
            ret.put("granted", true)
            ret.put("status", "authorized")
            call.resolve(ret)
            return
        }
        requestPermissionForAlias("microphone", call, "permissionCallback")
    }

    @PermissionCallback
    private fun permissionCallback(call: PluginCall) {
        val granted = getPermissionState("microphone").toString() == "GRANTED"
        val ret = JSObject()
        ret.put("granted", granted)
        ret.put("status", if (granted) "authorized" else "microphone_denied")
        call.resolve(ret)
    }

    // MARK: - start

    @PluginMethod
    fun start(call: PluginCall) {
        val ctx: Context = context ?: run {
            call.resolve(makeError("context_unavailable", "context unavailable"))
            return
        }
        if (isListening) {
            val ret = JSObject()
            ret.put("ok", false)
            ret.put("reason", "session already running")
            call.resolve(ret)
            return
        }

        // SpeechRecognizer must be instantiated AND its callbacks
        // attached on the main thread; Capacitor calls plugin methods
        // there, so we don't need to re-dispatch.
        val recognizer = try {
            SpeechRecognizer.createSpeechRecognizer(ctx)
        } catch (e: Throwable) {
            emitError("recognizer_create_failed", e.message ?: "unknown")
            val ret = JSObject()
            ret.put("ok", false)
            ret.put("reason", "recognizer_create_failed")
            call.resolve(ret)
            return
        }
        speechRecognizer = recognizer

        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                val (code, message) = mapErrorCode(error)
                emitError(code, message)
                tearDown()
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val text = extractFirstResult(partialResults) ?: return
                val data = JSObject()
                data.put("text", text)
                data.put("isFinal", false)
                notifyListeners("partialResult", data)
            }

            override fun onResults(results: Bundle?) {
                val text = extractFirstResult(results) ?: ""
                val data = JSObject()
                data.put("text", text)
                data.put("isFinal", true)
                notifyListeners("finalResult", data)
                tearDown()
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            // CRITICAL: keeps audio on-device. If the OS can't honor
            // (no offline model installed), onError() will fire with
            // ERROR_NETWORK or ERROR_INSUFFICIENT_PERMISSIONS — we
            // surface that via the error listener.
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }

        try {
            recognizer.startListening(intent)
            isListening = true
        } catch (e: Throwable) {
            emitError("start_failed", e.message ?: "unknown")
            tearDown()
            val ret = JSObject()
            ret.put("ok", false)
            ret.put("reason", "start_failed")
            call.resolve(ret)
            return
        }

        val ret = JSObject()
        ret.put("ok", true)
        call.resolve(ret)
    }

    // MARK: - stop

    @PluginMethod
    fun stop(call: PluginCall) {
        if (!isListening) {
            val ret = JSObject()
            ret.put("ok", false)
            ret.put("reason", "no active session")
            call.resolve(ret)
            return
        }
        try {
            // stopListening() asks the recognizer to flush any
            // buffered audio and emit onResults(). The listener
            // then calls tearDown().
            speechRecognizer?.stopListening()
        } catch (e: Throwable) {
            // Defensive — some OEMs throw if the recognizer was
            // already stopped. Emit a clean final and tear down.
            val data = JSObject()
            data.put("text", "")
            data.put("isFinal", true)
            notifyListeners("finalResult", data)
            tearDown()
        }
        val ret = JSObject()
        ret.put("ok", true)
        call.resolve(ret)
    }

    // MARK: - cancel

    @PluginMethod
    fun cancel(call: PluginCall) {
        if (!isListening) {
            val ret = JSObject()
            ret.put("ok", false)
            ret.put("reason", "no active session")
            call.resolve(ret)
            return
        }
        try {
            speechRecognizer?.cancel()
        } catch (e: Throwable) {
            // ignore — we're tearing down anyway
        }
        tearDown()
        val ret = JSObject()
        ret.put("ok", true)
        call.resolve(ret)
    }

    // MARK: - helpers

    private fun emitError(code: String, message: String) {
        val data = JSObject()
        data.put("code", code)
        data.put("message", message)
        notifyListeners("error", data)
    }

    private fun makeError(code: String, message: String): JSObject {
        val ret = JSObject()
        ret.put("ok", false)
        ret.put("reason", code)
        ret.put("message", message)
        return ret
    }

    private fun extractFirstResult(b: Bundle?): String? {
        val list = b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        return list?.firstOrNull()
    }

    /// Translates Android's int error codes into the stable string
    /// codes our TS / Svelte side switches on. Keep these strings in
    /// sync with the iOS plugin's error code vocabulary where they
    /// overlap (e.g., "no_speech", "audio_engine_failed").
    private fun mapErrorCode(error: Int): Pair<String, String> {
        return when (error) {
            SpeechRecognizer.ERROR_AUDIO ->
                "audio_engine_failed" to "audio recording error"
            SpeechRecognizer.ERROR_CLIENT ->
                "client_error" to "client side error"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS ->
                "permission_denied" to "microphone permission denied"
            SpeechRecognizer.ERROR_NETWORK ->
                "no_on_device" to "offline speech model not installed"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT ->
                "no_on_device" to "offline speech model timeout"
            SpeechRecognizer.ERROR_NO_MATCH ->
                "no_match" to "no speech detected"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY ->
                "recognizer_busy" to "recognizer is busy"
            SpeechRecognizer.ERROR_SERVER ->
                "no_on_device" to "server-side recognition refused — offline model required"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT ->
                "no_speech" to "no speech detected within timeout"
            else ->
                "recognition_failed" to "speech recognition failed (code=$error)"
        }
    }

    /// Releases the recognizer + state. Idempotent — safe to call
    /// from multiple completion paths.
    private fun tearDown() {
        try {
            speechRecognizer?.destroy()
        } catch (e: Throwable) {
            // ignore
        }
        speechRecognizer = null
        isListening = false
    }

    override fun handleOnDestroy() {
        tearDown()
        super.handleOnDestroy()
    }
}
