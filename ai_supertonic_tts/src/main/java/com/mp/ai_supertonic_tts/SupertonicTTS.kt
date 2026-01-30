package com.mp.ai_supertonic_tts

import android.content.Context
import android.net.Uri
import com.mp.ai_supertonic_tts.audio.AudioPlayer
import com.mp.ai_supertonic_tts.audio.AudioSaver
import com.mp.ai_supertonic_tts.callback.TTSCallback
import com.mp.ai_supertonic_tts.engine.TTSEngine
import com.mp.ai_supertonic_tts.models.AudioFormat
import com.mp.ai_supertonic_tts.models.SynthesisResult
import com.mp.ai_supertonic_tts.models.TTSConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Main SDK entry point for Supertonic TTS on Android.
 *
 * Provides a clean, high-level API for:
 * - Loading Supertonic ONNX models from any path
 * - Text-to-speech synthesis with configurable voice, speed, and quality
 * - Real-time audio playback via AudioTrack
 * - Saving audio in multiple formats (WAV 16-bit, WAV 32-float, PCM, raw)
 * - Reading text from strings, files, URIs, or InputStreams
 *
 * Usage:
 * ```kotlin
 * val tts = SupertonicTTS(context)
 * tts.loadModel("/path/to/supertonic-2")
 *
 * // Synthesize and play
 * tts.speak("Hello world")
 *
 * // Synthesize and save
 * val result = tts.synthesize("Hello world")
 * tts.saveAudio(result, "/path/output.wav")
 *
 * // Custom config
 * tts.speak("Bonjour", TTSConfig(
 *     voice = "F2",
 *     language = Language.FR,
 *     speed = 1.0f,
 *     steps = 5
 * ))
 *
 * // Cleanup
 * tts.release()
 * ```
 */
class SupertonicTTS(private val context: Context? = null) {

    private val nativeLib = SupertonicNativeLib()
    private val engine = TTSEngine(nativeLib)
    private val player = AudioPlayer()

    /** Last error message if an operation failed */
    var lastError: String? = null
        private set

    // ========================================================================
    // MODEL MANAGEMENT
    // ========================================================================

    /**
     * Load Supertonic ONNX models from a directory.
     *
     * Expected directory structure:
     * ```
     * modelDir/
     *   onnx/
     *     duration_predictor.onnx
     *     text_encoder.onnx
     *     vector_estimator.onnx
     *     vocoder.onnx
     *     tts.json
     *     unicode_indexer.json
     *   voice_styles/
     *     F1.json, F2.json, ..., M1.json, M2.json, ...
     * ```
     *
     * @param modelDir Root directory containing onnx/ and voice_styles/
     * @param useNNAPI Enable NNAPI GPU/NPU acceleration (device-dependent)
     * @return true if all models loaded successfully
     */
    fun loadModel(modelDir: String, useNNAPI: Boolean = false): Boolean {
        val success = engine.loadModel(modelDir, useNNAPI)
        if (!success) {
            lastError = engine.lastError
        }
        return success
    }

    /**
     * Check if models are loaded and ready for synthesis.
     */
    fun isModelLoaded(): Boolean = engine.isLoaded()

    /**
     * Get list of available voice style names (e.g. ["F1", "F2", "M1", ...]).
     */
    fun getAvailableVoices(): List<String> = engine.getAvailableVoices()

    /**
     * Release all resources (ONNX sessions, audio player).
     * Call this when you're done with the TTS engine.
     */
    fun release() {
        player.release()
        engine.release()
    }

    // ========================================================================
    // SYNTHESIS
    // ========================================================================

    /**
     * Synthesize speech from text and return the audio result.
     *
     * @param text Input text to synthesize
     * @param config Synthesis configuration (voice, speed, steps, language)
     * @return SynthesisResult containing audio data and metadata
     */
    suspend fun synthesize(
        text: String,
        config: TTSConfig = TTSConfig()
    ): SynthesisResult {
        return engine.synthesize(text, config)
    }

    /**
     * Synthesize speech with a progress callback.
     *
     * @param text Input text to synthesize
     * @param config Synthesis configuration
     * @param callback Progress and result callback
     * @return SynthesisResult containing audio data and metadata
     */
    suspend fun synthesize(
        text: String,
        config: TTSConfig = TTSConfig(),
        callback: TTSCallback
    ): SynthesisResult {
        return try {
            engine.synthesize(text, config, callback)
        } catch (e: Exception) {
            lastError = e.message
            callback.onError(e.message ?: "Synthesis failed")
            throw e
        }
    }

    // ========================================================================
    // PLAYBACK
    // ========================================================================

    /**
     * Synthesize and play speech.
     *
     * @param text Input text to speak
     * @param config Synthesis configuration
     */
    suspend fun speak(text: String, config: TTSConfig = TTSConfig()) {
        val result = synthesize(text, config)
        withContext(Dispatchers.IO) {
            player.playStreaming(result)
        }
    }

    /**
     * Synthesize and play speech with a progress callback.
     *
     * @param text Input text to speak
     * @param config Synthesis configuration
     * @param callback Progress and result callback
     */
    suspend fun speak(
        text: String,
        config: TTSConfig = TTSConfig(),
        callback: TTSCallback
    ) {
        val result = synthesize(text, config, callback)
        withContext(Dispatchers.IO) {
            player.playStreaming(result)
        }
    }

    /** Stop audio playback */
    fun stopPlayback() = player.stop()

    /** Pause audio playback */
    fun pausePlayback() = player.pause()

    /** Resume paused playback */
    fun resumePlayback() = player.resume()

    /** Check if audio is currently playing */
    fun isPlaying(): Boolean = player.isPlaying()

    /** Set playback volume (0.0 to 1.0) */
    fun setVolume(volume: Float) = player.setVolume(volume)

    // ========================================================================
    // INPUT SOURCES
    // ========================================================================

    /**
     * Read text from a file and speak it.
     */
    suspend fun speakFromFile(path: String, config: TTSConfig = TTSConfig()) {
        val text = File(path).readText()
        speak(text, config)
    }

    /**
     * Read text from a content URI and speak it.
     * Requires a Context (passed in constructor).
     */
    suspend fun speakFromUri(uri: Uri, config: TTSConfig = TTSConfig()) {
        val ctx = context ?: throw IllegalStateException("Context required for URI reading. Pass context in constructor.")
        val text = ctx.contentResolver.openInputStream(uri)?.use { stream ->
            BufferedReader(InputStreamReader(stream)).readText()
        } ?: throw IllegalArgumentException("Could not open URI: $uri")
        speak(text, config)
    }

    /**
     * Read text from an InputStream and speak it.
     */
    suspend fun speakFromStream(stream: InputStream, config: TTSConfig = TTSConfig()) {
        val text = BufferedReader(InputStreamReader(stream)).readText()
        speak(text, config)
    }

    /**
     * Read text from a file and return the synthesis result.
     */
    suspend fun synthesizeFromFile(
        path: String,
        config: TTSConfig = TTSConfig()
    ): SynthesisResult {
        val text = File(path).readText()
        return synthesize(text, config)
    }

    /**
     * Read text from a content URI and return the synthesis result.
     */
    suspend fun synthesizeFromUri(
        uri: Uri,
        config: TTSConfig = TTSConfig()
    ): SynthesisResult {
        val ctx = context ?: throw IllegalStateException("Context required for URI reading. Pass context in constructor.")
        val text = ctx.contentResolver.openInputStream(uri)?.use { stream ->
            BufferedReader(InputStreamReader(stream)).readText()
        } ?: throw IllegalArgumentException("Could not open URI: $uri")
        return synthesize(text, config)
    }

    // ========================================================================
    // AUDIO SAVING
    // ========================================================================

    /**
     * Save a synthesis result to a file path.
     *
     * @param result Audio data from synthesize()
     * @param path Output file path
     * @param format Audio format (default: WAV_16)
     * @return true if saved successfully
     */
    fun saveAudio(
        result: SynthesisResult,
        path: String,
        format: AudioFormat = AudioFormat.WAV_16
    ): Boolean {
        return AudioSaver.save(result, path, format, nativeLib)
    }

    /**
     * Save a synthesis result to a content URI.
     * Requires a Context (passed in constructor).
     *
     * @param result Audio data from synthesize()
     * @param uri Output URI
     * @param format Audio format (default: WAV_16)
     * @return true if saved successfully
     */
    fun saveAudio(
        result: SynthesisResult,
        uri: Uri,
        format: AudioFormat = AudioFormat.WAV_16
    ): Boolean {
        val ctx = context ?: throw IllegalStateException("Context required for URI saving. Pass context in constructor.")
        return AudioSaver.save(result, uri, ctx, format, nativeLib)
    }

    /**
     * Convert a synthesis result to a byte array in the specified format.
     *
     * @param result Audio data from synthesize()
     * @param format Audio format
     * @return Byte array of encoded audio
     */
    fun toByteArray(result: SynthesisResult, format: AudioFormat): ByteArray {
        return AudioSaver.toByteArray(result, format, nativeLib)
    }
}
