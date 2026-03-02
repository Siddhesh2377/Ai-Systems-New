package com.dark.ai_chatterbox

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/**
 * High-level manager for Chatterbox TTS.
 * Singleton — use [getInstance] to obtain.
 *
 * Manages model loading, voice preset loading, and text-to-speech synthesis.
 * State is exposed via [state] StateFlow.
 *
 * Usage:
 * ```kotlin
 * val mgr = ChatterboxManager.getInstance()
 * mgr.loadModel(config)
 * mgr.synthesize("Hello world")
 * mgr.state.collect { state -> ... }
 * ```
 */
class ChatterboxManager private constructor() {

    companion object {
        private const val TAG = "ChatterboxManager"

        @Volatile
        private var instance: ChatterboxManager? = null

        fun getInstance(): ChatterboxManager =
            instance ?: synchronized(this) {
                instance ?: ChatterboxManager().also { instance = it }
            }
    }

    private val nativeLib = ChatterboxNativeLib()
    private val stateLock = Any()

    private val _state = MutableStateFlow<ChatterboxState>(ChatterboxState.Idle)
    val state: StateFlow<ChatterboxState> = _state.asStateFlow()

    private fun updateState(newState: ChatterboxState) {
        synchronized(stateLock) { _state.value = newState }
    }

    /**
     * Load model and optional voice preset.
     * Runs on IO dispatcher. Updates [state] through Loading -> Ready or Error.
     *
     * @return true if models loaded successfully
     */
    suspend fun loadModel(config: ChatterboxConfig): Boolean = withContext(Dispatchers.IO) {
        updateState(ChatterboxState.Loading)
        try {
            // Set variant BEFORE loading models — determines I/O name count
            nativeLib.nativeSetVariant(
                if (config.variant == ChatterboxVariant.ORIGINAL) 1 else 0
            )
            nativeLib.nativeSetExaggeration(config.exaggeration)

            // Load tokenizer
            if (!nativeLib.nativeLoadTokenizer(config.tokenizerPath)) {
                updateState(ChatterboxState.Error("Failed to load tokenizer"))
                return@withContext false
            }

            // Load ONNX models
            if (!nativeLib.nativeLoadModels(config.modelDir)) {
                updateState(ChatterboxState.Error("Failed to load models"))
                return@withContext false
            }

            // Apply config
            nativeLib.nativeSetRepetitionPenalty(config.repetitionPenalty)
            nativeLib.nativeSetMaxTokens(config.maxTokens)

            // Load voice preset if provided
            if (config.voicePresetDir != null) {
                if (!nativeLib.nativeLoadVoicePreset(config.voicePresetDir)) {
                    updateState(ChatterboxState.Error("Failed to load voice preset"))
                    return@withContext false
                }
            }

            updateState(ChatterboxState.Ready)
            Log.i(TAG, "Model loaded: ${config.modelDir}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            updateState(ChatterboxState.Error(e.message ?: "Unknown error"))
            false
        }
    }

    /**
     * Load a voice preset (can be called after loadModel to swap voices).
     */
    fun loadVoicePreset(styleDir: String): Boolean {
        val result = nativeLib.nativeLoadVoicePreset(styleDir)
        if (!result) {
            Log.e(TAG, "Failed to load voice preset: $styleDir")
        }
        return result
    }

    /**
     * Synthesize text to speech.
     * Updates [state]: Generating(tokens) -> Complete(pcm, sampleRate) or Error.
     * Runs on IO dispatcher.
     *
     * @param text The text to synthesize
     * @return PCM audio data or null on failure
     */
    suspend fun synthesize(text: String): ShortArray? = withContext(Dispatchers.IO) {
        var result: ShortArray? = null

        val callback = object : ChatterboxCallback {
            override fun onSpeechTokenProgress(tokensGenerated: Int) {
                updateState(ChatterboxState.Generating(tokensGenerated))
            }

            override fun onAudioReady(pcmData: ShortArray, sampleRate: Int) {
                result = pcmData
                updateState(ChatterboxState.Complete(pcmData, sampleRate))
            }

            override fun onError(message: String) {
                updateState(ChatterboxState.Error(message))
            }
        }

        try {
            nativeLib.nativeSynthesize(text, callback)
        } catch (e: Exception) {
            Log.e(TAG, "Synthesis failed", e)
            updateState(ChatterboxState.Error(e.message ?: "Synthesis failed"))
        }

        result
    }

    /**
     * Stop an in-progress synthesis.
     * Safe to call from any thread.
     */
    fun stop() {
        nativeLib.nativeStop()
    }

    /**
     * Release all native resources.
     */
    fun release() {
        nativeLib.nativeRelease()
        updateState(ChatterboxState.Idle)
        Log.i(TAG, "Released")
    }

    /**
     * @return true if models are loaded and ready
     */
    fun isReady(): Boolean = nativeLib.nativeIsLoaded() && nativeLib.nativeIsVoiceLoaded()
}
