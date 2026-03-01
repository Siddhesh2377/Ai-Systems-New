package com.dark.backend_plugin_api.capability

import com.dark.backend_plugin_api.callback.TTSCallback
import com.dark.backend_plugin_api.model.TTSParams

/**
 * Text-to-speech capability.
 * Implement alongside [BackendPlugin] if the backend supports TTS.
 */
interface TTSBackend {

    suspend fun synthesize(text: String, config: TTSParams, callback: TTSCallback)

    fun stopSynthesis()

    /** List available voice IDs for this model */
    fun getAvailableVoices(): List<String>
}
