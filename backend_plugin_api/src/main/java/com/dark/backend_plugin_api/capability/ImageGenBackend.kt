package com.dark.backend_plugin_api.capability

import com.dark.backend_plugin_api.callback.ImageGenCallback
import com.dark.backend_plugin_api.model.ImageGenParams

/**
 * Image generation capability.
 * Implement alongside [BackendPlugin] if the backend supports IMAGE_GEN.
 */
interface ImageGenBackend {

    suspend fun generate(params: ImageGenParams, callback: ImageGenCallback)

    fun stopGeneration()

    /** Apply a LoRA adapter at runtime */
    fun applyLora(loraPath: String, weight: Float)

    /** Remove applied LoRA */
    fun clearLora()
}
