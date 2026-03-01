package com.dark.backend_plugin_api.capability

import com.dark.backend_plugin_api.callback.TextGenCallback

/**
 * Vision (VLM) capability — multimodal text + image input.
 * Implement alongside [TextGenBackend] if the backend supports VISION.
 */
interface VisionBackend {

    /**
     * Load a vision adapter/projector model (e.g., mmproj GGUF for LLaVA).
     * @param path file path or fd URI to the vision model
     */
    suspend fun loadVisionModel(path: String): Result<Unit>

    /**
     * Generate from text + image input.
     * @param messagesJson JSON with image references embedded
     * @param imagePaths list of image file paths
     * @param maxTokens maximum tokens to generate
     * @param callback receives tokens and completion
     */
    suspend fun generateWithVision(
        messagesJson: String,
        imagePaths: List<String>,
        maxTokens: Int,
        callback: TextGenCallback
    )

    fun releaseVisionModel()
}
