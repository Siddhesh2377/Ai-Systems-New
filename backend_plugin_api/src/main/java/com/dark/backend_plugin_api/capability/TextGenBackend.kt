package com.dark.backend_plugin_api.capability

import com.dark.backend_plugin_api.callback.TextGenCallback

/**
 * Text generation capability.
 * Implement alongside [BackendPlugin] if the backend supports TEXT_GEN.
 *
 * Example for ai_gguf:
 * ```
 * class GGUFBackendPlugin : BackendPlugin, TextGenBackend, EmbeddingBackend {
 *     // ...
 * }
 * ```
 */
interface TextGenBackend {

    /**
     * Stream tokens for a multi-turn conversation.
     * @param messagesJson JSON array of messages: [{"role":"user","content":"..."},...]
     * @param maxTokens maximum tokens to generate
     * @param callback receives tokens, tool calls, completion, errors
     */
    suspend fun generateStream(
        messagesJson: String,
        maxTokens: Int,
        callback: TextGenCallback
    )

    /** Halt generation mid-stream. The callback will receive onDone after stopping. */
    fun stopGeneration()

    /** Set stop strings that halt generation when encountered */
    fun setStopStrings(strings: Array<String>)

    /**
     * Update sampler parameters at runtime (temperature, top_p, etc.).
     * @param paramsJson JSON object with sampler keys to update
     */
    fun updateSamplerParams(paramsJson: String)
}
