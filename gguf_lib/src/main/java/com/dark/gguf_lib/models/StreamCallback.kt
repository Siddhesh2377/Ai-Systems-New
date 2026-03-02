package com.dark.gguf_lib.models

/**
 * Callback interface for streaming text generation.
 * Called from native code during token generation.
 */
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
    fun onMetrics(
        tps: Float, ttftMs: Float, totalMs: Float,
        tokensEvaluated: Int, tokensPredicted: Int,
        modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float
    )

    /** Prompt evaluation progress (0.0 to 1.0). Default no-op. */
    fun onProgress(progress: Float) {}

    /**
     * Zero-copy token delivery via pre-allocated byte array.
     * Only [length] bytes in [data] are valid (UTF-8 encoded).
     * Default implementation converts to String and calls [onToken].
     * Override for zero-copy processing (e.g. direct write to stream).
     */
    fun onTokenBytes(data: ByteArray, length: Int) {
        onToken(String(data, 0, length, Charsets.UTF_8))
    }
}
