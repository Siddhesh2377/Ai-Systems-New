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

    /**
     * VLM-only per-stage timing, emitted once after all image chunks have been
     * encoded and their embeddings pushed through the LLM, before generation starts.
     *
     * @param vlmEncodeMs Total time spent in the vision/audio encoder (ViT / conformer) forward passes.
     * @param vlmDecodeMs Total time spent running llama_decode on image+text chunks during prompt-eval.
     * @param imageTokens Number of image embedding tokens consumed by the LLM.
     *
     * Default no-op for backwards compatibility.
     */
    fun onVlmStageMetrics(vlmEncodeMs: Float, vlmDecodeMs: Float, imageTokens: Int) {}
}
