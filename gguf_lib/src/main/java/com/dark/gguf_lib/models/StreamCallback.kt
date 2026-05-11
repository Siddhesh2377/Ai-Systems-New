package com.dark.gguf_lib.models

/**
 * Callback interface for streaming text generation.
 * Called from native code during token generation.
 */
interface StreamCallback {
    fun onToken(token: String)
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

    /**
     * VT cache hit/miss for a single image chunk. Fired once per image when
     * a cache key was provided to [nativeVlmGenerateStream]. On hit, the ViT
     * forward pass is skipped — vlmEncodeMs in the subsequent
     * [onVlmStageMetrics] event will be ~0.
     *
     * @param hit     true → cached embeddings reused; false → encoder ran fresh
     * @param nTokens Number of image embedding tokens
     * @param nEmbd   Per-token embedding dimension (`llama_model_n_embd_inp`)
     */
    fun onVlmCacheStatus(hit: Boolean, nTokens: Int, nEmbd: Int) {}

    /**
     * VLM-KV cache hit/miss. Fired once per VLM call when a vlmKvKey was
     * supplied. On hit, the LLM context state captured at the post-image
     * boundary is restored — BOTH the vision encoder AND the ~9s image-prefill
     * llama_decode are skipped, taking TTFT from ~10s to ~hundreds of ms.
     *
     * @param hit     true → cached state restored; false → fresh decode path
     * @param nTokens Number of tokens in the restored prefix (n_past)
     */
    fun onVlmKvCacheStatus(hit: Boolean, nTokens: Int) {}
}
