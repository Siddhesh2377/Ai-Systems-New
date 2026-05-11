package com.dark.gguf_lib.models

/**
 * Per-stage callback for the VLM pre-warm pipeline. Fired from native code
 * during [com.dark.gguf_lib.GGUFNativeLib.nativeVlmPrecomputeKvState] so
 * host UIs can surface progress like "Encoding tile 3/5" or
 * "Decoding tile 3/5 (3.2 s)".
 *
 * Event order on a successful run:
 * ```
 * onStarted(totalChunks)
 *  ├─ onChunkStart(0, total, isImage=…)
 *  ├─ onChunkDone (0, total, encodeMs, decodeMs)
 *  ├─ … repeated per chunk up to and including the last image …
 * onStateStored(blobBytes, nTokens)
 * onDone(totalMs, cached)
 * ```
 *
 * On failure: any of the above followed by [onError]. Default no-ops let
 * implementers override only the events they care about.
 */
interface VlmPrewarmCallback {
    fun onStarted(totalChunks: Int) {}
    fun onChunkStart(index: Int, total: Int, isImage: Boolean) {}
    fun onChunkDone(index: Int, total: Int, encodeMs: Float, decodeMs: Float) {}
    fun onStateStored(blobBytes: Long, nTokens: Int) {}
    fun onDone(totalMs: Long, cached: Boolean) {}
    fun onError(message: String) {}
}
