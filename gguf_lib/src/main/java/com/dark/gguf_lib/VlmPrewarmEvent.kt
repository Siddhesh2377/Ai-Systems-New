package com.dark.gguf_lib

/**
 * Events emitted by [GGMLEngine.precomputeVlmKvStateFlow] — the streaming
 * counterpart of the per-stage [com.dark.gguf_lib.models.VlmPrewarmCallback].
 * Use these to drive UI like "Encoding tile 3/5" or "Decoding tile 3/5 (3.2 s)".
 *
 * Order on success:
 * ```
 * Started(totalChunks)
 *  ├─ ChunkStart(0, total, isImage=…)
 *  ├─ ChunkDone (0, total, encodeMs, decodeMs)
 *  ├─ … repeated per chunk up to and including the last image …
 * StateStored(blobBytes, nTokens)
 * Done(totalMs, cached)
 * ```
 */
sealed class VlmPrewarmEvent {
    data class Started(val totalChunks: Int) : VlmPrewarmEvent()
    data class ChunkStart(val index: Int, val total: Int, val isImage: Boolean) : VlmPrewarmEvent()
    data class ChunkDone(
        val index: Int,
        val total: Int,
        val encodeMs: Float,
        val decodeMs: Float,
    ) : VlmPrewarmEvent()
    data class StateStored(val blobBytes: Long, val nTokens: Int) : VlmPrewarmEvent()
    data class Done(val totalMs: Long, val cached: Boolean) : VlmPrewarmEvent()
    data class Error(val message: String) : VlmPrewarmEvent()
}
