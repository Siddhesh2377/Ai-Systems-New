package com.dark.demon_system.ui.vlm

import com.dark.gguf_lib.models.DecodingMetrics

/** UI state for the VLM screen. One sealed hierarchy across the full lifecycle. */
sealed interface VlmState {

    /** Nothing in flight. Model may or may not be downloaded — UI checks via [VlmViewModel.modelDownloaded]. */
    data object Idle : VlmState

    data class Downloading(
        val fileIndex: Int,
        val fileName: String,
        val bytesDownloaded: Long,
        val totalBytes: Long,
        val bytesPerSecond: Long,
        val overallPct: Float,
    ) : VlmState

    data class DownloadFailed(val message: String) : VlmState

    data object LoadingModel : VlmState
    data object LoadingProjector : VlmState

    data object Ready : VlmState

    /** Tokens are streaming in. [text] is the current concatenation. */
    data class Generating(
        val text: String,
        val vlmEncodeMs: Float? = null,
        val vlmDecodeMs: Float? = null,
        val imageTokens: Int? = null,
        val progress: Float? = null,
        val vtCacheHit: Boolean? = null,
        val vlmKvCacheHit: Boolean? = null,
    ) : VlmState

    data class GenerationDone(
        val text: String,
        val metrics: DecodingMetrics?,
        val vlmEncodeMs: Float?,
        val vlmDecodeMs: Float?,
        val imageTokens: Int?,
        val vtCacheHit: Boolean?,
        val vlmKvCacheHit: Boolean?,
    ) : VlmState

    data class Error(val message: String) : VlmState
}

/**
 * Independent state for the background pre-warm pass that fires when a new
 * image is picked. Lives on its own flow so it can overlay any [VlmState]
 * (Ready, Generating, GenerationDone, …) without disturbing the main UI
 * lifecycle.
 */
sealed interface PrewarmState {
    data object Idle : PrewarmState

    /**
     * Pre-warm in flight. [stage] is the human-readable current operation
     * (e.g. "Encoding tile 3/5"); [chunkIndex]/[totalChunks] drive a determinate
     * progress bar.
     */
    data class InProgress(
        val startedAt: Long,
        val stage: String = "Tokenizing…",
        val chunkIndex: Int = 0,
        val totalChunks: Int = 0,
        val lastEncodeMs: Float? = null,
        val lastDecodeMs: Float? = null,
    ) : PrewarmState

    /** Pre-warm completed in [durationMs] ms; [cached] = true if VLM-KV stored. */
    data class Done(
        val durationMs: Long,
        val cached: Boolean,
        val totalChunks: Int = 0,
        val blobBytes: Long = 0,
        val nTokens: Int = 0,
    ) : PrewarmState

    data class Failed(val message: String) : PrewarmState
}
