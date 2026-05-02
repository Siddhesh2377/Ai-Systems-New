package com.dark.gguf_lib.models

/**
 * Sealed class representing events during text generation.
 * Emitted through Flow-based generation APIs.
 */
sealed class GenerationEvent {
    data class Token(val text: String) : GenerationEvent()
    data class ToolCall(val name: String, val argsJson: String) : GenerationEvent()
    data object Done : GenerationEvent()
    data class Error(val message: String) : GenerationEvent()
    data class Metrics(val metrics: DecodingMetrics) : GenerationEvent()
    data class Progress(val progress: Float) : GenerationEvent()

    /**
     * VLM prompt-eval stage timings. Emitted once per VLM generate call, after
     * the image has been encoded and decoded but before the first generated token.
     *
     * - [vlmEncodeMs]: vision/audio encoder wall time
     * - [vlmDecodeMs]: llama_decode wall time for image embeddings + interleaved text
     * - [imageTokens]: number of image embedding tokens fed into the LLM
     */
    data class VlmStageMetrics(
        val vlmEncodeMs: Float,
        val vlmDecodeMs: Float,
        val imageTokens: Int
    ) : GenerationEvent()
}
