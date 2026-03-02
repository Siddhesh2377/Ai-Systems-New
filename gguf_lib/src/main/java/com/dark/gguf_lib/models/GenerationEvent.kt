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
}
