package com.dark.unified_inference.text

sealed class TextEvent {
    data class Token(val text: String) : TextEvent()
    data class ToolCall(val name: String, val argsJson: String) : TextEvent()
    data class Metrics(
        val tokensPerSecond: Float,
        val timeToFirstTokenMs: Float,
        val totalTimeMs: Float,
        val tokensEvaluated: Int,
        val tokensPredicted: Int
    ) : TextEvent()
    data class Progress(val progress: Float) : TextEvent()
    data object Done : TextEvent()
    data class Error(val message: String) : TextEvent()
}
