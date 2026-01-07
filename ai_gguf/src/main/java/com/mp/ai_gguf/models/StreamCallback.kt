package com.mp.ai_gguf.models

import androidx.annotation.Keep
import kotlinx.serialization.Serializable

@Serializable
@Keep
data class DecodingMetrics(
    val totalTokens: Int = 0,
    val promptTokens: Int = 0,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Float = 0f,
    val timeToFirstToken: Long = 0L,
    val totalTimeMs: Long = 0L
)

@Keep
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
    fun onMetrics(metrics: DecodingMetrics) {}
}