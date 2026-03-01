package com.mp.ai_gguf.models

import androidx.annotation.Keep

@Keep
data class DecodingMetrics(
    val tokensPerSecond: Float = 0f,
    val timeToFirstTokenMs: Float = 0f,
    val totalTimeMs: Float = 0f,
    val tokensEvaluated: Int = 0,
    val tokensPredicted: Int = 0,
    val modelSizeMB: Float = 0f,
    val contextSizeMB: Float = 0f,
    val peakMemoryMB: Float = 0f,
    val memoryUsagePercent: Float = 0f
) {
    fun isInteractivePerformance(): Boolean = tokensPerSecond >= 5f
    fun isMemoryCritical(): Boolean = memoryUsagePercent > 80f

    fun summary(): String = buildString {
        append("$tokensPredicted tokens @ ${String.format("%.1f", tokensPerSecond)} t/s")
        if (timeToFirstTokenMs > 0) append(", TTFT: ${String.format("%.0f", timeToFirstTokenMs)}ms")
        if (memoryUsagePercent > 0) append(", mem: ${String.format("%.0f", memoryUsagePercent)}%")
    }
}

@Keep
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)

    fun onMetrics(
        tps: Float, ttftMs: Float, totalMs: Float,
        tokensEvaluated: Int, tokensPredicted: Int,
        modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float
    ) {}
}

@Keep
open class SimpleStreamCallback : StreamCallback {
    private val tokens = StringBuilder()
    private var toolCallName: String? = null
    private var toolCallArgs: String? = null
    private var errorMessage: String? = null
    private var _metrics: DecodingMetrics? = null

    override fun onToken(token: String) { tokens.append(token) }

    override fun onToolCall(name: String, argsJson: String) {
        toolCallName = name
        toolCallArgs = argsJson
    }

    override fun onDone() {}

    override fun onError(message: String) { errorMessage = message }

    override fun onMetrics(
        tps: Float, ttftMs: Float, totalMs: Float,
        tokensEvaluated: Int, tokensPredicted: Int,
        modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float
    ) {
        _metrics = DecodingMetrics(tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted,
            modelMB, ctxMB, peakMB, memPct)
    }

    fun getResult(): String = tokens.toString()
    fun getToolCall(): Pair<String, String>? =
        toolCallName?.let { n -> toolCallArgs?.let { a -> n to a } }
    fun getError(): String? = errorMessage
    fun getMetrics(): DecodingMetrics? = _metrics
    fun hasToolCall(): Boolean = toolCallName != null
    fun hasError(): Boolean = errorMessage != null
}
