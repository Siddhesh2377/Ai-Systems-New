package com.mp.ai_gguf.models

import androidx.annotation.Keep
import kotlinx.serialization.Serializable

/**
 * Decoding performance metrics with memory tracking
 *
 * Optimizations for low-end devices:
 * - Memory usage tracking helps prevent OOM
 * - Tokens per second helps gauge device capability
 */
@Serializable
@Keep
data class DecodingMetrics @JvmOverloads constructor(
    val totalTokens: Int = 0,
    val promptTokens: Int = 0,
    val generatedTokens: Int = 0,
    val tokensPerSecond: Float = 0f,
    val timeToFirstToken: Long = 0L,
    val totalTimeMs: Long = 0L
) {
    // Memory metrics (can be set separately if tracked)
    var modelSizeMB: Float = 0f
        private set
    var contextSizeMB: Float = 0f
        private set
    var peakMemoryMB: Float = 0f
        private set
    var memoryUsagePercent: Float = 0f
        private set

    /**
     * Create metrics with memory info
     */
    fun withMemoryInfo(
        modelMB: Float,
        contextMB: Float,
        peakMB: Float,
        usagePercent: Float
    ): DecodingMetrics = this.also {
        it.modelSizeMB = modelMB
        it.contextSizeMB = contextMB
        it.peakMemoryMB = peakMB
        it.memoryUsagePercent = usagePercent
    }
    /**
     * Check if performance is acceptable for interactive use
     * (> 5 tokens/second is generally acceptable)
     */
    fun isInteractivePerformance(): Boolean = tokensPerSecond >= 5f

    /**
     * Check if memory usage is concerning (> 80%)
     */
    fun isMemoryCritical(): Boolean = memoryUsagePercent > 80f

    /**
     * Get a human-readable summary
     */
    fun summary(): String = buildString {
        append("$generatedTokens tokens @ ${String.format("%.1f", tokensPerSecond)} t/s")
        if (timeToFirstToken > 0) {
            append(", TTFT: ${timeToFirstToken}ms")
        }
        if (memoryUsagePercent > 0) {
            append(", mem: ${String.format("%.0f", memoryUsagePercent)}%")
        }
    }
}

/**
 * Callback interface for streaming generation
 *
 * All callbacks are invoked on the inference thread.
 * For UI updates, dispatch to the main thread.
 */
@Keep
interface StreamCallback {
    /**
     * Called for each generated token
     * @param token The decoded text for this token
     */
    fun onToken(token: String)

    /**
     * Called when a tool call is detected
     * @param name The name of the tool being called
     * @param argsJson The arguments as a JSON string
     */
    fun onToolCall(name: String, argsJson: String)

    /**
     * Called when generation completes successfully
     */
    fun onDone()

    /**
     * Called when an error occurs
     * @param message Error description
     */
    fun onError(message: String)

    /**
     * Called with performance metrics after generation
     * @param metrics Performance and memory metrics
     */
    fun onMetrics(metrics: DecodingMetrics) {}
}

/**
 * Simple implementation of StreamCallback that collects tokens
 */
@Keep
open class SimpleStreamCallback : StreamCallback {
    private val tokens = StringBuilder()
    private var toolCallName: String? = null
    private var toolCallArgs: String? = null
    private var errorMessage: String? = null
    private var metrics: DecodingMetrics? = null

    override fun onToken(token: String) {
        tokens.append(token)
    }

    override fun onToolCall(name: String, argsJson: String) {
        toolCallName = name
        toolCallArgs = argsJson
    }

    override fun onDone() {
        // Override in subclass if needed
    }

    override fun onError(message: String) {
        errorMessage = message
    }

    override fun onMetrics(metrics: DecodingMetrics) {
        this.metrics = metrics
    }

    fun getResult(): String = tokens.toString()
    fun getToolCall(): Pair<String, String>? =
        toolCallName?.let { name -> toolCallArgs?.let { args -> name to args } }
    fun getError(): String? = errorMessage
    fun getMetrics(): DecodingMetrics? = metrics
    fun hasToolCall(): Boolean = toolCallName != null
    fun hasError(): Boolean = errorMessage != null
}