package com.dark.backend_plugin_api.model

data class GenerationMetrics(
    val tokensPerSecond: Float,
    val timeToFirstTokenMs: Long,
    val totalTokens: Int,
    val promptTokens: Int
)
