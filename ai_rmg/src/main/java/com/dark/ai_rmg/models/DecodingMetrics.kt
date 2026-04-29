package com.dark.ai_rmg.models

data class DecodingMetrics(
    val tokensPerSecond: Float = 0f,
    val timeToFirstTokenMs: Float = 0f,
    val totalTimeMs: Float = 0f,
    val tokensEvaluated: Int = 0,
    val tokensPredicted: Int = 0
)
