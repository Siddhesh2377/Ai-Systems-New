// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

data class SileroVadModelConfig(
    val model: String = "",
    val threshold: Float = 0.5f,
    val minSilenceDuration: Float = 0.5f,
    val minSpeechDuration: Float = 0.25f,
    val maxSpeechDuration: Float = Float.MAX_VALUE,
    val windowSize: Int = 512
)

data class TenVadModelConfig(
    val model: String = "",
    val threshold: Float = 0.5f,
    val minSilenceDuration: Float = 0.5f,
    val minSpeechDuration: Float = 0.25f,
    val maxSpeechDuration: Float = Float.MAX_VALUE,
    val windowSize: Int = 256
)

data class VadModelConfig(
    val sileroVadModelConfig: SileroVadModelConfig = SileroVadModelConfig(),
    val tenVadModelConfig: TenVadModelConfig = TenVadModelConfig(),
    val sampleRate: Int = 16000,
    val numThreads: Int = 1,
    val provider: String = "cpu",
    val debug: Boolean = false
)
