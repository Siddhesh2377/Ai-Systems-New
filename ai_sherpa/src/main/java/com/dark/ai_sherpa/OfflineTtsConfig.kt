// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

data class OfflineTtsVitsModelConfig(
    val model: String = "",
    val lexicon: String = "",
    val tokens: String = "",
    val dataDir: String = "",
    val dictDir: String = "",
    val noiseScale: Float = 0.667f,
    val noiseScaleW: Float = 0.8f,
    val lengthScale: Float = 1.0f
)

data class OfflineTtsKokoroModelConfig(
    val model: String = "",
    val voices: String = "",
    val tokens: String = "",
    val dataDir: String = "",
    val dictDir: String = "",
    val lengthScale: Float = 1.0f
)

data class OfflineTtsModelConfig(
    val vits: OfflineTtsVitsModelConfig = OfflineTtsVitsModelConfig(),
    val kokoro: OfflineTtsKokoroModelConfig = OfflineTtsKokoroModelConfig(),
    val numThreads: Int = 1,
    val debug: Boolean = false,
    val provider: String = "cpu"
)

data class OfflineTtsConfig(
    val model: OfflineTtsModelConfig = OfflineTtsModelConfig(),
    val ruleFsts: String = "",
    val ruleFars: String = "",
    val maxNumSentences: Int = 1
)
