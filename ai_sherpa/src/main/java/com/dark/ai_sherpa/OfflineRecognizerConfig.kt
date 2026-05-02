// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

data class OfflineTransducerModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val joiner: String = ""
)

data class OfflineParaformerModelConfig(val model: String = "")

data class OfflineNemoEncDecCtcModelConfig(val model: String = "")

data class OfflineWhisperModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val language: String = "en",
    val task: String = "transcribe",
    val tailPaddings: Int = -1
)

data class OfflineTdnnModelConfig(val model: String = "")

data class OfflineNemoTransducerModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val joiner: String = ""
)

data class OfflineModelConfig(
    val transducer: OfflineTransducerModelConfig = OfflineTransducerModelConfig(),
    val paraformer: OfflineParaformerModelConfig = OfflineParaformerModelConfig(),
    val nemoCtc: OfflineNemoEncDecCtcModelConfig = OfflineNemoEncDecCtcModelConfig(),
    val whisper: OfflineWhisperModelConfig = OfflineWhisperModelConfig(),
    val tdnn: OfflineTdnnModelConfig = OfflineTdnnModelConfig(),
    val nemoTransducer: OfflineNemoTransducerModelConfig = OfflineNemoTransducerModelConfig(),
    val tokens: String = "",
    val numThreads: Int = 1,
    val debug: Boolean = false,
    val provider: String = "cpu",
    val modelType: String = "",
    val modelingUnit: String = "",
    val bpeVocab: String = ""
)

data class OfflineLMConfig(val model: String = "", val scale: Float = 0.5f)

data class OfflineRecognizerConfig(
    val featConfig: FeatureConfig = FeatureConfig(),
    val modelConfig: OfflineModelConfig = OfflineModelConfig(),
    val lmConfig: OfflineLMConfig = OfflineLMConfig(),
    val hr: HomophoneReplacerConfig = HomophoneReplacerConfig(),
    val decodingMethod: String = "greedy_search",
    val maxActivePaths: Int = 4,
    val hotwordsFile: String = "",
    val hotwordsScore: Float = 1.5f,
    val ruleFsts: String = "",
    val ruleFars: String = "",
    val blankPenalty: Float = 0.0f
)
