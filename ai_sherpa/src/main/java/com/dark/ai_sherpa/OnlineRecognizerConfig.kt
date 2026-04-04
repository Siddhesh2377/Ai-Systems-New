// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

data class FeatureConfig(
    val sampleRate: Int = 16000,
    val featureDim: Int = 80,
    val dither: Float = 0.0f
)

data class EndpointRule(
    val mustContainNonSilence: Boolean = false,
    val minTrailingSilence: Float = 2.0f,
    val minUtteranceLength: Float = 0.0f
)

data class EndpointConfig(
    val rule1: EndpointRule = EndpointRule(false, 2.4f, 0f),
    val rule2: EndpointRule = EndpointRule(true, 1.2f, 0f),
    val rule3: EndpointRule = EndpointRule(false, 0f, 20f)
)

data class OnlineTransducerModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val joiner: String = ""
)

data class OnlineParaformerModelConfig(
    val encoder: String = "",
    val decoder: String = ""
)

data class OnlineZipformer2CtcModelConfig(val model: String = "")

data class OnlineNeMoCtcModelConfig(val model: String = "")

data class OnlineModelConfig(
    val transducer: OnlineTransducerModelConfig = OnlineTransducerModelConfig(),
    val paraformer: OnlineParaformerModelConfig = OnlineParaformerModelConfig(),
    val zipformer2Ctc: OnlineZipformer2CtcModelConfig = OnlineZipformer2CtcModelConfig(),
    val neMoCtc: OnlineNeMoCtcModelConfig = OnlineNeMoCtcModelConfig(),
    val tokens: String = "",
    val numThreads: Int = 1,
    val debug: Boolean = false,
    val provider: String = "cpu",
    val modelType: String = "",
    val modelingUnit: String = "",
    val bpeVocab: String = ""
)

data class OnlineLMConfig(val model: String = "", val scale: Float = 0.5f)

data class OnlineCtcFstDecoderConfig(val graph: String = "", val maxActive: Int = 3000)

data class HomophoneReplacerConfig(val lexicon: String = "", val ruleFsts: String = "")

data class OnlineRecognizerConfig(
    val featConfig: FeatureConfig = FeatureConfig(),
    val modelConfig: OnlineModelConfig = OnlineModelConfig(),
    val lmConfig: OnlineLMConfig = OnlineLMConfig(),
    val endpointConfig: EndpointConfig = EndpointConfig(),
    val ctcFstDecoderConfig: OnlineCtcFstDecoderConfig = OnlineCtcFstDecoderConfig(),
    val hr: HomophoneReplacerConfig = HomophoneReplacerConfig(),
    val decodingMethod: String = "greedy_search",
    val maxActivePaths: Int = 4,
    val hotwordsFile: String = "",
    val hotwordsScore: Float = 1.5f,
    val ruleFsts: String = "",
    val ruleFars: String = "",
    val blankPenalty: Float = 0.0f,
    val enableEndpoint: Boolean = true
)
