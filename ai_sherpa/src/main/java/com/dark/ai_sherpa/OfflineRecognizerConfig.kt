// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * Default thread count for ONNX Runtime sessions. Coerced to the smaller of
 * available cores and 4 — beyond 4 threads, sherpa-onnx's intra-op parallelism
 * starts hitting diminishing returns on most mobile SoCs (big.LITTLE thrashing,
 * shared L2 contention) and on phones the OS will deschedule us anyway.
 *
 * Consumers can override per-config; this is just the default.
 */
internal val defaultNumThreads: Int =
    Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

/**
 * Mel-spectrogram feature extraction settings. Defaults match the sherpa-onnx
 * shipping models — 16 kHz / 80-bin filterbank — and should not normally
 * be changed unless the model card specifies otherwise.
 *
 * @property sampleRate Audio sample rate in Hz. Must match what the model was
 *   trained on; [OfflineStream.acceptWaveform] resamples if you feed a
 *   different rate, but accuracy degrades.
 * @property featureDim Number of filterbank bins. 80 for most modern models.
 * @property dither Additive noise during feature extraction. 0 disables.
 */
data class FeatureConfig(
    val sampleRate: Int = 16_000,
    val featureDim: Int = 80,
    val dither: Float = 0.0f,
) {
    init {
        require(sampleRate > 0) { "sampleRate must be > 0" }
        require(featureDim > 0) { "featureDim must be > 0" }
    }
}

/**
 * Homophone replacement using FST-based rules. Empty values disable.
 */
data class HomophoneReplacerConfig(
    val lexicon: String = "",
    val ruleFsts: String = "",
)

/** Paths to encoder/decoder/joiner ONNX files for transducer ASR models. */
data class OfflineTransducerModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val joiner: String = "",
)

/** Paraformer single-file model. */
data class OfflineParaformerModelConfig(val model: String = "")

/** NeMo CTC model. */
data class OfflineNemoEncDecCtcModelConfig(val model: String = "")

/**
 * Whisper encoder/decoder pair.
 *
 * @property language ISO-639-1 source language code (e.g. `"en"`, `"es"`).
 *   Use `"auto"` to enable Whisper's built-in language detection.
 * @property task Either `"transcribe"` (output source language) or
 *   `"translate"` (translate to English).
 * @property tailPaddings Tail padding in 80ms frames added to short
 *   utterances. `-1` lets the model pick.
 */
data class OfflineWhisperModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val language: String = "en",
    val task: String = "transcribe",
    val tailPaddings: Int = -1,
)

/** TDNN single-file model. */
data class OfflineTdnnModelConfig(val model: String = "")

/** NeMo transducer (encoder + decoder + joiner). */
data class OfflineNemoTransducerModelConfig(
    val encoder: String = "",
    val decoder: String = "",
    val joiner: String = "",
)

/**
 * Container for every supported offline model family. Set exactly one of
 * [transducer], [paraformer], [nemoCtc], [whisper], [tdnn], [nemoTransducer];
 * leave the others at their defaults. The recognizer detects which family
 * is in use by checking which model paths are non-empty.
 *
 * @property numThreads ONNX Runtime intra-op thread count. Defaults to
 *   [defaultNumThreads]. Clamped to `>= 1` both in Kotlin and again in C++.
 * @property provider Execution provider. `"cpu"` is the only universally
 *   supported value on Android. `"nnapi"` may work for some VITS/Whisper
 *   variants but often falls back to CPU for unsupported ops.
 * @property modelType Override for model type detection. Leave blank unless
 *   the recognizer fails to auto-detect.
 * @property modelingUnit Override for tokenization unit (`"cjkchar"`,
 *   `"bpe"`, `"cjkchar+bpe"`). Leave blank for default.
 * @property bpeVocab Path to a BPE vocab file when [modelingUnit] needs it.
 */
data class OfflineModelConfig(
    val transducer: OfflineTransducerModelConfig = OfflineTransducerModelConfig(),
    val paraformer: OfflineParaformerModelConfig = OfflineParaformerModelConfig(),
    val nemoCtc: OfflineNemoEncDecCtcModelConfig = OfflineNemoEncDecCtcModelConfig(),
    val whisper: OfflineWhisperModelConfig = OfflineWhisperModelConfig(),
    val tdnn: OfflineTdnnModelConfig = OfflineTdnnModelConfig(),
    val nemoTransducer: OfflineNemoTransducerModelConfig = OfflineNemoTransducerModelConfig(),
    val tokens: String = "",
    val numThreads: Int = defaultNumThreads,
    val debug: Boolean = false,
    val provider: String = "cpu",
    val modelType: String = "",
    val modelingUnit: String = "",
    val bpeVocab: String = "",
) {
    init {
        require(numThreads >= 1) { "numThreads must be >= 1, got $numThreads" }
    }
}

/** External LM rescoring (transducer-only). [scale] = 0 disables LM influence. */
data class OfflineLMConfig(
    val model: String = "",
    val scale: Float = 0.5f,
)

/**
 * Top-level offline ASR recognizer config.
 *
 * @property featConfig Feature extraction settings (rarely needs changing).
 * @property modelConfig Which model family to load and its threading / provider.
 * @property lmConfig Optional external LM for transducer rescoring.
 * @property hr Optional homophone-replacement post-processing.
 * @property decodingMethod `"greedy_search"` or `"modified_beam_search"`.
 *   Beam search is more accurate but ~2-4x slower.
 * @property maxActivePaths Beam width for `modified_beam_search`. Ignored
 *   for greedy. Clamped to `>= 1`.
 * @property hotwordsFile Optional file with one phrase per line to boost.
 * @property hotwordsScore LM score boost for hotword matches.
 * @property ruleFsts Comma-separated FST files for text normalization.
 * @property ruleFars FAR archives for [ruleFsts].
 * @property blankPenalty Penalty added to the blank token logit. Positive
 *   values discourage blanks, useful when the model under-emits text.
 */
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
    val blankPenalty: Float = 0.0f,
) {
    init {
        require(maxActivePaths >= 1) { "maxActivePaths must be >= 1, got $maxActivePaths" }
    }
}
