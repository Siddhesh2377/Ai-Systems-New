// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * VITS-family acoustic model config (e.g. piper, vits-icefall).
 *
 * @property model ONNX model path.
 * @property lexicon Optional lexicon file for grapheme-to-phoneme.
 * @property tokens Token vocabulary file. Required.
 * @property dataDir espeak-ng data directory for languages that need it.
 * @property dictDir Pinyin/jieba dict directory (Chinese models).
 * @property noiseScale Stochastic duration predictor noise. Default 0.667.
 * @property noiseScaleW Stochastic flow noise. Default 0.8.
 * @property lengthScale Inverse speaking rate. >1 slows down, <1 speeds up.
 *   Used by models that don't expose a separate `speed` field; otherwise
 *   prefer the `speed` arg to [OfflineTts.generate].
 */
data class OfflineTtsVitsModelConfig(
    val model: String = "",
    val lexicon: String = "",
    val tokens: String = "",
    val dataDir: String = "",
    val dictDir: String = "",
    val noiseScale: Float = 0.667f,
    val noiseScaleW: Float = 0.8f,
    val lengthScale: Float = 1.0f,
)

/**
 * Kokoro TTS model config. Kokoro requires a separate `voices.bin` and
 * espeak-ng data directory.
 *
 * @property voices Path to `voices.bin`.
 * @property lengthScale Inverse speaking rate; >1 slower, <1 faster.
 */
data class OfflineTtsKokoroModelConfig(
    val model: String = "",
    val voices: String = "",
    val tokens: String = "",
    val dataDir: String = "",
    val dictDir: String = "",
    val lengthScale: Float = 1.0f,
)

/**
 * Container for the supported TTS model families. Set exactly one of [vits]
 * or [kokoro] — leaving both empty makes [OfflineTts.fromFile] throw.
 *
 * @property numThreads ONNX Runtime intra-op thread count. Defaults to
 *   [defaultNumThreads]. Clamped to `>= 1`.
 * @property provider Execution provider. `"cpu"` is recommended on Android.
 */
data class OfflineTtsModelConfig(
    val vits: OfflineTtsVitsModelConfig = OfflineTtsVitsModelConfig(),
    val kokoro: OfflineTtsKokoroModelConfig = OfflineTtsKokoroModelConfig(),
    val numThreads: Int = defaultNumThreads,
    val debug: Boolean = false,
    val provider: String = "cpu",
) {
    init {
        require(numThreads >= 1) { "numThreads must be >= 1, got $numThreads" }
    }
}

/**
 * Top-level offline TTS config.
 *
 * @property model The acoustic model and its threading.
 * @property ruleFsts Comma-separated FST files for text normalization.
 * @property ruleFars FAR archives for [ruleFsts].
 * @property maxNumSentences Maximum sentences batched per call. Higher
 *   values increase latency on the first audio sample but reduce per-token
 *   overhead. Clamped to `>= 1`.
 */
data class OfflineTtsConfig(
    val model: OfflineTtsModelConfig = OfflineTtsModelConfig(),
    val ruleFsts: String = "",
    val ruleFars: String = "",
    val maxNumSentences: Int = 1,
) {
    init {
        require(maxNumSentences >= 1) { "maxNumSentences must be >= 1, got $maxNumSentences" }
    }
}
