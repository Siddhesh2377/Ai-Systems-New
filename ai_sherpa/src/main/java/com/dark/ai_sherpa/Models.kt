// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * ASR result returned by [OfflineRecognizer.getResult].
 *
 * @property text Concatenated transcript with model-default spacing/punctuation.
 * @property tokens Per-token strings in decode order. Same length as [timestamps].
 * @property timestamps Per-token start times in seconds, relative to the start
 *   of the audio fed via [OfflineStream.acceptWaveform]. Empty for models that
 *   don't emit timestamps (e.g. Whisper without word-level alignment).
 */
class OfflineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is OfflineRecognizerResult) return false
        return text == other.text &&
            tokens.contentEquals(other.tokens) &&
            timestamps.contentEquals(other.timestamps)
    }

    override fun hashCode(): Int {
        var h = text.hashCode()
        h = 31 * h + tokens.contentHashCode()
        h = 31 * h + timestamps.contentHashCode()
        return h
    }

    override fun toString(): String =
        "OfflineRecognizerResult(text='$text', tokens=${tokens.size}, timestamps=${timestamps.size})"
}

/**
 * TTS output returned by [OfflineTts.generate].
 *
 * @property samples Mono PCM in the range `[-1.0, 1.0]`.
 * @property sampleRate Output sample rate in Hz, defined by the underlying model.
 *   Use this rather than assuming a fixed rate — Kokoro returns 24000, VITS
 *   variants return 16000–22050 depending on the checkpoint.
 */
class GeneratedAudio(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GeneratedAudio) return false
        return sampleRate == other.sampleRate && samples.contentEquals(other.samples)
    }

    override fun hashCode(): Int = 31 * samples.contentHashCode() + sampleRate

    override fun toString(): String =
        "GeneratedAudio(samples=${samples.size}, sampleRate=$sampleRate)"
}
