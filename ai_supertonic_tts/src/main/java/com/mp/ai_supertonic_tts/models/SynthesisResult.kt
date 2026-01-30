package com.mp.ai_supertonic_tts.models

/**
 * Result of a text-to-speech synthesis operation.
 *
 * Contains the raw audio data and metadata about the synthesis.
 *
 * @param audioData Raw float32 audio samples in [-1.0, 1.0] range
 * @param sampleRate Sample rate in Hz (44100 for Supertonic v2)
 * @param channels Number of audio channels (1 = mono)
 * @param durationMs Audio duration in milliseconds
 * @param synthesisTimeMs Wall-clock time for synthesis in milliseconds
 */
data class SynthesisResult(
    val audioData: FloatArray,
    val sampleRate: Int,
    val channels: Int = 1,
    val durationMs: Long,
    val synthesisTimeMs: Long
) {
    /** Real-time factor: < 1.0 means faster than real-time */
    val realtimeFactor: Float
        get() = if (durationMs > 0) synthesisTimeMs.toFloat() / durationMs else 0f

    /** Number of audio samples */
    val sampleCount: Int get() = audioData.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SynthesisResult) return false
        return sampleRate == other.sampleRate &&
                channels == other.channels &&
                durationMs == other.durationMs &&
                audioData.contentEquals(other.audioData)
    }

    override fun hashCode(): Int {
        var result = audioData.contentHashCode()
        result = 31 * result + sampleRate
        result = 31 * result + channels
        return result
    }

    override fun toString(): String =
        "SynthesisResult(samples=${audioData.size}, rate=$sampleRate, duration=${durationMs}ms, synth=${synthesisTimeMs}ms, RTF=${"%.2f".format(realtimeFactor)})"
}
