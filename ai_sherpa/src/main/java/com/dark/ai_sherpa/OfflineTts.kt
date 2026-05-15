// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * Offline (non-streaming) text-to-speech synthesizer.
 *
 * Memory: each loaded TTS holds the acoustic-model ONNX session — typically
 * 30–200 MB. Single-speaker VITS variants are smaller; Kokoro is ~310 MB.
 *
 * Threading: [fromFile] and [generate] are synchronous and CPU-heavy.
 * Call them from a background dispatcher. Generation time scales linearly
 * with output length; expect 0.1–0.5x real-time on a modern phone CPU.
 *
 * Lifecycle: implements [AutoCloseable]; use `use { }` or call [close] manually.
 */
class OfflineTts private constructor(@Volatile private var ptr: Long) : AutoCloseable {

    /** Output sample rate in Hz, dictated by the loaded model. */
    val sampleRate: Int
        get() {
            check(ptr != 0L) { "OfflineTts is closed" }
            return getSampleRate(ptr)
        }

    /**
     * Number of speakers the model supports. `1` for single-speaker checkpoints.
     * Multi-speaker models accept `sid` in the range `0 until numSpeakers`.
     */
    val numSpeakers: Int
        get() {
            check(ptr != 0L) { "OfflineTts is closed" }
            return getNumSpeakers(ptr)
        }

    /**
     * Synthesizes [text] into mono PCM samples.
     *
     * @param text The text to speak. Non-empty. Models do their own
     *   sentence-splitting, but very long input increases peak memory.
     * @param sid Speaker ID for multi-speaker models. Ignored by single-speaker
     *   models. Must be in `0 until numSpeakers`.
     * @param speed Speaking-rate multiplier; 1.0 is the model's default,
     *   2.0 doubles the rate, 0.5 halves it. Range typically `0.5..2.0`.
     * @return [GeneratedAudio] with samples and the model's sample rate.
     * @throws IllegalStateException if generation fails.
     */
    fun generate(text: String, sid: Int = 0, speed: Float = 1.0f): GeneratedAudio {
        check(ptr != 0L) { "OfflineTts is closed" }
        require(text.isNotEmpty()) { "text is empty" }
        require(sid >= 0) { "sid must be >= 0, got $sid" }
        require(speed > 0f) { "speed must be > 0, got $speed" }
        return generate(ptr, text, sid, speed)
    }

    /** Releases the native session. Safe to call multiple times. */
    override fun close() {
        val p = ptr
        if (p != 0L) {
            ptr = 0L
            delete(p)
        }
    }

    private external fun getSampleRate(ptr: Long): Int
    private external fun getNumSpeakers(ptr: Long): Int
    private external fun generate(ptr: Long, text: String, sid: Int, speed: Float): GeneratedAudio
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        /**
         * Loads a TTS engine from on-disk model files described by [config].
         * Blocks during ONNX session init (1–10s typical for VITS, 5–20s for
         * Kokoro). Throws `IllegalStateException` if the model fails to load —
         * the structured reason is emitted to the `:tn_security` sink.
         */
        fun fromFile(config: OfflineTtsConfig): OfflineTts {
            val p = newFromFile(config)
            check(p != 0L) { "Failed to create OfflineTts" }
            return OfflineTts(p)
        }

        @JvmStatic private external fun newFromFile(config: OfflineTtsConfig): Long
    }
}
