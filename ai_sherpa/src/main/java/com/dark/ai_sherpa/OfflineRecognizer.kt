// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * Non-streaming (offline) speech recognizer. Wraps a sherpa-onnx
 * `OfflineRecognizer` and its `OfflineStream` lifecycle.
 *
 * Memory: each loaded recognizer holds the ONNX session(s) for its model
 * family — typically 50–500 MB depending on quantization. Streams are
 * cheap (<1 MB) but each new audio segment should use a fresh stream;
 * reusing a stream after [getResult] returns stale state.
 *
 * Threading: [fromFile], [decode] and [getResult] run synchronously and
 * may block for seconds (loading) or hundreds of ms (decoding). Call them
 * from a background dispatcher (e.g. `Dispatchers.IO`).
 *
 * Lifecycle: implements [AutoCloseable] — wrap with `use { ... }` or call
 * [close] manually. Forgetting to close leaks the native session. The
 * Kotlin object is safe to call after [close] (subsequent native calls
 * throw `NullPointerException` from the JNI guard).
 */
class OfflineRecognizer private constructor(@Volatile private var ptr: Long) : AutoCloseable {

    /**
     * Allocates a fresh decoding stream for one utterance. Caller owns the
     * returned stream and must close it.
     */
    fun createStream(): OfflineStream {
        check(ptr != 0L) { "OfflineRecognizer is closed" }
        val p = createStream(ptr)
        check(p != 0L) { "Failed to create stream" }
        return OfflineStream(p)
    }

    /** Runs the model over [stream]'s buffered audio. Blocking. */
    fun decode(stream: OfflineStream) {
        check(ptr != 0L) { "OfflineRecognizer is closed" }
        decode(ptr, stream.ptr)
    }

    /**
     * Reads the decoded transcript out of [stream]. Must be called after
     * [decode]. Returns `null` only if the native side reported a fatal error
     * (rare; usually the call throws instead).
     */
    fun getResult(stream: OfflineStream): OfflineRecognizerResult {
        check(ptr != 0L) { "OfflineRecognizer is closed" }
        return getResult(ptr, stream.ptr)
            ?: error("OfflineRecognizer.getResult returned null")
    }

    /** Releases the native session. Safe to call multiple times. */
    override fun close() {
        val p = ptr
        if (p != 0L) {
            ptr = 0L
            delete(p)
        }
    }

    private external fun createStream(ptr: Long): Long
    private external fun decode(ptr: Long, streamPtr: Long)
    private external fun getResult(ptr: Long, streamPtr: Long): OfflineRecognizerResult?
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        /**
         * Loads a recognizer from on-disk model files described by [config].
         * Blocks for the duration of the ONNX session init (1–10s typical).
         * Throws `IllegalStateException` if the underlying model fails to load —
         * the structured reason is emitted to the `:tn_security` sink.
         */
        fun fromFile(config: OfflineRecognizerConfig): OfflineRecognizer {
            val p = newFromFile(config)
            check(p != 0L) { "Failed to create OfflineRecognizer" }
            return OfflineRecognizer(p)
        }

        @JvmStatic private external fun newFromFile(config: OfflineRecognizerConfig): Long
    }
}

/**
 * One-shot decoding stream owned by an [OfflineRecognizer]. Feed audio with
 * [acceptWaveform], then have the parent recognizer [OfflineRecognizer.decode]
 * it. Each stream is good for exactly one utterance.
 */
class OfflineStream internal constructor(@Volatile internal var ptr: Long) : AutoCloseable {

    /**
     * Appends `samples` (mono PCM, range `[-1, 1]`) at [sampleRate] Hz.
     * Sherpa-onnx resamples internally if [sampleRate] differs from the
     * model's expected rate, but accuracy degrades — match if you can.
     *
     * The sample buffer is read synchronously and not retained; the caller
     * may reuse or free it immediately after this returns.
     */
    fun acceptWaveform(sampleRate: Int, samples: FloatArray) {
        check(ptr != 0L) { "OfflineStream is closed" }
        require(sampleRate > 0) { "sampleRate must be > 0" }
        if (samples.isEmpty()) return
        acceptWaveform(ptr, sampleRate, samples)
    }

    override fun close() {
        val p = ptr
        if (p != 0L) {
            ptr = 0L
            delete(p)
        }
    }

    private external fun acceptWaveform(ptr: Long, sampleRate: Int, samples: FloatArray)
    private external fun delete(ptr: Long)
}
