// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

class VoiceActivityDetector private constructor(private var ptr: Long) : AutoCloseable {

    fun acceptWaveform(samples: FloatArray) = acceptWaveform(ptr, samples)
    fun isEmpty(): Boolean = empty(ptr)
    fun pop() = pop(ptr)
    fun clear() = clear(ptr)
    fun front(): SpeechSegment = front(ptr)
    fun isSpeechDetected(): Boolean = isSpeechDetected(ptr)
    fun reset() = reset(ptr)
    fun flush() = flush(ptr)
    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun acceptWaveform(ptr: Long, samples: FloatArray)
    private external fun empty(ptr: Long): Boolean
    private external fun pop(ptr: Long)
    private external fun clear(ptr: Long)
    private external fun front(ptr: Long): SpeechSegment
    private external fun isSpeechDetected(ptr: Long): Boolean
    private external fun reset(ptr: Long)
    private external fun flush(ptr: Long)
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        fun fromFile(config: VadModelConfig, bufferSizeInSeconds: Int = 30): VoiceActivityDetector {
            val p = newFromFile(config, bufferSizeInSeconds)
            check(p != 0L) { "Failed to create VoiceActivityDetector" }
            return VoiceActivityDetector(p)
        }

        @JvmStatic private external fun newFromFile(config: VadModelConfig, bufferSizeInSeconds: Int): Long
    }
}
