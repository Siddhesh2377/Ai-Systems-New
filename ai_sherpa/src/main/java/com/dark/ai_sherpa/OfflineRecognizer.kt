// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

import android.content.res.AssetManager

class OfflineRecognizer private constructor(private var ptr: Long) : AutoCloseable {

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        check(p != 0L) { "Failed to create stream" }
        return OfflineStream(p)
    }

    fun decode(stream: OfflineStream) = decode(ptr, stream.ptr)

    fun getResult(stream: OfflineStream): OfflineRecognizerResult = getResult(ptr, stream.ptr)

    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun createStream(ptr: Long): Long
    private external fun decode(ptr: Long, streamPtr: Long)
    private external fun getResult(ptr: Long, streamPtr: Long): OfflineRecognizerResult
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        fun fromFile(config: OfflineRecognizerConfig): OfflineRecognizer {
            val p = newFromFile(config)
            check(p != 0L) { "Failed to create OfflineRecognizer" }
            return OfflineRecognizer(p)
        }

        fun fromAsset(assetManager: AssetManager, config: OfflineRecognizerConfig): OfflineRecognizer {
            val p = newFromAsset(assetManager, config)
            check(p != 0L) { "Failed to create OfflineRecognizer from assets" }
            return OfflineRecognizer(p)
        }

        @JvmStatic private external fun newFromFile(config: OfflineRecognizerConfig): Long
        @JvmStatic private external fun newFromAsset(assetManager: AssetManager, config: OfflineRecognizerConfig): Long
    }
}

class OfflineStream internal constructor(internal var ptr: Long) : AutoCloseable {

    fun acceptWaveform(sampleRate: Int, samples: FloatArray) = acceptWaveform(ptr, sampleRate, samples)

    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun acceptWaveform(ptr: Long, sampleRate: Int, samples: FloatArray)
    private external fun delete(ptr: Long)
}
