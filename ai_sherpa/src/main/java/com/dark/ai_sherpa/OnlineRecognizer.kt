// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

import android.content.res.AssetManager

class OnlineRecognizer private constructor(private var ptr: Long) : AutoCloseable {

    val sampleRate: Int get() = getSampleRate(ptr)

    fun createStream(hotwords: String = ""): OnlineStream {
        val p = createStream(ptr, hotwords)
        check(p != 0L) { "Failed to create stream" }
        return OnlineStream(p)
    }

    fun decode(stream: OnlineStream) = decode(ptr, stream.ptr)

    fun decodeStreams(streams: Array<OnlineStream>) {
        decodeStreams(ptr, LongArray(streams.size) { streams[it].ptr })
    }

    fun isReady(stream: OnlineStream) = isReady(ptr, stream.ptr)
    fun isEndpoint(stream: OnlineStream) = isEndpoint(ptr, stream.ptr)
    fun reset(stream: OnlineStream) = reset(ptr, stream.ptr)
    fun getResult(stream: OnlineStream): OnlineRecognizerResult = getResult(ptr, stream.ptr)

    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun getSampleRate(ptr: Long): Int
    private external fun createStream(ptr: Long, hotwords: String): Long
    private external fun decode(ptr: Long, streamPtr: Long)
    private external fun decodeStreams(ptr: Long, streamPtrs: LongArray)
    private external fun isReady(ptr: Long, streamPtr: Long): Boolean
    private external fun isEndpoint(ptr: Long, streamPtr: Long): Boolean
    private external fun reset(ptr: Long, streamPtr: Long)
    private external fun getResult(ptr: Long, streamPtr: Long): OnlineRecognizerResult
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        fun fromFile(config: OnlineRecognizerConfig): OnlineRecognizer {
            val p = newFromFile(config)
            check(p != 0L) { "Failed to create OnlineRecognizer" }
            return OnlineRecognizer(p)
        }

        fun fromAsset(assetManager: AssetManager, config: OnlineRecognizerConfig): OnlineRecognizer {
            val p = newFromAsset(assetManager, config)
            check(p != 0L) { "Failed to create OnlineRecognizer from assets" }
            return OnlineRecognizer(p)
        }

        @JvmStatic private external fun newFromFile(config: OnlineRecognizerConfig): Long
        @JvmStatic private external fun newFromAsset(assetManager: AssetManager, config: OnlineRecognizerConfig): Long
    }
}

class OnlineStream internal constructor(internal var ptr: Long) : AutoCloseable {

    fun acceptWaveform(sampleRate: Int, samples: FloatArray) = acceptWaveform(ptr, sampleRate, samples)
    fun inputFinished() = inputFinished(ptr)

    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun acceptWaveform(ptr: Long, sampleRate: Int, samples: FloatArray)
    private external fun inputFinished(ptr: Long)
    private external fun delete(ptr: Long)
}
