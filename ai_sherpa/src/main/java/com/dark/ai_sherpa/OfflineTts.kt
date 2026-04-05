// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

class OfflineTts private constructor(private var ptr: Long) : AutoCloseable {

    val sampleRate: Int get() = getSampleRate(ptr)
    val numSpeakers: Int get() = getNumSpeakers(ptr)

    fun generate(text: String, sid: Int = 0, speed: Float = 1.0f): GeneratedAudio =
        generate(ptr, text, sid, speed)

    override fun close() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0L
        }
    }

    private external fun getSampleRate(ptr: Long): Int
    private external fun getNumSpeakers(ptr: Long): Int
    private external fun generate(ptr: Long, text: String, sid: Int, speed: Float): GeneratedAudio
    private external fun delete(ptr: Long)

    companion object {
        init { SherpaLib.init() }

        fun fromFile(config: OfflineTtsConfig): OfflineTts {
            val p = newFromFile(config)
            check(p != 0L) { "Failed to create OfflineTts" }
            return OfflineTts(p)
        }

        @JvmStatic private external fun newFromFile(config: OfflineTtsConfig): Long
    }
}
