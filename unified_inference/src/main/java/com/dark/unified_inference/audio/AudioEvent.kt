package com.dark.unified_inference.audio

sealed class AudioEvent {
    data class SynthesisStarted(val textLength: Int, val chunkCount: Int) : AudioEvent()
    data class ChunkProgress(val chunkIndex: Int, val totalChunks: Int) : AudioEvent()
    data class AudioReady(val pcmData: ByteArray, val sampleRate: Int) : AudioEvent()
    data class Error(val message: String) : AudioEvent()
}
