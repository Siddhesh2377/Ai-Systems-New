package com.dark.backend_plugin_api.callback

interface TTSCallback {
    fun onAudioChunk(pcmData: FloatArray, sampleRate: Int)
    fun onComplete(totalDurationMs: Long)
    fun onError(message: String)
}
