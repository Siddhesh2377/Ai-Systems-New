package com.dark.ai_chatterbox

interface ChatterboxCallback {
    fun onSpeechTokenProgress(tokensGenerated: Int)
    fun onAudioReady(pcmData: ShortArray, sampleRate: Int)
    fun onError(message: String)
}
