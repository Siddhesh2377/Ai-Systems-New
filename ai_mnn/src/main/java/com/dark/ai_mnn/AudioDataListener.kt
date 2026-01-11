package com.dark.ai_mnn

interface AudioDataListener {
    fun onAudioData(data: FloatArray, isEnd: Boolean): Boolean
}