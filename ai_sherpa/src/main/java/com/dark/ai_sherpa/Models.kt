// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

data class OnlineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,
    val ysProbs: FloatArray
)

data class OfflineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray
)

data class WaveData(val samples: FloatArray, val sampleRate: Int)

data class SpeechSegment(val start: Int, val samples: FloatArray)

data class GeneratedAudio(val samples: FloatArray, val sampleRate: Int)
