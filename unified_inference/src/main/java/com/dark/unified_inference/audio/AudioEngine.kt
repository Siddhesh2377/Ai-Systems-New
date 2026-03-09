package com.dark.unified_inference.audio

import com.dark.unified_inference.core.InferenceEngine
import kotlinx.coroutines.flow.Flow

interface AudioEngine : InferenceEngine {
    fun synthesize(text: String, config: AudioConfig): Flow<AudioEvent>
}
