package com.dark.unified_inference.text

import com.dark.unified_inference.core.InferenceEngine
import kotlinx.coroutines.flow.Flow

interface TextEngine : InferenceEngine {
    fun generateFlow(prompt: String, maxTokens: Int): Flow<TextEvent>
    fun generateMultiTurnFlow(messagesJson: String, maxTokens: Int): Flow<TextEvent>
}
