package com.dark.unified_inference.capability

import com.dark.unified_inference.model.ModelDescriptor
import com.dark.unified_inference.text.TextEvent
import kotlinx.coroutines.flow.Flow

interface VisionCapable {
    fun loadVisionProjector(descriptor: ModelDescriptor): Boolean
    fun releaseVisionProjector()
    fun generateVisionFlow(messagesJson: String, imageData: List<ByteArray>, maxTokens: Int): Flow<TextEvent>
}
