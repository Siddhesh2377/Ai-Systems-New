package com.dark.unified_inference.image

import com.dark.unified_inference.core.InferenceEngine
import kotlinx.coroutines.flow.Flow

interface ImageEngine : InferenceEngine {
    fun generateImage(params: ImageGenerationParams): Flow<ImageEvent>
}
