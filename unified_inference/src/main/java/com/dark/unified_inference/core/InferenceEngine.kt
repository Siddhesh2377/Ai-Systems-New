package com.dark.unified_inference.core

import com.dark.unified_inference.model.ModelDescriptor
import com.dark.unified_inference.model.ModelFormat

interface InferenceEngine {
    val engineId: String
    val displayName: String
    val providerTag: String
    val supportedFormats: List<ModelFormat>
    fun isModelLoaded(): Boolean
    suspend fun loadModel(descriptor: ModelDescriptor, params: String? = null): Boolean
    suspend fun unload()
    fun stopGeneration()
}
