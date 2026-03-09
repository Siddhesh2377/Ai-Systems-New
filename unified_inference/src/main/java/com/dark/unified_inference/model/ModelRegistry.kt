package com.dark.unified_inference.model

import com.dark.unified_inference.core.InferenceEngine

class ModelRegistry {
    private val engines = mutableListOf<InferenceEngine>()

    fun register(engine: InferenceEngine) {
        engines.add(engine)
    }

    fun unregister(engine: InferenceEngine) {
        engines.remove(engine)
    }

    fun enginesForFormat(format: ModelFormat): List<InferenceEngine> =
        engines.filter { format in it.supportedFormats }

    fun resolveFormat(fileName: String): ModelFormat? = when {
        fileName.endsWith(".gguf") -> ModelFormat.GGUF
        fileName.endsWith(".tflite") -> ModelFormat.TFLite
        fileName.endsWith(".litertlm") -> ModelFormat.LiteRTLM
        fileName.endsWith(".onnx") -> ModelFormat.ONNX
        else -> null
    }

    fun allEngines(): List<InferenceEngine> = engines.toList()
}
