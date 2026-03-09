package com.dark.unified_inference.model

sealed class ModelFormat(val id: String) {
    data object GGUF : ModelFormat("gguf")
    data object TFLite : ModelFormat("tflite")
    data object LiteRTLM : ModelFormat("litertlm")
    data object DiffusionDir : ModelFormat("diffusion_dir")
    data object ONNX : ModelFormat("onnx")
    data class Custom(val formatId: String) : ModelFormat(formatId)
}
