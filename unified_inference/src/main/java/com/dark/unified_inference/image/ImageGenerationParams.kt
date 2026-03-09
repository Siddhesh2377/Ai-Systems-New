package com.dark.unified_inference.image

data class ImageGenerationParams(
    val prompt: String,
    val negativePrompt: String = "",
    val steps: Int = 28,
    val cfgScale: Float = 7f,
    val seed: Long = -1L,
    val width: Int = 512,
    val height: Int = 512,
    val scheduler: String = "dpm",
    val inputImage: String? = null,
    val mask: String? = null,
    val denoiseStrength: Float = 0.6f
)
