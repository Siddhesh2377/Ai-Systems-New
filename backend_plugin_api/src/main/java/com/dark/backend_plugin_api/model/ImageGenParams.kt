package com.dark.backend_plugin_api.model

data class ImageGenParams(
    val prompt: String,
    val negativePrompt: String = "",
    val width: Int = 512,
    val height: Int = 512,
    val steps: Int = 20,
    val cfgScale: Float = 7.0f,
    val seed: Long = -1,
    val extras: Map<String, String> = emptyMap()
)
