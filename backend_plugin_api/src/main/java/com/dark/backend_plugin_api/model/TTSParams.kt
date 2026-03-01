package com.dark.backend_plugin_api.model

data class TTSParams(
    val voice: String = "default",
    val speed: Float = 1.0f,
    val language: String = "en",
    val extras: Map<String, String> = emptyMap()
)
