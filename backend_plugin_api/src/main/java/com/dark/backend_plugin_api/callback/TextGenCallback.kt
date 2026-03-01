package com.dark.backend_plugin_api.callback

import com.dark.backend_plugin_api.model.GenerationMetrics

interface TextGenCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone(metrics: GenerationMetrics?)
    fun onError(message: String)
}
