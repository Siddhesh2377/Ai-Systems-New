package com.dark.gguf_lib.models

/**
 * Callback interface for text embedding operations.
 */
interface EmbeddingCallback {
    fun onComplete(result: EmbeddingResult)
    fun onError(message: String)
}
