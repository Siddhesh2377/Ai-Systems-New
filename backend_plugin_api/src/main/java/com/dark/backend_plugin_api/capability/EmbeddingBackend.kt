package com.dark.backend_plugin_api.capability

/**
 * Embedding capability — encode text into dense vectors.
 * Implement alongside [BackendPlugin] if the backend supports EMBEDDING.
 */
interface EmbeddingBackend {

    suspend fun encode(text: String, normalize: Boolean = true): Result<FloatArray>

    suspend fun encodeBatch(
        texts: List<String>,
        normalize: Boolean = true
    ): Result<List<FloatArray>>

    /** Embedding dimension for the loaded model */
    fun embeddingDimension(): Int
}
