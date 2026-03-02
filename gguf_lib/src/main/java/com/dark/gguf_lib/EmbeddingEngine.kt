package com.dark.gguf_lib

import com.dark.gguf_lib.models.EmbeddingCallback
import com.dark.gguf_lib.models.EmbeddingResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlin.coroutines.resume

/**
 * EmbeddingEngine - Separate model instance for text embeddings.
 *
 * Runs independently of the main GGMLEngine, allowing simultaneous
 * LLM generation and embedding computation.
 *
 * Usage:
 * ```
 * val embedder = EmbeddingEngine()
 * embedder.load("/path/to/embedding-model.gguf")
 *
 * val vector = embedder.embed("Hello world")
 * println("Dimension: ${vector?.size}")
 *
 * embedder.close()
 * ```
 */
class EmbeddingEngine : AutoCloseable {

    private var loaded = false

    /**
     * Load an embedding model.
     *
     * @param path Path to the .gguf embedding model
     * @param threads Number of threads (0 = auto-detect)
     * @param contextSize Context size for embeddings (default 512)
     */
    fun load(path: String, threads: Int = 0, contextSize: Int = 512): Boolean {
        loaded = GGUFNativeLib.nativeLoadEmbeddingModel(path, threads, contextSize)
        return loaded
    }

    val isLoaded: Boolean get() = loaded

    /**
     * Generate embeddings for text. Returns normalized float vector.
     * Times out after 15 seconds.
     */
    suspend fun embed(text: String, normalize: Boolean = true): FloatArray? = withContext(Dispatchers.IO) {
        if (!loaded) return@withContext null
        try {
            withTimeout(15_000) {
                suspendCancellableCoroutine { cont ->
                    val cb = object : EmbeddingCallback {
                        override fun onComplete(result: EmbeddingResult) {
                            if (cont.isActive) cont.resume(result.embeddings)
                        }
                        override fun onError(message: String) {
                            if (cont.isActive) cont.resume(null)
                        }
                    }
                    GGUFNativeLib.nativeEncodeText(text, normalize, cb)
                }
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Batch embed multiple texts.
     */
    suspend fun embedBatch(texts: List<String>, normalize: Boolean = true): List<FloatArray?> =
        texts.map { embed(it, normalize) }

    override fun close() {
        if (loaded) {
            GGUFNativeLib.nativeReleaseEmbeddingModel()
            loaded = false
        }
    }
}
