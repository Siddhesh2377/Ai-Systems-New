package com.dark.gguf_lib

import com.dark.gguf_lib.models.EmbeddingCallback
import com.dark.gguf_lib.models.EmbeddingResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlin.coroutines.resume

/**
 * Standalone text embedding engine. Holds its own llama.cpp model + context,
 * independent of [GGMLEngine] — both can run concurrently.
 *
 * Each call to [embed] tokenizes the text, decodes a single batch, and pulls
 * the sequence embedding (or the last-token embedding if the model doesn't
 * expose pooled embeddings). [embedBatch] is a simple sequential map; consumers
 * with concurrent batching needs should drive [embed] from their own dispatcher.
 *
 * ```kotlin
 * EmbeddingEngine().use { embedder ->
 *     embedder.load("/path/to/embedding-model.gguf")
 *     val v = embedder.embed("hello world")
 * }
 * ```
 */
class EmbeddingEngine : AutoCloseable {

    @Volatile private var loaded = false

    /**
     * Load an embedding model.
     *
     * @param path        Absolute path to the .gguf embedding model.
     * @param threads     0 = inherit batch threads from the current thread mode.
     * @param contextSize Max input length in tokens. Default 512 — bump only
     *                    if you need to embed long passages.
     */
    suspend fun load(path: String, threads: Int = 0, contextSize: Int = 512): Boolean =
        withContext(Dispatchers.IO) {
            loaded = GGUFNativeLib.nativeLoadEmbeddingModel(path, threads, contextSize)
            loaded
        }

    val isLoaded: Boolean get() = loaded

    /**
     * Compute an embedding for [text]. Returns null on tokenize/decode failure
     * or if the model isn't loaded. Times out after 15 seconds.
     *
     * @param normalize L2-normalize the result so cosine similarity reduces to a dot product.
     */
    suspend fun embed(text: String, normalize: Boolean = true): FloatArray? = withContext(Dispatchers.IO) {
        if (!loaded) return@withContext null
        try {
            withTimeout(15_000) {
                suspendCancellableCoroutine<FloatArray?> { cont ->
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

    /** Sequential batch embedding. Returns one FloatArray per input (null on per-item failure). */
    suspend fun embedBatch(texts: List<String>, normalize: Boolean = true): List<FloatArray?> =
        texts.map { embed(it, normalize) }

    override fun close() {
        if (loaded) {
            GGUFNativeLib.nativeReleaseEmbeddingModel()
            loaded = false
        }
    }
}
