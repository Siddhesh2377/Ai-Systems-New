package com.mp.ai_gguf.models

import androidx.annotation.Keep
import kotlinx.serialization.Serializable

/**
 * Result of text embedding operation
 *
 * @property embeddings The embedding vector (normalized if requested)
 * @property dimension The dimension of the embedding vector
 * @property poolingType The pooling method used ("mean", "cls", "last", etc.)
 * @property numTokens Number of tokens processed
 * @property timeMs Time taken for encoding in milliseconds
 */
@Serializable
@Keep
data class EmbeddingResult @JvmOverloads constructor(
    val embeddings: FloatArray,
    val dimension: Int = embeddings.size,
    val poolingType: String = "mean",
    val numTokens: Int = 0,
    val timeMs: Long = 0L
) {
    /**
     * Get embedding as list for easier JSON serialization
     */
    fun toList(): List<Float> = embeddings.toList()

    /**
     * Compute cosine similarity with another embedding
     */
    fun cosineSimilarity(other: EmbeddingResult): Float {
        require(dimension == other.dimension) { "Embeddings must have same dimension" }

        var dotProduct = 0f
        var normA = 0f
        var normB = 0f

        for (i in 0 until dimension) {
            dotProduct += embeddings[i] * other.embeddings[i]
            normA += embeddings[i] * embeddings[i]
            normB += other.embeddings[i] * other.embeddings[i]
        }

        return if (normA > 0 && normB > 0) {
            dotProduct / (kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB))
        } else {
            0f
        }
    }

    /**
     * Get L2 norm (magnitude) of the embedding
     */
    fun norm(): Float {
        var sum = 0f
        for (value in embeddings) {
            sum += value * value
        }
        return kotlin.math.sqrt(sum)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as EmbeddingResult

        if (!embeddings.contentEquals(other.embeddings)) return false
        if (dimension != other.dimension) return false
        if (poolingType != other.poolingType) return false

        return true
    }

    override fun hashCode(): Int {
        var result = embeddings.contentHashCode()
        result = 31 * result + dimension
        result = 31 * result + poolingType.hashCode()
        return result
    }

    override fun toString(): String {
        return "EmbeddingResult(dimension=$dimension, pooling=$poolingType, tokens=$numTokens, time=${timeMs}ms)"
    }
}

/**
 * Callback interface for embedding generation with progress tracking
 *
 * All callbacks are invoked on the inference thread.
 * For UI updates, dispatch to the main thread.
 */
@Keep
interface EmbeddingCallback {
    /**
     * Called to report encoding progress
     * @param progress Progress from 0.0 to 1.0
     * @param currentTokens Number of tokens processed so far
     * @param totalTokens Total number of tokens to process
     */
    fun onProgress(progress: Float, currentTokens: Int, totalTokens: Int) {}

    /**
     * Called when encoding completes successfully
     * @param result The embedding result
     */
    fun onComplete(result: EmbeddingResult)

    /**
     * Called when an error occurs
     * @param message Error description
     */
    fun onError(message: String)
}

/**
 * Simple implementation of EmbeddingCallback that blocks until completion
 */
@Keep
open class SimpleEmbeddingCallback : EmbeddingCallback {
    private var result: EmbeddingResult? = null
    private var errorMessage: String? = null
    private val lock = Object()

    override fun onProgress(progress: Float, currentTokens: Int, totalTokens: Int) {
        // Override in subclass if progress tracking needed
    }

    override fun onComplete(result: EmbeddingResult) {
        synchronized(lock) {
            this.result = result
            lock.notifyAll()
        }
    }

    override fun onError(message: String) {
        synchronized(lock) {
            this.errorMessage = message
            lock.notifyAll()
        }
    }

    /**
     * Block until result is available
     * @param timeoutMs Maximum time to wait in milliseconds (0 = wait forever)
     * @return The embedding result, or null if timeout or error
     */
    fun waitForResult(timeoutMs: Long = 0): EmbeddingResult? {
        synchronized(lock) {
            if (result == null && errorMessage == null) {
                if (timeoutMs > 0) {
                    lock.wait(timeoutMs)
                } else {
                    lock.wait()
                }
            }
            return result
        }
    }

    fun getResult(): EmbeddingResult? = result
    fun getError(): String? = errorMessage
    fun hasError(): Boolean = errorMessage != null
}