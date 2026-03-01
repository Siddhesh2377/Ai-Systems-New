package com.mp.ai_gguf.models

import androidx.annotation.Keep

@Keep
data class EmbeddingResult @JvmOverloads constructor(
    val embeddings: FloatArray,
    val dimension: Int = embeddings.size,
    val poolingType: String = "mean",
    val numTokens: Int = 0,
    val timeMs: Long = 0L
) {
    fun toList(): List<Float> = embeddings.toList()

    fun cosineSimilarity(other: EmbeddingResult): Float {
        require(dimension == other.dimension) { "Embeddings must have same dimension" }
        var dot = 0f; var nA = 0f; var nB = 0f
        for (i in 0 until dimension) {
            dot += embeddings[i] * other.embeddings[i]
            nA += embeddings[i] * embeddings[i]
            nB += other.embeddings[i] * other.embeddings[i]
        }
        return if (nA > 0 && nB > 0) dot / (kotlin.math.sqrt(nA) * kotlin.math.sqrt(nB)) else 0f
    }

    fun norm(): Float {
        var sum = 0f
        for (v in embeddings) sum += v * v
        return kotlin.math.sqrt(sum)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as EmbeddingResult
        return embeddings.contentEquals(other.embeddings) && dimension == other.dimension
    }

    override fun hashCode(): Int = 31 * embeddings.contentHashCode() + dimension

    override fun toString(): String =
        "EmbeddingResult(dimension=$dimension, pooling=$poolingType, tokens=$numTokens, time=${timeMs}ms)"
}

@Keep
interface EmbeddingCallback {
    fun onProgress(progress: Float, currentTokens: Int, totalTokens: Int) {}
    fun onComplete(result: EmbeddingResult)
    fun onError(message: String)
}

@Keep
open class SimpleEmbeddingCallback : EmbeddingCallback {
    private var result: EmbeddingResult? = null
    private var errorMessage: String? = null
    private val lock = Object()

    override fun onComplete(result: EmbeddingResult) {
        synchronized(lock) { this.result = result; lock.notifyAll() }
    }

    override fun onError(message: String) {
        synchronized(lock) { this.errorMessage = message; lock.notifyAll() }
    }

    fun waitForResult(timeoutMs: Long = 0): EmbeddingResult? {
        synchronized(lock) {
            if (result == null && errorMessage == null) {
                if (timeoutMs > 0) lock.wait(timeoutMs) else lock.wait()
            }
            return result
        }
    }

    fun getResult(): EmbeddingResult? = result
    fun getError(): String? = errorMessage
    fun hasError(): Boolean = errorMessage != null
}
