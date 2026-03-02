package com.dark.gguf_lib.models

/**
 * Result from text embedding operation.
 * Constructed from native code via JNI.
 */
data class EmbeddingResult(val embeddings: FloatArray) {

    val dimension: Int get() = embeddings.size

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EmbeddingResult) return false
        return embeddings.contentEquals(other.embeddings)
    }

    override fun hashCode(): Int = embeddings.contentHashCode()
}
