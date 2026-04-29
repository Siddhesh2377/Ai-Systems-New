package com.dark.ai_rmg.models

class GenerationResult(
    val tokenIds: IntArray,
    val text: String?,
    val success: Boolean,
    val error: String? = null,
    val metrics: DecodingMetrics? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GenerationResult) return false
        return success == other.success
                && error == other.error
                && text == other.text
                && metrics == other.metrics
                && tokenIds.contentEquals(other.tokenIds)
    }

    override fun hashCode(): Int {
        var h = tokenIds.contentHashCode()
        h = 31 * h + (text?.hashCode() ?: 0)
        h = 31 * h + success.hashCode()
        h = 31 * h + (error?.hashCode() ?: 0)
        h = 31 * h + (metrics?.hashCode() ?: 0)
        return h
    }

    override fun toString(): String =
        "GenerationResult(tokens=${tokenIds.size}, success=$success, error=$error, metrics=$metrics)"
}
