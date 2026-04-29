package com.dark.ai_rmg.models

sealed class GenerationEvent {
    /**
     * One token from the generator. [bytes] is the raw byte form of the token
     * from the embedded tokenizer (empty if the model has no tokenizer).
     * BPE tokens may carry partial UTF-8 sequences — accumulate across tokens
     * before decoding to String, or use RmgEngine.decode() at end of stream.
     */
    class Token(val tokenId: Int, val bytes: ByteArray) : GenerationEvent() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Token) return false
            return tokenId == other.tokenId && bytes.contentEquals(other.bytes)
        }
        override fun hashCode(): Int = 31 * tokenId + bytes.contentHashCode()
        override fun toString(): String = "Token(tokenId=$tokenId, bytes=${bytes.size}B)"
    }
    data class Progress(val progress: Float) : GenerationEvent()
    data class Metrics(val metrics: DecodingMetrics) : GenerationEvent()
    data class Error(val message: String) : GenerationEvent()
    data object Done : GenerationEvent()
}
