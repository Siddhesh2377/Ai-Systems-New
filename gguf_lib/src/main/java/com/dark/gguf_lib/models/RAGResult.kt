package com.dark.gguf_lib.models

/**
 * A single retrieval result from the RAG engine.
 *
 * @param text The chunk text that matched the query
 * @param docId The document ID this chunk belongs to
 * @param chunkIndex The chunk index within the document
 * @param score Cosine similarity score (0.0 to 1.0)
 */
data class RAGResult(
    val text: String,
    val docId: String,
    val chunkIndex: Int,
    val score: Float
)
