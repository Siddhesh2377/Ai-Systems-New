package com.dark.gguf_lib

import com.dark.gguf_lib.models.RAGResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray

/**
 * RAG Engine - Context-preserving retrieval-augmented generation.
 *
 * Uses a separate embedding model (e.g., EmbeddingGemma-300M Q4) with late chunking
 * for context-aware embeddings and binary quantization for compact storage.
 * Model-agnostic: the RAG index survives LLM swaps.
 *
 * Usage:
 * ```
 * val rag = RAGEngine()
 * rag.create()
 * rag.loadModel("/path/to/embedding-model.gguf")
 *
 * rag.addDocument("Long document text...", "doc-1")
 * rag.addDocument("Another document...", "doc-2")
 *
 * val results = rag.query("search query")
 * results.forEach { println("${it.docId}: ${it.score} - ${it.text}") }
 *
 * // Or build an augmented prompt for LLM generation
 * val prompt = rag.buildPrompt("user question", "Answer based on context:")
 *
 * rag.close()
 * ```
 */
class RAGEngine : AutoCloseable {

    private var created = false
    private var modelLoaded = false

    /**
     * Create the RAG engine with given parameters.
     *
     * @param threads Number of threads (0 = auto-detect)
     * @param chunkSize Tokens per chunk (default 256)
     * @param chunkOverlap Overlap tokens between chunks (default 32)
     * @param dims Matryoshka embedding dimensions: 768/512/256/128 (default 256)
     * @param topK BQ search candidates before re-ranking (default 32)
     * @param topN Final results after cosine re-rank (default 5)
     * @param lateChunking Embed full doc then chunk for context-aware embeddings (default true)
     */
    fun create(
        threads: Int = 0,
        chunkSize: Int = 256,
        chunkOverlap: Int = 32,
        dims: Int = 256,
        topK: Int = 32,
        topN: Int = 5,
        lateChunking: Boolean = true
    ): Boolean {
        created = GGUFNativeLib.nativeCreateRagEngine(
            threads, chunkSize, chunkOverlap, dims, topK, topN, lateChunking
        )
        modelLoaded = false
        return created
    }

    val isCreated: Boolean get() = created
    val isModelLoaded: Boolean get() = modelLoaded && GGUFNativeLib.nativeRagIsLoaded()

    /**
     * Load the embedding model for RAG. Call after [create].
     *
     * @param path Path to the embedding model GGUF file
     */
    fun loadModel(path: String): Boolean {
        if (!created) return false
        modelLoaded = GGUFNativeLib.nativeLoadRagModel(path)
        return modelLoaded
    }

    /**
     * Load the embedding model from a file descriptor (Android SAF).
     *
     * @param fd File descriptor from ContentResolver.openFileDescriptor()
     */
    fun loadModelFromFd(fd: Int): Boolean {
        if (!created) return false
        modelLoaded = GGUFNativeLib.nativeLoadRagModelFromFd(fd)
        return modelLoaded
    }

    /**
     * Add a document to the RAG index. The document is chunked and embedded.
     *
     * @param text Full document text
     * @param docId Unique document identifier
     * @return Number of chunks created, or -1 on error
     */
    suspend fun addDocument(text: String, docId: String): Int = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext -1
        GGUFNativeLib.nativeRagAddDocument(text, docId)
    }

    /**
     * Remove a document and all its chunks from the index.
     *
     * @param docId Document identifier to remove
     * @return 0 on success, -1 if not found
     */
    fun removeDocument(docId: String): Int {
        if (!created) return -1
        return GGUFNativeLib.nativeRagRemoveDocument(docId)
    }

    /** Clear all documents and chunks from the index. */
    fun clear() {
        if (created) GGUFNativeLib.nativeRagClear()
    }

    /** Number of indexed documents. */
    val documentCount: Int get() = if (created) GGUFNativeLib.nativeRagDocumentCount() else 0

    /** Number of indexed chunks across all documents. */
    val chunkCount: Int get() = if (created) GGUFNativeLib.nativeRagChunkCount() else 0

    /**
     * Query the RAG index. Returns ranked results by relevance.
     *
     * Uses two-stage retrieval: binary quantization Hamming search for candidates,
     * then cosine similarity re-ranking for final results.
     *
     * @param query The search query
     * @return List of [RAGResult] sorted by descending score
     */
    suspend fun query(query: String): List<RAGResult> = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext emptyList()

        val jsonStr = GGUFNativeLib.nativeRagQuery(query) ?: return@withContext emptyList()

        try {
            val arr = JSONArray(jsonStr)
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                RAGResult(
                    text = obj.getString("text"),
                    docId = obj.getString("doc_id"),
                    chunkIndex = obj.getInt("chunk_index"),
                    score = obj.getDouble("score").toFloat()
                )
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    /**
     * Query the index and build an augmented prompt with retrieved context.
     *
     * @param query The search query (used for retrieval)
     * @param userPrompt The user's prompt template to augment with context
     * @return Augmented prompt with context injected, or null on error
     */
    suspend fun buildPrompt(query: String, userPrompt: String): String? = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext null
        GGUFNativeLib.nativeRagBuildPrompt(query, userPrompt)
    }

    /**
     * Get RAG engine info as JSON string.
     * Includes model status, chunk count, document count, and configuration.
     */
    fun info(): String? {
        if (!created) return null
        return GGUFNativeLib.nativeRagInfo()
    }

    override fun close() {
        if (created) {
            GGUFNativeLib.nativeReleaseRagEngine()
            created = false
            modelLoaded = false
        }
    }
}
