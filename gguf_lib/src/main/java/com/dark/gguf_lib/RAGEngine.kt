package com.dark.gguf_lib

import com.dark.gguf_lib.models.RAGResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray

/**
 * Retrieval-augmented generation index with a separate embedding model and
 * a binary-quantized vector index.
 *
 * Two-stage retrieval: a binary-quantization Hamming search produces top-K
 * candidates, then cosine similarity re-ranks down to top-N. Indexes are
 * model-agnostic — re-loading a different LLM doesn't invalidate them; only
 * the embedding model fingerprint must match for [importIndex].
 *
 * ```kotlin
 * RAGEngine().use { rag ->
 *     rag.create()
 *     rag.loadModel("/path/to/embedding-model.gguf")
 *     rag.addDocument("Long document text...", docId = "doc-1")
 *     val hits = rag.query("search query")
 * }
 * ```
 */
class RAGEngine : AutoCloseable {

    @Volatile private var created = false
    @Volatile private var modelLoaded = false

    /**
     * Configure the engine. Must be called before any other method.
     *
     * @param threads      0 = auto-detect.
     * @param chunkSize    Tokens per chunk.
     * @param chunkOverlap Tokens overlapping between adjacent chunks.
     * @param dims         Matryoshka embedding truncation: 768/512/256/128.
     * @param topK         BQ candidates retrieved before re-ranking.
     * @param topN         Final results returned after cosine re-rank.
     * @param lateChunking Embed the full document once, then chunk the
     *                     context-aware token embeddings (preferred).
     */
    fun create(
        threads: Int = 0,
        chunkSize: Int = 256,
        chunkOverlap: Int = 32,
        dims: Int = 256,
        topK: Int = 32,
        topN: Int = 5,
        lateChunking: Boolean = true,
    ): Boolean {
        created = GGUFNativeLib.nativeCreateRagEngine(
            threads, chunkSize, chunkOverlap, dims, topK, topN, lateChunking,
        )
        modelLoaded = false
        return created
    }

    val isCreated: Boolean get() = created
    val isModelLoaded: Boolean get() = modelLoaded && GGUFNativeLib.nativeRagIsLoaded()

    /** Load the embedding model. Call after [create]. */
    suspend fun loadModel(path: String): Boolean = withContext(Dispatchers.IO) {
        if (!created) return@withContext false
        modelLoaded = GGUFNativeLib.nativeLoadRagModel(path)
        modelLoaded
    }

    /** Load the embedding model from a file descriptor (Android SAF). */
    suspend fun loadModelFromFd(fd: Int): Boolean = withContext(Dispatchers.IO) {
        if (!created) return@withContext false
        modelLoaded = GGUFNativeLib.nativeLoadRagModelFromFd(fd)
        modelLoaded
    }

    /**
     * Chunk [text] and add the embeddings to the index.
     *
     * @return Number of chunks created, or -1 on error.
     */
    suspend fun addDocument(text: String, docId: String): Int = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext -1
        GGUFNativeLib.nativeRagAddDocument(text, docId)
    }

    /**
     * Parse raw document bytes natively (PDF, DOCX, EPUB, ODT, PPTX, XLSX,
     * RTF, HTML, or plain text), extract text, and index it.
     *
     * @return Number of chunks (>= 0) or a negative error code:
     *   -1 unsupported format, -2 parse error, -3 empty, -4 OOM, -5 internal,
     *   -6 engine not ready.
     */
    suspend fun ingestBytes(
        bytes: ByteArray,
        mimeHint: String? = null,
        nameHint: String? = null,
        docId: String,
    ): Int = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext -6
        GGUFNativeLib.nativeRagIngestBytes(bytes, mimeHint, nameHint, docId)
    }

    /** Detect document kind from raw bytes / MIME / filename hints. */
    fun detectKind(
        bytes: ByteArray? = null,
        mimeHint: String? = null,
        nameHint: String? = null,
    ): DocKind = DocKind.fromNative(GGUFNativeLib.nativeRagDetectKind(bytes, mimeHint, nameHint))

    /**
     * Remove a document and all its chunks.
     *
     * @return 0 on success, -1 if not found.
     */
    fun removeDocument(docId: String): Int =
        if (created) GGUFNativeLib.nativeRagRemoveDocument(docId) else -1

    /** Drop everything from the index. */
    fun clear() { if (created) GGUFNativeLib.nativeRagClear() }

    val documentCount: Int get() = if (created) GGUFNativeLib.nativeRagDocumentCount() else 0
    val chunkCount: Int    get() = if (created) GGUFNativeLib.nativeRagChunkCount()    else 0

    /** Run a query. Results are sorted by descending score. */
    suspend fun query(query: String): List<RAGResult> = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext emptyList()
        parseResults(GGUFNativeLib.nativeRagQuery(query))
    }

    /**
     * Query restricted to chunks whose docId starts with [docIdPrefix]. An
     * empty prefix is equivalent to [query].
     */
    suspend fun queryFiltered(query: String, docIdPrefix: String): List<RAGResult> =
        withContext(Dispatchers.IO) {
            if (!isModelLoaded) return@withContext emptyList()
            parseResults(GGUFNativeLib.nativeRagQueryFiltered(query, docIdPrefix))
        }

    private fun parseResults(jsonStr: String?): List<RAGResult> {
        if (jsonStr.isNullOrEmpty()) return emptyList()
        return try {
            val arr = JSONArray(jsonStr)
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                RAGResult(
                    text = obj.getString("text"),
                    docId = obj.getString("doc_id"),
                    chunkIndex = obj.getInt("chunk_index"),
                    score = obj.getDouble("score").toFloat(),
                )
            }
        } catch (_: Exception) {
            emptyList()
        }
    }

    /**
     * Extract plain UTF-8 text from raw bytes without ingesting. Useful for
     * downstream Kotlin-side text handling (FTS5, summarization, etc.).
     *
     * @return null on parse failure / unsupported / empty bytes.
     */
    suspend fun extractText(
        bytes: ByteArray,
        mimeHint: String? = null,
        nameHint: String? = null,
    ): String? = withContext(Dispatchers.IO) {
        GGUFNativeLib.nativeRagExtractText(bytes, mimeHint, nameHint)
    }

    /**
     * Serialize the in-memory index (chunks, BQ vectors, float embeddings,
     * doc metadata, model fingerprint) to a portable byte buffer. Persist
     * this and call [importIndex] on the next launch to skip re-embedding.
     *
     * @return null on error / engine not created.
     */
    fun exportIndex(): ByteArray? =
        if (created) GGUFNativeLib.nativeRagExportIndex() else null

    /**
     * Restore an index serialized by [exportIndex]. The embedding model must
     * be loaded and match the fingerprint stored in the buffer.
     *
     * @return 0 on success; otherwise:
     *   -1 magic mismatch, -2 version mismatch, -3 dim mismatch,
     *   -4 model fingerprint mismatch, -5 corrupt buffer, -6 engine not ready.
     */
    fun importIndex(buf: ByteArray): Int =
        if (created) GGUFNativeLib.nativeRagImportIndex(buf) else -6

    /**
     * Run [query], retrieve context, and return [userPrompt] augmented with
     * the retrieved passages.
     */
    suspend fun buildPrompt(query: String, userPrompt: String): String? = withContext(Dispatchers.IO) {
        if (!isModelLoaded) return@withContext null
        GGUFNativeLib.nativeRagBuildPrompt(query, userPrompt)
    }

    /** Engine info as JSON: model status, chunk count, document count, configuration. */
    fun info(): String? = if (created) GGUFNativeLib.nativeRagInfo() else null

    override fun close() {
        if (created) {
            GGUFNativeLib.nativeReleaseRagEngine()
            created = false
            modelLoaded = false
        }
    }
}
