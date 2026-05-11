package com.dark.gguf_lib

import com.dark.gguf_lib.models.EmbeddingCallback
import com.dark.gguf_lib.models.StreamCallback
import com.dark.gguf_lib.models.VlmPrewarmCallback

/**
 * Low-level JNI bridge to llama.cpp + tool-neuron engine helpers.
 *
 * Consumers should not call this directly — use the higher-level wrappers
 * ([GGMLEngine], [EmbeddingEngine], [RAGEngine]) instead. Names and
 * signatures here are load-bearing — `consumer-rules.pro` keeps every
 * native method by name, and the C++ side looks them up via JNI
 * auto-discovery (no `RegisterNatives`).
 */
internal object GGUFNativeLib {

    init {
        System.loadLibrary("gguf_lib")
    }

    external fun nativeLoadModel(
        path: String,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        flashAttn: Boolean,
        useMmap: Boolean,
        useMlock: Boolean,
        cacheTypeK: String,
        cacheTypeV: String,
        opOffload: Boolean,
    ): Boolean

    external fun nativeLoadModelFromFd(
        fd: Int,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        flashAttn: Boolean,
        useMmap: Boolean,
        useMlock: Boolean,
        cacheTypeK: String,
        cacheTypeV: String,
        opOffload: Boolean,
    ): Boolean

    external fun nativeRelease()

    external fun nativeGetModelInfo(): String?

    external fun nativeSetSampling(
        temperature: Float, topK: Int, topP: Float, minP: Float,
        mirostat: Int, mirostatTau: Float, mirostatEta: Float, seed: Int,
    )

    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeSetChatTemplate(template: String)
    external fun nativeUpdateSamplerParams(paramsJson: String): Boolean
    external fun nativeSetLogitBias(biasJson: String)

    external fun nativeGenerateStream(
        prompt: String, maxTokens: Int, callback: StreamCallback,
    ): Boolean

    external fun nativeGenerateStreamMultiTurn(
        messagesJson: String, maxTokens: Int, callback: StreamCallback,
    ): Boolean

    external fun nativeStopGeneration()

    external fun nativeGetStateSize(): Long
    external fun nativeGetContextUsage(): Float
    external fun nativeGetMemoryStatsJson(): String?
    external fun nativeStateSaveToFile(path: String): Boolean
    external fun nativeStateLoadFromFile(path: String): Boolean

    /** StreamingLLM-style eviction. nWindow=0 disables, falls back to context shift. */
    external fun nativeSetKvPolicy(nSink: Int, nWindow: Int, evictAtFull: Boolean)

    /** Apply eviction immediately — useful after a long prefill. */
    external fun nativeEvictToBudget()

    external fun nativeSupportsThinking(): Boolean
    external fun nativeSetThinkingEnabled(enabled: Boolean)

    external fun nativeSetThreadMode(mode: Int)

    // ── Power engine / decode diagnostics ──────────────────────────────────
    //
    // The thermal/auto-mode surface lives outside the load path so the host
    // can toggle it independently of the model lifecycle. All these calls are
    // safe to invoke when no model is loaded — they just no-op on ctx-touching
    // sub-paths (e.g. nativeAutoModeTick won't re-attach a threadpool when
    // there's no context yet).

    /**
     * JSON snapshot of the last completed generate call's per-stage timing:
     * `{tokens, sample_us, detok_us, stop_us, decode_us, total_us}`.
     * All `_us` fields are aggregate microseconds across the run; divide by
     * `tokens` for per-token cost. Returns "{}"-equivalent if no generate has
     * happened on this process yet.
     */
    external fun nativeGetLastDecodeBreakdown(): String

    /**
     * Thermal snapshot:
     * `{maxTempMilliC, batteryTempMilliC, throttlingLevel, nZonesRead}`.
     * `throttlingLevel`: 0 COOL, 1 WARM, 2 HOT, 3 CRITICAL.
     * Reading is stateless and safe to call concurrently with generation.
     */
    external fun nativeGetThermalState(): String

    /**
     * Enable/disable auto-mode. When on, the engine reads thermal state at
     * each [nativeAutoModeTick] and may de-rate the requested thread mode if
     * the device is hot.
     */
    external fun nativeSetAutoMode(enabled: Boolean)

    external fun nativeIsAutoModeEnabled(): Boolean

    /** Returns the *effective* thread mode (what the engine is actually running). */
    external fun nativeGetEffectiveThreadMode(): Int

    /**
     * Override default thermal thresholds. Defaults: warm=60000, hot=75000,
     * crit=85000 (milli-Celsius). Pass <=0 to keep a field's current value.
     */
    external fun nativeSetThermalThresholds(warmMilliC: Int, hotMilliC: Int, critMilliC: Int)

    /**
     * Tick the auto-mode loop. When auto-mode is on, polls thermal state and
     * adjusts the effective thread mode if needed. Returns the effective mode
     * after the tick (0/1/2). Cheap to call (~100 us); host typically calls
     * it once before each generate.
     */
    external fun nativeAutoModeTick(): Int

    /**
     * Token-batching threshold in bytes. Larger = fewer Binder/JNI calls but
     * higher latency to first visible token. 64 = direct JNI; 256 = default;
     * 512+ = AIDL service to amortize Binder IPC (~20-50us/call).
     */
    external fun nativeSetTokenBatchSize(bytes: Int)

    external fun nativeSetPromptCacheDir(path: String)
    external fun nativeWarmUp(): Boolean

    external fun nativeLoadEmbeddingModel(path: String, nThreads: Int, nCtx: Int): Boolean
    external fun nativeEncodeText(text: String, normalize: Boolean, callback: EmbeddingCallback): Boolean
    external fun nativeReleaseEmbeddingModel()

    external fun nativeCreateRagEngine(
        nThreads: Int, chunkSize: Int, chunkOverlap: Int,
        nDims: Int, topK: Int, topN: Int, lateChunking: Boolean,
    ): Boolean

    external fun nativeLoadRagModel(path: String): Boolean
    external fun nativeLoadRagModelFromFd(fd: Int): Boolean
    external fun nativeRagIsLoaded(): Boolean

    external fun nativeRagAddDocument(text: String, docId: String): Int
    external fun nativeRagRemoveDocument(docId: String): Int
    external fun nativeRagClear()
    external fun nativeRagDocumentCount(): Int
    external fun nativeRagChunkCount(): Int

    external fun nativeRagIngestBytes(
        bytes: ByteArray, mimeHint: String?, nameHint: String?, docId: String,
    ): Int

    external fun nativeRagDetectKind(
        bytes: ByteArray?, mimeHint: String?, nameHint: String?,
    ): Int

    external fun nativeErrorInit()
    external fun nativeErrorSetCrashLogPath(path: String)
    external fun nativeErrorGetLastJson(): String
    external fun nativeErrorClear()

    external fun nativeTextDigest(
        text: String,
        query: String?,
        targetTokens: Int,
        wQuery: Float,
        wCentrality: Float,
        wLead: Float,
        wEntity: Float,
        mmrLambda: Float,
        maxSentences: Int,
        minSentenceChars: Int,
        maxSentenceChars: Int,
        textrankIterations: Int,
        textrankDamping: Float,
    ): String?

    /** Returns JSON array `[{text, doc_id, chunk_index, score}, ...]`. */
    external fun nativeRagQuery(query: String): String?

    /** Same as [nativeRagQuery] but restricted to chunks whose docId starts with [docIdPrefix]. */
    external fun nativeRagQueryFiltered(query: String, docIdPrefix: String?): String?

    /** Extract plain UTF-8 text from raw bytes without ingesting. Returns null on parse failure. */
    external fun nativeRagExtractText(
        bytes: ByteArray, mimeHint: String?, nameHint: String?,
    ): String?

    /** Serialize the in-memory RAG index to a portable byte buffer. */
    external fun nativeRagExportIndex(): ByteArray?

    /**
     * Import a buffer produced by [nativeRagExportIndex]. Engine must be created
     * and an embedding model loaded.
     *
     * @return 0 on success, or:
     *   -1 magic mismatch, -2 version mismatch, -3 dim mismatch,
     *   -4 model fingerprint mismatch, -5 corrupt buffer, -6 engine not ready.
     */
    external fun nativeRagImportIndex(buf: ByteArray): Int

    /** Returns an augmented prompt with retrieved context injected. */
    external fun nativeRagBuildPrompt(query: String, userPrompt: String): String?

    /** Returns JSON info about the RAG engine state. */
    external fun nativeRagInfo(): String?

    external fun nativeReleaseRagEngine()

    /**
     * Load a vision/audio projector (mmproj GGUF) onto the currently loaded text model.
     *
     * @param nThreads 0 = auto (inherits the engine's batch threads).
     * @param imageMinTokens / imageMaxTokens -1 = model default. For LFM2-VL,
     *   imageMaxTokens caps only the overview image, not the per-tile grid
     *   (the latter is a compile-time constant in clip.cpp).
     *
     * The mtmd projector binds n_threads at init. To pick up a new thread mode,
     * call [nativeVlmRelease] then reload.
     */
    external fun nativeVlmLoadProjector(
        path: String, nThreads: Int, imageMinTokens: Int, imageMaxTokens: Int,
    ): Boolean

    external fun nativeVlmLoadProjectorFromFd(
        fd: Int, nThreads: Int, imageMinTokens: Int, imageMaxTokens: Int,
    ): Boolean

    external fun nativeVlmRelease()
    external fun nativeVlmGetInfo(): String?
    external fun nativeVlmGetDefaultMarker(): String

    /**
     * Run only the vision encoder for [imageData] and store the resulting
     * embeddings in the VT cache under [vtKey] (32 bytes). No LLM context is
     * touched — purely a ViT warm-up. Subsequent
     * [nativeVlmGenerateStream] calls with the same [vtKey] hit the cache and
     * skip the ~9s ViT pass.
     *
     * Requires: text model loaded, projector loaded, VT cache initialised.
     * Returns true on successful encode + store.
     */
    external fun nativeVlmPrecomputeVisionEmbeddings(
        imageData: ByteArray,
        vtKey: ByteArray,
        imageQuality: Int,            // 0=LOW, 1=MEDIUM, 2=HIGH
    ): Boolean

    /**
     * Pre-warm the VLM-KV cache: encode the image AND run the LLM
     * image-prefill, then capture the post-image LLM state under [vlmKvKey].
     * The next [nativeVlmGenerateStream] call with the same key restores
     * the state and skips both the ViT pass AND the ~9s LLM image-prefill,
     * so even the *very first* user prompt against this image gets
     * sub-second TTFT.
     *
     * [messagesJson] should be the canonical pre-warm prompt — the
     * system + user-prefix-up-to-image-marker the host plans to use later.
     * The cache key must match what the host passes at generate time
     * (use [GGMLEngine.computeVlmKvKey] for both).
     *
     * Pass [vtKey] (32 bytes) to also populate the VT cache as a
     * side-effect; pass null to skip the VT-side write.
     *
     * Requires: text model loaded, projector loaded, VLM-KV cache initialised.
     */
    external fun nativeVlmPrecomputeKvState(
        messagesJson: String,
        imageData: ByteArray,
        vtKey: ByteArray?,
        vlmKvKey: ByteArray,
        imageQuality: Int,            // 0=LOW, 1=MEDIUM, 2=HIGH
        callback: VlmPrewarmCallback?,
    ): Boolean

    /**
     * Generate from text + images. messagesJson must contain image markers
     * (from [nativeVlmGetDefaultMarker]) where each image should appear.
     *
     * @param vtKeys Optional 32-byte SHA256 keys, parallel to [imageData].
     *   Pass null (or null entries) to skip the VT cache for that image.
     *   When the cache is initialised and a key is provided, native first
     *   tries [vt_cache_lookup]; on hit it skips the ~10s ViT pass entirely.
     * @param vlmKvKey Optional single 32-byte SHA256 covering the *whole*
     *   pre-question state (system prompt + chat template + image bytes +
     *   projector + image_max_tokens + model fingerprint). On hit, the LLM
     *   context state captured at the post-image-chunk boundary is restored
     *   and BOTH the ViT pass AND the ~9s LLM image-prefill are skipped.
     *   Pass null to disable.
     */
    external fun nativeVlmGenerateStream(
        messagesJson: String,
        imageData: Array<ByteArray>,
        vtKeys: Array<ByteArray>?,
        vlmKvKey: ByteArray?,
        imageQuality: Int,            // 0=LOW, 1=MEDIUM, 2=HIGH (passthrough)
        maxTokens: Int,
        callback: StreamCallback,
    ): Boolean

    // ── VT (Vision Token) cache ─────────────────────────────────────────────
    //
    // Content-addressed store for ViT-encoded image embeddings. Survives
    // process restarts. LRU-evicted when total bytes exceeds the budget.

    /** Open the cache at [dir] with [budgetBytes] (0 = default 200 MB). */
    external fun nativeVtCacheInit(dir: String, budgetBytes: Long): Boolean

    /** Close the cache. Files on disk persist; only the in-memory index is freed. */
    external fun nativeVtCacheRelease()

    /** Drop every entry from disk and reset stats. */
    external fun nativeVtCacheClear()

    external fun nativeVtCacheSetBudget(bytes: Long)

    /** Returns JSON: `{initialized, total_bytes, budget_bytes, entry_count, hits, misses}`. */
    external fun nativeVtCacheStatsJson(): String

    /** Returns JSON array of `{hash, n_tokens, n_embd, size_bytes, last_access_ms}`. */
    external fun nativeVtCacheListEntriesJson(): String

    /** Drop a single entry by 32-byte hash. Returns true if it was present. */
    external fun nativeVtCacheRemove(hash: ByteArray): Boolean

    // ── VLM-KV cache ────────────────────────────────────────────────────────
    //
    // Stores the LLM context state captured at the post-image-chunk boundary
    // during VLM prompt-eval. On hit, both the vision encoder AND the
    // image-prefill llama_decode are skipped. Survives process restarts.

    /** Open the cache at [dir] with [budgetBytes] (0 = default 300 MB). */
    external fun nativeVlmKvCacheInit(dir: String, budgetBytes: Long): Boolean

    /** Close the cache. Files on disk persist; only the in-memory index is freed. */
    external fun nativeVlmKvCacheRelease()

    /** Drop every entry from disk and reset stats. */
    external fun nativeVlmKvCacheClear()

    external fun nativeVlmKvCacheSetBudget(bytes: Long)

    /** Returns JSON: `{initialized, total_bytes, budget_bytes, entry_count, hits, misses}`. */
    external fun nativeVlmKvCacheStatsJson(): String

    /** Returns JSON array of `{hash, n_tokens, size_bytes, last_access_ms}`. */
    external fun nativeVlmKvCacheListEntriesJson(): String

    /** Drop a single entry by 32-byte hash. Returns true if it was present. */
    external fun nativeVlmKvCacheRemove(hash: ByteArray): Boolean

    /**
     * Returns a JSON snapshot of every ggml backend + device registered at
     * startup. Purely diagnostic — calling this does not change which backend
     * llama.cpp uses for compute.
     *
     * Shape:
     * ```
     * {
     *   "backends": [{"name": "CPU"}, {"name": "Vulkan"}, ...],
     *   "devices":  [{"name": "...", "description": "...", "type": "cpu|gpu|igpu|accel",
     *                 "memory_free": 1234567, "memory_total": 12345678,
     *                 "async": false, "events": false}, ...]
     * }
     * ```
     */
    external fun nativeListBackendsJson(): String
}
