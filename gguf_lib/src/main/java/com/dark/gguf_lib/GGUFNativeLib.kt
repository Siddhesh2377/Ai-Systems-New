package com.dark.gguf_lib

import com.dark.gguf_lib.models.AgentCallback
import com.dark.gguf_lib.models.EmbeddingCallback
import com.dark.gguf_lib.models.StreamCallback

/**
 * Low-level JNI bridge to llama.cpp + tool-neuron engine helpers.
 *
 * Consumers should not call this directly — use the higher-level wrappers
 * ([GGMLEngine], [ToolManager], [CharacterEngine], [EmbeddingEngine],
 * [RAGEngine]) instead. Names and signatures here are load-bearing —
 * `consumer-rules.pro` keeps every native method by name, and the C++ side
 * looks them up via JNI auto-discovery (no `RegisterNatives`).
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

    external fun nativeIsToolCallingSupported(): Boolean
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetGrammarMode(mode: Int)
    external fun nativeSetTypedGrammar(enabled: Boolean)

    external fun nativeLoadControlVectors(vectorsJson: String): Boolean
    external fun nativeClearControlVector()

    external fun nativeGetStateSize(): Long
    external fun nativeGetContextUsage(): Float
    external fun nativeStateSaveToFile(path: String): Boolean
    external fun nativeStateLoadFromFile(path: String): Boolean

    /** StreamingLLM-style eviction. nWindow=0 disables, falls back to context shift. */
    external fun nativeSetKvPolicy(nSink: Int, nWindow: Int, evictAtFull: Boolean)

    /** Apply eviction immediately — useful after a long prefill. */
    external fun nativeEvictToBudget()

    external fun nativeSetPersonality(paramsJson: String)
    external fun nativeSetMood(mood: Int)
    external fun nativeGetCharacterContext(): String
    external fun nativeSetUncensored(enabled: Boolean)
    external fun nativeGetUncensored(): Boolean
    external fun nativeSupportsThinking(): Boolean
    external fun nativeSetThinkingEnabled(enabled: Boolean)

    external fun nativeSetThreadMode(mode: Int)

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

    external fun nativeInitAgentSystem(callback: AgentCallback, toolSchemasJson: String): Boolean
    external fun nativeRunAgentStep(userMessage: String, systemPrompt: String, maxRounds: Int)
    external fun nativeStopAgent()
    external fun nativeReleaseAgentSystem()

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
    external fun nativeVlmIsLoaded(): Boolean
    external fun nativeVlmGetInfo(): String?
    external fun nativeVlmGetDefaultMarker(): String

    /**
     * Generate from text + images. messagesJson must contain image markers
     * (from [nativeVlmGetDefaultMarker]) where each image should appear.
     */
    external fun nativeVlmGenerateStream(
        messagesJson: String,
        imageData: Array<ByteArray>,
        maxTokens: Int,
        callback: StreamCallback,
    ): Boolean
}
