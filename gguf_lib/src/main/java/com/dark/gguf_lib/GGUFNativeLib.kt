package com.dark.gguf_lib

import com.dark.gguf_lib.models.AgentCallback
import com.dark.gguf_lib.models.EmbeddingCallback
import com.dark.gguf_lib.models.StreamCallback

/**
 * Low-level JNI bridge to llama.cpp.
 *
 * All native methods are declared here. Higher-level APIs
 * (GGMLEngine, ToolManager, CharacterEngine) wrap these.
 */
object GGUFNativeLib {

    init {
        System.loadLibrary("gguf_lib")
    }

    // ---- Model Loading ----

    external fun nativeLoadModel(
        path: String, nCtx: Int, threadMode: Int,
        flashAttn: Boolean, cacheTypeK: String, cacheTypeV: String
    ): Boolean

    external fun nativeLoadModelFromFd(
        fd: Int, nCtx: Int, threadMode: Int,
        flashAttn: Boolean, cacheTypeK: String, cacheTypeV: String
    ): Boolean

    external fun nativeRelease()

    // ---- Model Info ----

    external fun nativeGetModelInfo(): String?

    // ---- Sampling ----

    external fun nativeSetSampling(
        temperature: Float, topK: Int, topP: Float, minP: Float,
        mirostat: Int, mirostatTau: Float, mirostatEta: Float, seed: Int
    )

    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeSetChatTemplate(template: String)
    external fun nativeUpdateSamplerParams(paramsJson: String): Boolean
    external fun nativeSetLogitBias(biasJson: String)

    // ---- Generation ----

    external fun nativeGenerateStream(
        prompt: String, maxTokens: Int, callback: StreamCallback
    ): Boolean

    external fun nativeGenerateStreamMultiTurn(
        messagesJson: String, maxTokens: Int, callback: StreamCallback
    ): Boolean

    external fun nativeStopGeneration()

    // ---- Tool Calling ----

    external fun nativeIsToolCallingSupported(): Boolean
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetGrammarMode(mode: Int)
    external fun nativeSetTypedGrammar(enabled: Boolean)

    // ---- Control Vectors ----

    external fun nativeLoadControlVectors(vectorsJson: String): Boolean
    external fun nativeClearControlVector()

    // ---- KV Cache State ----

    external fun nativeGetStateSize(): Long
    external fun nativeGetContextUsage(): Float
    external fun nativeStateSaveToFile(path: String): Boolean
    external fun nativeStateLoadFromFile(path: String): Boolean

    // ---- Character Engine ----

    external fun nativeSetPersonality(paramsJson: String)
    external fun nativeSetMood(mood: Int)
    external fun nativeGetCharacterContext(): String
    external fun nativeSetUncensored(enabled: Boolean)
    external fun nativeGetUncensored(): Boolean
    external fun nativeSupportsThinking(): Boolean
    external fun nativeSetThinkingEnabled(enabled: Boolean)

    // ---- Thread Mode ----

    external fun nativeSetThreadMode(mode: Int)

    // ---- Token Batch Size (tune for AIDL vs direct JNI) ----

    /** Set token batch size before each JNI/Binder callback. Default=256. Use 64 for direct, 512+ for AIDL. */
    external fun nativeSetTokenBatchSize(bytes: Int)

    // ---- Optimization Controls ----

    external fun nativeSetPromptCacheDir(path: String)
    external fun nativeWarmUp(): Boolean

    // ---- Embedding Engine ----

    external fun nativeLoadEmbeddingModel(path: String, nThreads: Int, nCtx: Int): Boolean
    external fun nativeEncodeText(text: String, normalize: Boolean, callback: EmbeddingCallback): Boolean
    external fun nativeReleaseEmbeddingModel()

    // ---- RAG Engine ----

    external fun nativeCreateRagEngine(
        nThreads: Int, chunkSize: Int, chunkOverlap: Int,
        nDims: Int, topK: Int, topN: Int, lateChunking: Boolean
    ): Boolean

    external fun nativeLoadRagModel(path: String): Boolean
    external fun nativeLoadRagModelFromFd(fd: Int): Boolean
    external fun nativeRagIsLoaded(): Boolean

    external fun nativeRagAddDocument(text: String, docId: String): Int
    external fun nativeRagRemoveDocument(docId: String): Int
    external fun nativeRagClear()
    external fun nativeRagDocumentCount(): Int
    external fun nativeRagChunkCount(): Int

    /** Returns JSON array of results: [{text, doc_id, chunk_index, score}, ...] */
    external fun nativeRagQuery(query: String): String?

    /** Returns augmented prompt with retrieved context injected */
    external fun nativeRagBuildPrompt(query: String, userPrompt: String): String?

    /** Returns JSON info about the RAG engine state */
    external fun nativeRagInfo(): String?

    external fun nativeReleaseRagEngine()

    // ---- Agent Engine ----

    external fun nativeInitAgentSystem(callback: AgentCallback, toolSchemasJson: String): Boolean
    external fun nativeRunAgentStep(userMessage: String, systemPrompt: String, maxRounds: Int)
    external fun nativeStopAgent()
    external fun nativeReleaseAgentSystem()

    // ---- VLM (Vision Language Model) ----

    external fun nativeVlmLoadProjector(path: String, nThreads: Int): Boolean
    external fun nativeVlmLoadProjectorFromFd(fd: Int, nThreads: Int): Boolean
    external fun nativeVlmRelease()
    external fun nativeVlmIsLoaded(): Boolean
    external fun nativeVlmGetInfo(): String?
    external fun nativeVlmGetDefaultMarker(): String

    /**
     * Generate from text + images. The prompt (inside messagesJson) should contain
     * image markers (from nativeVlmGetDefaultMarker()) where images should appear.
     * @param messagesJson JSON array of chat messages
     * @param imageData array of byte arrays — each is raw file bytes (JPEG/PNG)
     */
    external fun nativeVlmGenerateStream(
        messagesJson: String,
        imageData: Array<ByteArray>,
        maxTokens: Int,
        callback: StreamCallback
    ): Boolean
}
