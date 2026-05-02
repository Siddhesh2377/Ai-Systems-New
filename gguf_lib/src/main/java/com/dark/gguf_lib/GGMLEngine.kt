package com.dark.gguf_lib

import android.content.Context
import android.net.Uri
import com.dark.gguf_lib.models.DecodingMetrics
import com.dark.gguf_lib.models.GenerationEvent
import com.dark.gguf_lib.models.StreamCallback
import com.dark.gguf_lib.toolcalling.GrammarMode
import com.dark.gguf_lib.toolcalling.ToolCallingConfig
import com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray

/**
 * GGMLEngine - High-level LLM inference SDK for Android.
 *
 * Wraps llama.cpp via JNI with Flow-based streaming generation,
 * tool calling, persona engine, embeddings, and KV cache state persistence.
 *
 * Usage:
 * ```
 * val engine = GGMLEngine()
 *
 * // Load model
 * engine.load("/path/to/model.gguf")
 *
 * // Configure sampling
 * engine.setSampling(temperature = 0.7f, topP = 0.9f)
 *
 * // Stream generation
 * engine.generateFlow("Hello!", maxTokens = 512).collect { event ->
 *     when (event) {
 *         is TextEvent.Token -> print(event.text)
 *         is TextEvent.Done -> println("\nDone")
 *         is TextEvent.Metrics -> println("${event.tokensPerSecond} t/s")
 *         is TextEvent.ToolCall -> handleTool(event.name, event.argsJson)
 *         is TextEvent.Error -> println("Error: ${event.message}")
 *     }
 * }
 *
 * // Cleanup
 * engine.unload()
 * ```
 */
class GGMLEngine {

    @Volatile private var loaded = false

    // ---- Model Loading ----

    /**
     * Load a GGUF model from file path.
     *
     * @param path Absolute path to the .gguf file
     * @param contextSize Context window size (default 4096)
     * @param threadMode Thread mode: 0=power_saving, 1=balanced, 2=performance
     * @param flashAttn Enable flash attention
     * @param cacheTypeK KV cache type for keys: "f16", "q8_0", "q4_0", etc.
     * @param cacheTypeV KV cache type for values
     */
    fun load(
        path: String,
        contextSize: Int = 4096,
        threadMode: Int = 1,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        loaded = GGUFNativeLib.nativeLoadModel(path, contextSize, threadMode, flashAttn, cacheTypeK, cacheTypeV)
        return loaded
    }

    /**
     * Load a GGUF model from Android file descriptor (for SAF/content:// URIs).
     */
    fun loadFromFd(
        fd: Int,
        contextSize: Int = 4096,
        threadMode: Int = 1,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        loaded = GGUFNativeLib.nativeLoadModelFromFd(fd, contextSize, threadMode, flashAttn, cacheTypeK, cacheTypeV)
        return loaded
    }

    /**
     * Load a GGUF model from a content:// URI via SAF.
     */
    fun load(
        context: Context,
        uri: Uri,
        contextSize: Int = 4096,
        threadMode: Int = 1,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return false
        return try {
            loadFromFd(pfd.fd, contextSize, threadMode, flashAttn, cacheTypeK, cacheTypeV)
        } finally {
            pfd.close()
        }
    }

    /**
     * Switch thread mode at runtime without reloading the model.
     * @param mode 0=power_saving, 1=balanced, 2=performance
     */
    fun setThreadMode(mode: Int) = GGUFNativeLib.nativeSetThreadMode(mode)

    /**
     * Set token accumulation size before each callback (JNI/AIDL Binder call).
     * Tune this based on your IPC cost:
     * - Direct in-process JNI: 64 bytes (~1 token) for low latency
     * - AIDL service: 256-512 bytes to amortize Binder overhead (~20-50µs/call)
     */
    fun setTokenBatchSize(bytes: Int) = GGUFNativeLib.nativeSetTokenBatchSize(bytes)

    /**
     * Configure StreamingLLM-style KV eviction policy.
     *
     * When the context fills up, instead of a hard stop:
     * - Keeps [0, nSink) tokens (attention sinks — typically 4)
     * - Keeps the most recent [nPast-nWindow, nPast) tokens
     * - Evicts everything in between
     *
     * Set nWindow=0 to disable (default — falls back to normal context shift).
     *
     * @param nSink   Tokens at the start to never evict. Default: 4.
     * @param nWindow Max recency window. 0 = disabled.
     * @param evictAtFull Auto-evict at generation start if context would overflow.
     */
    fun setKvPolicy(nSink: Int = 4, nWindow: Int = 0, evictAtFull: Boolean = false) =
        GGUFNativeLib.nativeSetKvPolicy(nSink, nWindow, evictAtFull)

    /**
     * Apply KV eviction immediately (SnapKV-style post-prefill budget enforcement).
     * Call after loading a long system prompt to trim KV cache to the policy window.
     */
    fun evictToBudget() = GGUFNativeLib.nativeEvictToBudget()

    /**
     * Release the loaded model and free all resources.
     * Runs on [Dispatchers.IO] — `nativeRelease()` blocks for the duration of
     * KV cache + context teardown (can be hundreds of ms).
     */
    suspend fun unload() = withContext(Dispatchers.IO) {
        if (loaded) {
            GGUFNativeLib.nativeRelease()
            loaded = false
        }
    }

    val isLoaded: Boolean get() = loaded

    // ---- Model Info ----

    /**
     * Get model metadata as JSON string.
     * Contains: description, n_ctx, n_params, model_size, name, architecture, n_vocab
     */
    fun getModelInfoJson(): String? = if (loaded) GGUFNativeLib.nativeGetModelInfo() else null

    /**
     * Check if the loaded model supports thinking/reasoning blocks.
     */
    fun supportsThinking(): Boolean = loaded && GGUFNativeLib.nativeSupportsThinking()

    fun setThinkingEnabled(enabled: Boolean) {
        GGUFNativeLib.nativeSetThinkingEnabled(enabled)
    }

    // ---- Sampling Configuration ----

    /**
     * Set core sampling parameters.
     */
    fun setSampling(
        temperature: Float = 0.7f,
        topK: Int = 40,
        topP: Float = 0.9f,
        minP: Float = 0.05f,
        mirostat: Int = 0,
        mirostatTau: Float = 5.0f,
        mirostatEta: Float = 0.1f,
        seed: Int = -1,
    ) {
        GGUFNativeLib.nativeSetSampling(temperature, topK, topP, minP, mirostat, mirostatTau, mirostatEta, seed)
    }

    /**
     * Update sampling parameters dynamically from JSON.
     * Accepts both camelCase and snake_case keys.
     * Supports: temperature, topK/top_k, topP/top_p, minP/min_p,
     * repeatPenalty, frequencyPenalty, presencePenalty, penaltyLastN,
     * dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
     * xtcProbability, xtcThreshold, mirostat, mirostatTau, mirostatEta, seed
     */
    fun updateSamplerParams(paramsJson: String): Boolean =
        GGUFNativeLib.nativeUpdateSamplerParams(paramsJson)

    /**
     * Set per-token logit biases.
     * @param biasJson JSON object {"token_id": bias} or array [{"token": id, "bias": val}]
     */
    fun setLogitBias(biasJson: String) = GGUFNativeLib.nativeSetLogitBias(biasJson)

    fun setSystemPrompt(prompt: String) = GGUFNativeLib.nativeSetSystemPrompt(prompt)
    fun setChatTemplate(template: String) = GGUFNativeLib.nativeSetChatTemplate(template)

    // ---- Generation ----

    /**
     * Single-turn streaming generation as a Flow (raw library events).
     */
    fun generateRawFlow(prompt: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
        val job = launch(Dispatchers.IO) {
            val cb = object : StreamCallback {
                override fun onToken(token: String) { trySend(GenerationEvent.Token(token)) }
                override fun onToolCall(name: String, argsJson: String) { trySend(GenerationEvent.ToolCall(name, argsJson)) }
                override fun onDone() { trySend(GenerationEvent.Done); channel.close() }
                override fun onError(message: String) { trySend(GenerationEvent.Error(message)); channel.close() }
                override fun onProgress(progress: Float) { trySend(GenerationEvent.Progress(progress)) }
                override fun onMetrics(tps: Float, ttftMs: Float, totalMs: Float, tokensEvaluated: Int, tokensPredicted: Int, modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float) {
                    trySend(GenerationEvent.Metrics(DecodingMetrics(tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted, modelMB, ctxMB, peakMB, memPct)))
                }
            }
            GGUFNativeLib.nativeGenerateStream(prompt, maxTokens, cb)
        }
        awaitClose { job.cancel(); GGUFNativeLib.nativeStopGeneration() }
    }

    /**
     * Multi-turn streaming generation as a Flow (raw library events).
     * @param messagesJson JSON array of messages: [{"role":"user","content":"..."},...]
     */
    fun generateMultiTurnRawFlow(messagesJson: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
        val job = launch(Dispatchers.IO) {
            val cb = object : StreamCallback {
                override fun onToken(token: String) { trySend(GenerationEvent.Token(token)) }
                override fun onToolCall(name: String, argsJson: String) { trySend(GenerationEvent.ToolCall(name, argsJson)) }
                override fun onDone() { trySend(GenerationEvent.Done); channel.close() }
                override fun onError(message: String) { trySend(GenerationEvent.Error(message)); channel.close() }
                override fun onProgress(progress: Float) { trySend(GenerationEvent.Progress(progress)) }
                override fun onMetrics(tps: Float, ttftMs: Float, totalMs: Float, tokensEvaluated: Int, tokensPredicted: Int, modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float) {
                    trySend(GenerationEvent.Metrics(DecodingMetrics(tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted, modelMB, ctxMB, peakMB, memPct)))
                }
            }
            GGUFNativeLib.nativeGenerateStreamMultiTurn(messagesJson, maxTokens, cb)
        }
        awaitClose { job.cancel(); GGUFNativeLib.nativeStopGeneration() }
    }

    fun generateFlow(prompt: String, maxTokens: Int): Flow<GenerationEvent> =
        generateRawFlow(prompt, maxTokens)

    fun generateMultiTurnFlow(messagesJson: String, maxTokens: Int): Flow<GenerationEvent> =
        generateMultiTurnRawFlow(messagesJson, maxTokens)

    /**
     * Simple non-streaming generation. Returns the complete text.
     */
    suspend fun generate(prompt: String, maxTokens: Int = 4096): GenerationResult = withContext(Dispatchers.IO) {
        val text = StringBuilder()
        var metrics: DecodingMetrics? = null
        var error: String? = null

        val cb = object : StreamCallback {
            override fun onToken(token: String) { text.append(token) }
            override fun onToolCall(name: String, argsJson: String) {}
            override fun onDone() {}
            override fun onError(message: String) { error = message }
            override fun onMetrics(tps: Float, ttftMs: Float, totalMs: Float, tokensEvaluated: Int, tokensPredicted: Int, modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float) {
                metrics = DecodingMetrics(tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted, modelMB, ctxMB, peakMB, memPct)
            }
        }

        val ok = GGUFNativeLib.nativeGenerateStream(prompt, maxTokens, cb)
        GenerationResult(text = text.toString(), success = ok && error == null, metrics = metrics, error = error)
    }

    fun stopGeneration() = GGUFNativeLib.nativeStopGeneration()

    // ---- Tool Calling ----

    /**
     * Enable tool calling with a list of tool definitions (convenience overload).
     */
    fun enableToolCalling(
        tools: List<ToolDefinitionBuilder.ToolDefinition>,
        config: ToolCallingConfig = ToolCallingConfig(),
    ) {
        val arr = JSONArray()
        tools.forEach { arr.put(it.toOpenAIFormat()) }
        GGUFNativeLib.nativeSetToolsJson(arr.toString())
        GGUFNativeLib.nativeSetGrammarMode(config.grammarMode.value)
        GGUFNativeLib.nativeSetTypedGrammar(config.useTypedGrammar)
    }

    /**
     * Set tools from raw OpenAI-format JSON string.
     */
    fun setToolsJson(toolsJson: String) = GGUFNativeLib.nativeSetToolsJson(toolsJson)

    /**
     * Disable tool calling.
     */
    fun clearTools() = GGUFNativeLib.nativeSetToolsJson("")

    fun isToolCallingSupported(): Boolean = loaded && GGUFNativeLib.nativeIsToolCallingSupported()

    // ---- Control Vectors ----

    /**
     * Load control vectors for personality/emotional tuning.
     * @param vectorsJson JSON array [{"path":"/path/to/vec.gguf","scale":1.0}]
     */
    fun loadControlVectors(vectorsJson: String): Boolean =
        GGUFNativeLib.nativeLoadControlVectors(vectorsJson)

    fun clearControlVector() = GGUFNativeLib.nativeClearControlVector()

    fun getStateSize(): Long = if (loaded) GGUFNativeLib.nativeGetStateSize() else 0
    fun stateSaveToFile(path: String): Boolean = GGUFNativeLib.nativeStateSaveToFile(path)
    fun stateLoadFromFile(path: String): Boolean = GGUFNativeLib.nativeStateLoadFromFile(path)

    // get current KV cache utilization (0.0 = empty, 1.0 = full)
    fun getContextUsage(): Float = if (loaded) GGUFNativeLib.nativeGetContextUsage() else 0f

    // ---- Optimization Controls ----

    /**
     * Set directory for disk-backed prompt cache.
     * When set, system prompt KV state is saved/restored across sessions,
     * eliminating re-evaluation of the system prompt on cold starts.
     */
    fun setPromptCacheDir(path: String) = GGUFNativeLib.nativeSetPromptCacheDir(path)

    /**
     * Run a warm-up decode pass to fault-in model weight pages.
     * Called automatically during load(), but can be re-invoked manually.
     */
    fun warmUp(): Boolean = if (loaded) GGUFNativeLib.nativeWarmUp() else false

    // ---- VLM (Vision Language Model) ----

    @Volatile private var vlmLoaded = false

    /**
     * Load a vision projector (mmproj GGUF). Must be called after loading the text model.
     *
     * @param path Absolute path to the mmproj .gguf file
     * @param threads Threads for vision encoding (0 = auto)
     * @param imageMinTokens Minimum image-token budget (-1 = model default)
     * @param imageMaxTokens Maximum image-token budget for the overview image (-1 = model default).
     *        Lowering this caps overview resolution. For LFM2-VL it does NOT cap the tile grid;
     *        the tile count is a compile-time constant in clip.cpp.
     */
    fun loadVlmProjector(
        path: String,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1
    ): Boolean {
        if (!loaded) return false
        vlmLoaded = GGUFNativeLib.nativeVlmLoadProjector(path, threads, imageMinTokens, imageMaxTokens)
        return vlmLoaded
    }

    /**
     * Load a vision projector from Android file descriptor.
     */
    fun loadVlmProjectorFromFd(
        fd: Int,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1
    ): Boolean {
        if (!loaded) return false
        vlmLoaded = GGUFNativeLib.nativeVlmLoadProjectorFromFd(fd, threads, imageMinTokens, imageMaxTokens)
        return vlmLoaded
    }

    /**
     * Load a vision projector from a content:// URI via SAF.
     */
    fun loadVlmProjector(
        context: Context,
        uri: Uri,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1
    ): Boolean {
        if (!loaded) return false
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return false
        return try {
            loadVlmProjectorFromFd(pfd.fd, threads, imageMinTokens, imageMaxTokens)
        } finally {
            pfd.close()
        }
    }

    fun releaseVlmProjector() {
        if (vlmLoaded) {
            GGUFNativeLib.nativeVlmRelease()
            vlmLoaded = false
        }
    }

    val isVlmLoaded: Boolean get() = vlmLoaded

    /**
     * Get VLM info as JSON string (supports_vision, supports_audio, default_marker).
     */
    fun getVlmInfoJson(): String? = if (vlmLoaded) GGUFNativeLib.nativeVlmGetInfo() else null

    /**
     * Get the default image marker to use in prompts (e.g. "<__image__>").
     */
    fun getVlmDefaultMarker(): String = GGUFNativeLib.nativeVlmGetDefaultMarker()

    /**
     * Stream generation with text + images (raw library events).
     * @param messagesJson JSON array of chat messages. User message content should
     *                     contain the image marker where each image should appear.
     * @param imageData List of raw image file bytes (JPEG/PNG)
     */
    fun generateVlmFlow(
        messagesJson: String,
        imageData: List<ByteArray>,
        maxTokens: Int = 4096
    ): Flow<GenerationEvent> = callbackFlow {
        val job = launch(Dispatchers.IO) {
            val cb = object : StreamCallback {
                override fun onToken(token: String) { trySend(GenerationEvent.Token(token)) }
                override fun onToolCall(name: String, argsJson: String) { trySend(GenerationEvent.ToolCall(name, argsJson)) }
                override fun onDone() { trySend(GenerationEvent.Done); channel.close() }
                override fun onError(message: String) { trySend(GenerationEvent.Error(message)); channel.close() }
                override fun onProgress(progress: Float) { trySend(GenerationEvent.Progress(progress)) }
                override fun onMetrics(tps: Float, ttftMs: Float, totalMs: Float, tokensEvaluated: Int, tokensPredicted: Int, modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float) {
                    trySend(GenerationEvent.Metrics(DecodingMetrics(tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted, modelMB, ctxMB, peakMB, memPct)))
                }
                override fun onVlmStageMetrics(vlmEncodeMs: Float, vlmDecodeMs: Float, imageTokens: Int) {
                    trySend(GenerationEvent.VlmStageMetrics(vlmEncodeMs, vlmDecodeMs, imageTokens))
                }
            }
            GGUFNativeLib.nativeVlmGenerateStream(
                messagesJson, imageData.toTypedArray(), maxTokens, cb
            )
        }
        awaitClose { job.cancel(); GGUFNativeLib.nativeStopGeneration() }
    }

    // ---- Device Tier ----

    companion object {
        /**
         * Detect device capability tier based on available RAM.
         */
        fun detectDeviceTier(context: Context): DeviceTier {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val memInfo = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(memInfo)
            val totalGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)
            return when {
                totalGB < 4.0 -> DeviceTier.LOW_END
                totalGB < 8.0 -> DeviceTier.MID_RANGE
                else -> DeviceTier.HIGH_END
            }
        }

        /**
         * Get recommended loading parameters for the device.
         */
        fun getRecommendedParams(context: Context): LoadingParams {
            return when (detectDeviceTier(context)) {
                DeviceTier.LOW_END -> LoadingParams(contextSize = 2048, threadMode = 0, cacheTypeK = "q4_0", cacheTypeV = "q4_0")
                DeviceTier.MID_RANGE -> LoadingParams(contextSize = 4096, threadMode = 1, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
                DeviceTier.HIGH_END -> LoadingParams(contextSize = 8192, threadMode = 2, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
            }
        }
    }
}

// ---- Data classes ----

enum class DeviceTier { LOW_END, MID_RANGE, HIGH_END }

data class LoadingParams(
    val contextSize: Int = 4096,
    val threadMode: Int = 1,
    val flashAttn: Boolean = false,
    val cacheTypeK: String = "q8_0",
    val cacheTypeV: String = "q8_0",
)

data class GenerationResult(
    val text: String,
    val success: Boolean,
    val metrics: DecodingMetrics? = null,
    val error: String? = null,
)
