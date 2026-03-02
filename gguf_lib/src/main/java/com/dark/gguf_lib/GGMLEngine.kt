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
 *         is GenerationEvent.Token -> print(event.text)
 *         is GenerationEvent.Done -> println("\nDone")
 *         is GenerationEvent.Metrics -> println("${event.metrics.tokensPerSecond} t/s")
 *         is GenerationEvent.ToolCall -> handleTool(event.name, event.argsJson)
 *         is GenerationEvent.Error -> println("Error: ${event.message}")
 *     }
 * }
 *
 * // Cleanup
 * engine.unload()
 * ```
 */
class GGMLEngine {

    private var loaded = false

    // ---- Model Loading ----

    /**
     * Load a GGUF model from file path.
     *
     * @param path Absolute path to the .gguf file
     * @param contextSize Context window size (default 4096)
     * @param threads Number of threads (0 = auto-detect, recommended)
     * @param flashAttn Enable flash attention (may crash on some ARM devices with q8_0)
     * @param cacheTypeK KV cache type for keys: "f16", "q8_0", "q4_0", etc.
     * @param cacheTypeV KV cache type for values
     */
    fun load(
        path: String,
        contextSize: Int = 4096,
        threads: Int = 0,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        loaded = GGUFNativeLib.nativeLoadModel(path, contextSize, threads, flashAttn, cacheTypeK, cacheTypeV)
        return loaded
    }

    /**
     * Load a GGUF model from Android file descriptor (for SAF/content:// URIs).
     */
    fun loadFromFd(
        fd: Int,
        contextSize: Int = 4096,
        threads: Int = 0,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        loaded = GGUFNativeLib.nativeLoadModelFromFd(fd, contextSize, threads, flashAttn, cacheTypeK, cacheTypeV)
        return loaded
    }

    /**
     * Load a GGUF model from a content:// URI via SAF.
     */
    fun load(
        context: Context,
        uri: Uri,
        contextSize: Int = 4096,
        threads: Int = 0,
        flashAttn: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return false
        return try {
            loadFromFd(pfd.fd, contextSize, threads, flashAttn, cacheTypeK, cacheTypeV)
        } finally {
            pfd.close()
        }
    }

    /**
     * Release the loaded model and free all resources.
     */
    fun unload() {
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
     * Single-turn streaming generation as a Flow.
     */
    fun generateFlow(prompt: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
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
     * Multi-turn streaming generation as a Flow.
     * @param messagesJson JSON array of messages: [{"role":"user","content":"..."},...]
     */
    fun generateMultiTurnFlow(messagesJson: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
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
     * Enable tool calling with a list of tool definitions.
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

    // ---- KV Cache State ----

    fun getStateSize(): Long = if (loaded) GGUFNativeLib.nativeGetStateSize() else 0
    fun stateSaveToFile(path: String): Boolean = GGUFNativeLib.nativeStateSaveToFile(path)
    fun stateLoadFromFile(path: String): Boolean = GGUFNativeLib.nativeStateLoadFromFile(path)

    // get current KV cache utilization (0.0 = empty, 1.0 = full)
    fun getContextUsage(): Float = if (loaded) GGUFNativeLib.nativeGetContextUsage() else 0f

    // ---- Optimization Controls ----

    /**
     * Enable/disable ngram self-speculative decoding.
     * For repetitive/structured output (JSON, code, lists), yields 1.3-2x throughput.
     * @param enabled true to enable
     * @param nDraft max tokens to draft per step (default 4)
     * @param ngramSize ngram size for history lookup (default 4)
     */
    fun setSpeculativeDecoding(enabled: Boolean, nDraft: Int = 4, ngramSize: Int = 4) =
        GGUFNativeLib.nativeSetSpeculativeDecoding(enabled, nDraft, ngramSize)

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
                DeviceTier.LOW_END -> LoadingParams(contextSize = 2048, threads = 0, cacheTypeK = "q4_0", cacheTypeV = "q4_0")
                DeviceTier.MID_RANGE -> LoadingParams(contextSize = 4096, threads = 0, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
                DeviceTier.HIGH_END -> LoadingParams(contextSize = 8192, threads = 0, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
            }
        }
    }
}

// ---- Data classes ----

enum class DeviceTier { LOW_END, MID_RANGE, HIGH_END }

data class LoadingParams(
    val contextSize: Int = 4096,
    val threads: Int = 0,
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
