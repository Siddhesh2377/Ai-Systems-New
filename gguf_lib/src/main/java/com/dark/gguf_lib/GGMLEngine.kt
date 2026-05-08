package com.dark.gguf_lib

import android.content.Context
import android.net.Uri
import com.dark.gguf_lib.models.DecodingMetrics
import com.dark.gguf_lib.models.GenerationEvent
import com.dark.gguf_lib.models.StreamCallback
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
 * On-device GGUF inference engine — primary entry point of the SDK.
 *
 * Wraps llama.cpp via JNI with Flow-based streaming generation, tool calling,
 * a sampler-level personality engine, embeddings, RAG, VLM (vision), and KV
 * cache state persistence. One model is loaded at a time, app-wide.
 *
 * Threading
 * ---------
 * Native calls that touch the model (load, generate, unload) are serialized
 * by an internal mutex on the C++ side. The Kotlin wrappers offload blocking
 * calls to [Dispatchers.IO] via `withContext`. Streaming flows use
 * `callbackFlow` and call [stopGeneration] on close, so cancelling the
 * collecting coroutine cleanly halts generation.
 *
 * Lifecycle
 * ---------
 * Construct, [load] (or [loadFromFd]), use, [unload]. The class is not
 * `Closeable` because [unload] is `suspend` — call it explicitly from a
 * coroutine when you're done.
 *
 * ```kotlin
 * val engine = GGMLEngine()
 * engine.load("/path/to/model.gguf")
 * engine.generateFlow("Hello", maxTokens = 256).collect { event ->
 *     if (event is GenerationEvent.Token) print(event.text)
 * }
 * engine.unload()
 * ```
 */
class GGMLEngine {

    @Volatile private var loaded = false
    @Volatile private var vlmLoaded = false

    /**
     * Load a GGUF model from a local file path.
     *
     * @param path        Absolute path to the .gguf file. Must be readable and seekable.
     * @param contextSize Context window size in tokens. Caps how much conversation
     *                    fits in the KV cache; bigger = more RAM.
     * @param threads     Generation threads. 0 = auto from current thread mode.
     *                    A positive value forces a literal thread count for both
     *                    generation and batch decode.
     * @param batchSize   Prompt-eval batch size. 0 = auto from thread mode (recommended).
     *                    Larger batches use more memory but speed up long prompts.
     * @param flashAttn   Enable flash attention. Reduces memory bandwidth on long
     *                    contexts; can crash on some ARM devices with certain
     *                    cache types.
     * @param useMmap     Memory-map the model file (default true). Disable on
     *                    devices where the OS aggressively evicts mapped pages.
     * @param useMlock    Lock model pages in RAM (default false). Prevents
     *                    swap-out at the cost of fixed memory usage; requires
     *                    sufficient unprivileged mlock budget on the device.
     * @param cacheTypeK  KV cache type for keys: `f32`, `f16`, `q8_0`, `q4_0`,
     *                    `q4_1`, `q5_0`, `q5_1`. Defaults to `q8_0` (~50% of f16).
     * @param cacheTypeV  KV cache type for values. Same options as [cacheTypeK].
     *
     * @return true on success. On failure, see [com.dark.gguf_lib.error] via
     *         `GGUFNativeLib.nativeErrorGetLastJson()` (used internally).
     */
    suspend fun load(
        path: String,
        contextSize: Int = 4096,
        threads: Int = 0,
        batchSize: Int = 0,
        flashAttn: Boolean = false,
        useMmap: Boolean = true,
        useMlock: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean = withContext(Dispatchers.IO) {
        loaded = GGUFNativeLib.nativeLoadModel(
            path, contextSize, threads, batchSize,
            flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV,
        )
        loaded
    }

    /**
     * Load a GGUF model from an Android file descriptor.
     *
     * The native side `dup()`s the fd so the caller's [java.io.FileDescriptor]
     * (typically a `ParcelFileDescriptor`) is safe to close immediately after.
     * The fd must be seekable — SAF pipe-based providers are not supported.
     */
    suspend fun loadFromFd(
        fd: Int,
        contextSize: Int = 4096,
        threads: Int = 0,
        batchSize: Int = 0,
        flashAttn: Boolean = false,
        useMmap: Boolean = true,
        useMlock: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean = withContext(Dispatchers.IO) {
        loaded = GGUFNativeLib.nativeLoadModelFromFd(
            fd, contextSize, threads, batchSize,
            flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV,
        )
        loaded
    }

    /**
     * Load a GGUF model from a SAF `content://` URI.
     *
     * Opens the URI with `ContentResolver.openFileDescriptor()`, calls
     * [loadFromFd], and closes the [android.os.ParcelFileDescriptor] when done.
     */
    suspend fun load(
        context: Context,
        uri: Uri,
        contextSize: Int = 4096,
        threads: Int = 0,
        batchSize: Int = 0,
        flashAttn: Boolean = false,
        useMmap: Boolean = true,
        useMlock: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
    ): Boolean = withContext(Dispatchers.IO) {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return@withContext false
        try {
            loadFromFd(
                pfd.fd, contextSize, threads, batchSize,
                flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV,
            )
        } finally {
            pfd.close()
        }
    }

    /**
     * Switch the thread profile at runtime. Cheap; safe to call between turns.
     *
     * @param mode 0 = power saving, 1 = balanced, 2 = performance.
     *
     * Note: the VLM projector binds `n_threads` at init time. Switching modes
     * here does NOT update the projector — call [releaseVlmProjector] then
     * [loadVlmProjector] to re-bind.
     */
    fun setThreadMode(mode: Int) = GGUFNativeLib.nativeSetThreadMode(mode)

    /**
     * Tune the token-batching threshold (bytes accumulated before each callback).
     *
     * - 64  for direct in-process JNI (lowest latency)
     * - 256 default
     * - 512 or higher for AIDL services (Binder IPC ~20-50us per call)
     */
    fun setTokenBatchSize(bytes: Int) = GGUFNativeLib.nativeSetTokenBatchSize(bytes)

    /**
     * Configure StreamingLLM-style KV eviction. When the context fills:
     *
     * - tokens `[0, nSink)` are kept (attention sinks — typically 4)
     * - tokens `[n_past - nWindow, n_past)` are kept (recency window)
     * - everything in between is evicted
     *
     * Set `nWindow = 0` to disable; the engine then falls back to a simple
     * half-discard context shift.
     */
    fun setKvPolicy(nSink: Int = 4, nWindow: Int = 0, evictAtFull: Boolean = false) =
        GGUFNativeLib.nativeSetKvPolicy(nSink, nWindow, evictAtFull)

    /** Apply the configured KV eviction immediately (SnapKV-style post-prefill trim). */
    fun evictToBudget() = GGUFNativeLib.nativeEvictToBudget()

    /**
     * Release the loaded model and free all native resources. Safe to call
     * multiple times. Blocks for the duration of KV cache + context teardown
     * (potentially hundreds of ms), so runs on [Dispatchers.IO].
     */
    suspend fun unload() = withContext(Dispatchers.IO) {
        if (loaded) {
            GGUFNativeLib.nativeRelease()
            loaded = false
        }
    }

    val isLoaded: Boolean get() = loaded

    /**
     * Model metadata as a JSON string, or null if no model is loaded.
     *
     * Keys: `description`, `n_ctx`, `n_params`, `model_size`, `name`,
     * `architecture`, `file_type`, `n_vocab`.
     */
    fun getModelInfoJson(): String? = if (loaded) GGUFNativeLib.nativeGetModelInfo() else null

    /** Returns true if the model's chat template advertises a thinking/reasoning mode. */
    fun supportsThinking(): Boolean = loaded && GGUFNativeLib.nativeSupportsThinking()

    /** Toggle thinking-block emission for templates that support it (Qwen3, etc.). */
    fun setThinkingEnabled(enabled: Boolean) = GGUFNativeLib.nativeSetThinkingEnabled(enabled)

    /**
     * Set core sampling parameters. Lower-effort callers can use this; richer
     * configurations (DRY, XTC, repetition penalty) are exposed via
     * [updateSamplerParams].
     *
     * @param seed -1 for a random seed.
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
        GGUFNativeLib.nativeSetSampling(
            temperature, topK, topP, minP, mirostat, mirostatTau, mirostatEta, seed,
        )
    }

    /**
     * Update sampler parameters from a JSON string. Accepts both camelCase
     * and snake_case keys; unknown keys are ignored. Recognized keys:
     *
     * `temperature`, `topK`/`top_k`, `topP`/`top_p`, `minP`/`min_p`,
     * `repeatPenalty`, `frequencyPenalty`, `presencePenalty`, `penaltyLastN`,
     * `dryMultiplier`, `dryBase`, `dryAllowedLength`, `dryPenaltyLastN`,
     * `xtcProbability`, `xtcThreshold`, `mirostat`, `mirostatTau`,
     * `mirostatEta`, `seed`.
     *
     * @return true on success, false if the JSON failed to parse.
     */
    fun updateSamplerParams(paramsJson: String): Boolean =
        GGUFNativeLib.nativeUpdateSamplerParams(paramsJson)

    /**
     * Set per-token logit biases.
     *
     * @param biasJson Either an object `{"token_id": bias}` or array
     *                 `[{"token": id_or_string, "bias": float}, ...]`.
     */
    fun setLogitBias(biasJson: String) = GGUFNativeLib.nativeSetLogitBias(biasJson)

    /** Set the system prompt prepended to every chat. */
    fun setSystemPrompt(prompt: String) = GGUFNativeLib.nativeSetSystemPrompt(prompt)

    /** Override the model's chat template (advanced — usually not needed). */
    fun setChatTemplate(template: String) = GGUFNativeLib.nativeSetChatTemplate(template)

    /**
     * Single-turn streaming generation as a [Flow] of [GenerationEvent].
     *
     * The flow emits [GenerationEvent.Token] chunks during decode, optionally
     * [GenerationEvent.ToolCall] / [GenerationEvent.Progress] /
     * [GenerationEvent.Metrics] / [GenerationEvent.Error], and terminates with
     * [GenerationEvent.Done]. Cancelling the collector calls [stopGeneration]
     * on the native side.
     */
    fun generateFlow(prompt: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
        val cb = streamCallback(::trySend, ::close)
        val job = launch(Dispatchers.IO) {
            GGUFNativeLib.nativeGenerateStream(prompt, maxTokens, cb)
        }
        awaitClose {
            job.cancel()
            GGUFNativeLib.nativeStopGeneration()
        }
    }

    /**
     * Multi-turn streaming generation. Same event model as [generateFlow].
     *
     * @param messagesJson JSON array of `{role, content}` objects. Roles:
     *                     `system`, `user`, `assistant`, `tool`. Anything else
     *                     is remapped to `assistant` on the native side.
     */
    fun generateMultiTurnFlow(messagesJson: String, maxTokens: Int = 4096): Flow<GenerationEvent> = callbackFlow {
        val cb = streamCallback(::trySend, ::close)
        val job = launch(Dispatchers.IO) {
            GGUFNativeLib.nativeGenerateStreamMultiTurn(messagesJson, maxTokens, cb)
        }
        awaitClose {
            job.cancel()
            GGUFNativeLib.nativeStopGeneration()
        }
    }

    /** Non-streaming wrapper around [generateFlow] — collects all tokens into a [GenerationResult]. */
    suspend fun generate(prompt: String, maxTokens: Int = 4096): GenerationResult = withContext(Dispatchers.IO) {
        val text = StringBuilder()
        var metrics: DecodingMetrics? = null
        var error: String? = null

        val cb = object : StreamCallback {
            override fun onToken(token: String) { text.append(token) }
            override fun onToolCall(name: String, argsJson: String) {}
            override fun onDone() {}
            override fun onError(message: String) { error = message }
            override fun onMetrics(
                tps: Float, ttftMs: Float, totalMs: Float,
                tokensEvaluated: Int, tokensPredicted: Int,
                modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float,
            ) {
                metrics = DecodingMetrics(
                    tps, ttftMs, totalMs,
                    tokensEvaluated, tokensPredicted,
                    modelMB, ctxMB, peakMB, memPct,
                )
            }
        }

        val ok = GGUFNativeLib.nativeGenerateStream(prompt, maxTokens, cb)
        GenerationResult(text = text.toString(), success = ok && error == null, metrics = metrics, error = error)
    }

    /** Request the current generation to stop. Idempotent; cheap. */
    fun stopGeneration() = GGUFNativeLib.nativeStopGeneration()

    /**
     * Enable tool calling with a list of tool definitions.
     *
     * @param config Grammar mode + typed grammar toggle. STRICT forces a tool
     *               call; LAZY lets the model choose between text and tools.
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

    /** Set tools from a raw OpenAI-format JSON string. Pass `""` to disable. */
    fun setToolsJson(toolsJson: String) = GGUFNativeLib.nativeSetToolsJson(toolsJson)

    /** Disable tool calling. */
    fun clearTools() = GGUFNativeLib.nativeSetToolsJson("")

    /** Whether the loaded model's chat template supports tool calling. */
    fun isToolCallingSupported(): Boolean = loaded && GGUFNativeLib.nativeIsToolCallingSupported()

    /**
     * Load control vectors for personality / emotional tuning.
     *
     * @param vectorsJson JSON array `[{"path":"/path/to/vec.gguf","scale":1.0}, ...]`.
     */
    fun loadControlVectors(vectorsJson: String): Boolean =
        GGUFNativeLib.nativeLoadControlVectors(vectorsJson)

    /** Clear any active control vectors. */
    fun clearControlVector() = GGUFNativeLib.nativeClearControlVector()

    /** Bytes needed to serialize the current KV cache state. */
    fun getStateSize(): Long = if (loaded) GGUFNativeLib.nativeGetStateSize() else 0

    /** Persist the full KV cache + token list to a file. */
    fun stateSaveToFile(path: String): Boolean = GGUFNativeLib.nativeStateSaveToFile(path)

    /** Restore a previously saved KV cache. The model must match. */
    fun stateLoadFromFile(path: String): Boolean = GGUFNativeLib.nativeStateLoadFromFile(path)

    /** Current KV cache fill: 0.0 = empty, 1.0 = full. */
    fun getContextUsage(): Float = if (loaded) GGUFNativeLib.nativeGetContextUsage() else 0f

    /**
     * Set the directory used for the disk-backed prompt cache. When set, the
     * system prompt KV state is saved/restored across sessions, eliminating
     * re-evaluation of the system prompt on cold starts.
     */
    fun setPromptCacheDir(path: String) = GGUFNativeLib.nativeSetPromptCacheDir(path)

    /**
     * Run a warm-up decode to fault-in model weight pages. Called automatically
     * during [load]; expose here for callers that want to explicitly re-warm
     * after long idle periods.
     */
    fun warmUp(): Boolean = if (loaded) GGUFNativeLib.nativeWarmUp() else false

    /**
     * Load a vision projector (mmproj GGUF). Must be called after the text model.
     *
     * @param threads        0 = inherit the engine's batch threads.
     * @param imageMinTokens -1 = model default.
     * @param imageMaxTokens -1 = model default. Caps the *overview* image only;
     *                       per-tile counts are compile-time constants.
     */
    suspend fun loadVlmProjector(
        path: String,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1,
    ): Boolean = withContext(Dispatchers.IO) {
        if (!loaded) return@withContext false
        vlmLoaded = GGUFNativeLib.nativeVlmLoadProjector(path, threads, imageMinTokens, imageMaxTokens)
        vlmLoaded
    }

    /** Load a vision projector from a file descriptor. See [loadVlmProjector]. */
    suspend fun loadVlmProjectorFromFd(
        fd: Int,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1,
    ): Boolean = withContext(Dispatchers.IO) {
        if (!loaded) return@withContext false
        vlmLoaded = GGUFNativeLib.nativeVlmLoadProjectorFromFd(fd, threads, imageMinTokens, imageMaxTokens)
        vlmLoaded
    }

    /** Load a vision projector from a SAF `content://` URI. */
    suspend fun loadVlmProjector(
        context: Context,
        uri: Uri,
        threads: Int = 0,
        imageMinTokens: Int = -1,
        imageMaxTokens: Int = -1,
    ): Boolean = withContext(Dispatchers.IO) {
        if (!loaded) return@withContext false
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return@withContext false
        try {
            loadVlmProjectorFromFd(pfd.fd, threads, imageMinTokens, imageMaxTokens)
        } finally {
            pfd.close()
        }
    }

    /** Release the vision projector. The text model stays loaded. */
    fun releaseVlmProjector() {
        if (vlmLoaded) {
            GGUFNativeLib.nativeVlmRelease()
            vlmLoaded = false
        }
    }

    val isVlmLoaded: Boolean get() = vlmLoaded

    /** VLM info JSON: `{supports_vision, supports_audio, default_marker}`. */
    fun getVlmInfoJson(): String? = if (vlmLoaded) GGUFNativeLib.nativeVlmGetInfo() else null

    /** Default image marker to embed in prompts (e.g. `<__image__>`). */
    fun getVlmDefaultMarker(): String = GGUFNativeLib.nativeVlmGetDefaultMarker()

    /**
     * Stream generation with text + images. The user message content should
     * contain the marker from [getVlmDefaultMarker] at each image's position.
     *
     * @param imageData Raw file bytes (JPEG/PNG) for each image.
     */
    fun generateVlmFlow(
        messagesJson: String,
        imageData: List<ByteArray>,
        maxTokens: Int = 4096,
    ): Flow<GenerationEvent> = callbackFlow {
        val cb = streamCallback(::trySend, ::close)
        val job = launch(Dispatchers.IO) {
            GGUFNativeLib.nativeVlmGenerateStream(
                messagesJson, imageData.toTypedArray(), maxTokens, cb,
            )
        }
        awaitClose {
            job.cancel()
            GGUFNativeLib.nativeStopGeneration()
        }
    }

    companion object {
        /** Categorize the host device by total RAM. */
        fun detectDeviceTier(context: Context): DeviceTier {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val memInfo = android.app.ActivityManager.MemoryInfo()
            am.getMemoryInfo(memInfo)
            val totalGB = memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)
            return when {
                totalGB < 4.0 -> DeviceTier.LOW_END
                totalGB < 8.0 -> DeviceTier.MID_RANGE
                else          -> DeviceTier.HIGH_END
            }
        }

        /** Conservative default loading parameters keyed off [detectDeviceTier]. */
        fun getRecommendedParams(context: Context): LoadingParams = when (detectDeviceTier(context)) {
            DeviceTier.LOW_END   -> LoadingParams(contextSize = 2048, cacheTypeK = "q4_0", cacheTypeV = "q4_0")
            DeviceTier.MID_RANGE -> LoadingParams(contextSize = 4096, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
            DeviceTier.HIGH_END  -> LoadingParams(contextSize = 8192, cacheTypeK = "q8_0", cacheTypeV = "q8_0")
        }
    }
}

// Builds a StreamCallback that forwards every native event onto the supplied
// channel-send + close lambdas. Used by all three generate Flows.
private inline fun streamCallback(
    crossinline send: (GenerationEvent) -> Unit,
    crossinline close: () -> Unit,
): StreamCallback = object : StreamCallback {
    override fun onToken(token: String) { send(GenerationEvent.Token(token)) }
    override fun onToolCall(name: String, argsJson: String) { send(GenerationEvent.ToolCall(name, argsJson)) }
    override fun onDone() { send(GenerationEvent.Done); close() }
    override fun onError(message: String) { send(GenerationEvent.Error(message)); close() }
    override fun onProgress(progress: Float) { send(GenerationEvent.Progress(progress)) }
    override fun onMetrics(
        tps: Float, ttftMs: Float, totalMs: Float,
        tokensEvaluated: Int, tokensPredicted: Int,
        modelMB: Float, ctxMB: Float, peakMB: Float, memPct: Float,
    ) {
        send(GenerationEvent.Metrics(DecodingMetrics(
            tps, ttftMs, totalMs, tokensEvaluated, tokensPredicted,
            modelMB, ctxMB, peakMB, memPct,
        )))
    }
    override fun onVlmStageMetrics(vlmEncodeMs: Float, vlmDecodeMs: Float, imageTokens: Int) {
        send(GenerationEvent.VlmStageMetrics(vlmEncodeMs, vlmDecodeMs, imageTokens))
    }
}

/** Coarse device buckets used by [GGMLEngine.detectDeviceTier]. */
enum class DeviceTier { LOW_END, MID_RANGE, HIGH_END }

/**
 * Recommended loading parameters (mirrors [GGMLEngine.load] arguments).
 * Returned by [GGMLEngine.getRecommendedParams].
 */
data class LoadingParams(
    val contextSize: Int = 4096,
    val threads: Int = 0,
    val batchSize: Int = 0,
    val flashAttn: Boolean = false,
    val useMmap: Boolean = true,
    val useMlock: Boolean = false,
    val cacheTypeK: String = "q8_0",
    val cacheTypeV: String = "q8_0",
)

/** Result of the non-streaming [GGMLEngine.generate]. */
data class GenerationResult(
    val text: String,
    val success: Boolean,
    val metrics: DecodingMetrics? = null,
    val error: String? = null,
)
