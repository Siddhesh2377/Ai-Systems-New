package com.dark.gguf_lib

import android.content.Context
import android.net.Uri
import com.dark.gguf_lib.models.DecodingMetrics
import com.dark.gguf_lib.models.GenerationEvent
import com.dark.gguf_lib.models.StreamCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * On-device GGUF inference engine — primary entry point of the SDK.
 *
 * Wraps llama.cpp via JNI with Flow-based streaming generation, embeddings,
 * RAG, VLM (vision), and KV cache state persistence. One model is loaded
 * at a time, app-wide.
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
     * @param opOffload   Per-op CPU/GPU routing. When true and a non-CPU
     *                    backend (Vulkan, etc.) is registered with ggml,
     *                    large ops (batch ≥ 32 by default) route to GPU
     *                    while single-token decode stays on CPU. No layer
     *                    weights are moved — purely a compute hint, so
     *                    decode latency is preserved. See VLM.md
     *                    "Per-op routing" for the full trade-off.
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
        opOffload: Boolean = false,
    ): Boolean = withContext(Dispatchers.IO) {
        loaded = GGUFNativeLib.nativeLoadModel(
            path, contextSize, threads, batchSize,
            flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV, opOffload,
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
        opOffload: Boolean = false,
    ): Boolean = withContext(Dispatchers.IO) {
        loaded = GGUFNativeLib.nativeLoadModelFromFd(
            fd, contextSize, threads, batchSize,
            flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV, opOffload,
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
        opOffload: Boolean = false,
    ): Boolean = withContext(Dispatchers.IO) {
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return@withContext false
        try {
            loadFromFd(
                pfd.fd, contextSize, threads, batchSize,
                flashAttn, useMmap, useMlock, cacheTypeK, cacheTypeV, opOffload,
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

    // ── Power engine + decode diagnostics ──────────────────────────────────

    /**
     * Per-stage breakdown of the last completed generate call's time. All
     * fields are aggregate microseconds across the run — divide by [tokens]
     * for per-token cost. Returned by [getLastDecodeBreakdown].
     *
     * @property tokens    Number of tokens decoded in the run.
     * @property sampleUs  Time inside the sampler chain (temp / top-k / etc).
     * @property detokUs   Time spent converting token IDs to UTF-8 bytes.
     * @property stopUs    Time spent matching antiprompt strings.
     * @property decodeUs  Time inside `llama_decode` — the model forward pass.
     *                     On a memory-bandwidth-bound model this dominates.
     * @property totalUs   Sum of the four; ~equal to wall time per token.
     */
    data class DecodeBreakdown(
        val tokens: Long,
        val sampleUs: Long,
        val detokUs: Long,
        val stopUs: Long,
        val decodeUs: Long,
        val totalUs: Long,
    )

    fun getLastDecodeBreakdown(): DecodeBreakdown {
        val raw = runCatching { GGUFNativeLib.nativeGetLastDecodeBreakdown() }
            .getOrNull() ?: "{}"
        val j = runCatching { org.json.JSONObject(raw) }.getOrNull() ?: org.json.JSONObject()
        return DecodeBreakdown(
            tokens   = j.optLong("tokens"),
            sampleUs = j.optLong("sample_us"),
            detokUs  = j.optLong("detok_us"),
            stopUs   = j.optLong("stop_us"),
            decodeUs = j.optLong("decode_us"),
            totalUs  = j.optLong("total_us"),
        )
    }

    /** Severity buckets that the power-engine maps the hottest SoC zone into. */
    enum class ThrottlingLevel(val value: Int) {
        COOL(0), WARM(1), HOT(2), CRITICAL(3);
        companion object {
            fun of(i: Int): ThrottlingLevel = values().firstOrNull { it.value == i } ?: COOL
        }
    }

    /**
     * Snapshot of the device's thermal state as seen by the engine.
     *
     * @property maxTempMilliC      Hottest *compute* zone in milli-Celsius. `-1`
     *                              when no /sys/class/thermal entries could be
     *                              read (sandboxed device or unsupported OS).
     * @property batteryTempMilliC  Battery zone reading, `-1` if unavailable.
     * @property level              Mapped severity. The auto-mode loop uses
     *                              this to decide whether to de-rate.
     * @property nZonesRead         How many thermal_zoneN entries were parsed.
     */
    data class ThermalState(
        val maxTempMilliC: Int,
        val batteryTempMilliC: Int,
        val level: ThrottlingLevel,
        val nZonesRead: Int,
    )

    fun getThermalState(): ThermalState {
        val raw = runCatching { GGUFNativeLib.nativeGetThermalState() }
            .getOrNull() ?: "{}"
        val j = runCatching { org.json.JSONObject(raw) }.getOrNull() ?: org.json.JSONObject()
        return ThermalState(
            maxTempMilliC     = j.optInt("maxTempMilliC", -1),
            batteryTempMilliC = j.optInt("batteryTempMilliC", -1),
            level             = ThrottlingLevel.of(j.optInt("throttlingLevel", 0)),
            nZonesRead        = j.optInt("nZonesRead", 0),
        )
    }

    /**
     * Enable auto-mode: the engine reads thermal state on each
     * [autoModeTick] and may step the effective thread mode down if the SoC
     * is hot. Disabling restores the user's requested mode.
     */
    fun setAutoMode(enabled: Boolean) = GGUFNativeLib.nativeSetAutoMode(enabled)
    fun isAutoModeEnabled(): Boolean = GGUFNativeLib.nativeIsAutoModeEnabled()

    /** The thread mode the engine is actually running (after any auto-mode adjustment). */
    fun getEffectiveThreadMode(): Int = GGUFNativeLib.nativeGetEffectiveThreadMode()

    /**
     * Override the per-level thermal thresholds (milli-Celsius). Pass `<=0`
     * for any field to keep the current value. Values are clamped to
     * `[30000, 110000]`. Defaults are tuned for Snapdragon 7-class SoCs:
     *
     *   warm=60000  hot=75000  crit=85000
     */
    fun setThermalThresholds(warmMilliC: Int, hotMilliC: Int, critMilliC: Int) =
        GGUFNativeLib.nativeSetThermalThresholds(warmMilliC, hotMilliC, critMilliC)

    /**
     * Tick the auto-mode loop. Polls thermal state; if auto-mode is on, may
     * adjust the effective thread mode. Returns the effective mode (0/1/2)
     * after the tick. Cheap (~100 us); host typically calls it before each
     * generate call. No-op when auto-mode is off.
     */
    fun autoModeTick(): Int = GGUFNativeLib.nativeAutoModeTick()

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
     * [GenerationEvent.Progress] / [GenerationEvent.Metrics] /
     * [GenerationEvent.Error], and terminates with [GenerationEvent.Done].
     * Cancelling the collector calls [stopGeneration] on the native side.
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
     *                     `system`, `user`, `assistant`. Anything else
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

    /** Bytes needed to serialize the current KV cache state. */
    fun getStateSize(): Long = if (loaded) GGUFNativeLib.nativeGetStateSize() else 0

    /** Persist the full KV cache + token list to a file. */
    fun stateSaveToFile(path: String): Boolean = GGUFNativeLib.nativeStateSaveToFile(path)

    /** Restore a previously saved KV cache. The model must match. */
    fun stateLoadFromFile(path: String): Boolean = GGUFNativeLib.nativeStateLoadFromFile(path)

    /** Current KV cache fill: 0.0 = empty, 1.0 = full. */
    fun getContextUsage(): Float = if (loaded) GGUFNativeLib.nativeGetContextUsage() else 0f

    /**
     * Comprehensive process-wide stats snapshot in one JSON blob. Includes
     * accurate resident memory (VmRSS/VmHWM, not virtual VmPeak), context
     * usage (raw n_ctx + n_used + percentage), and which sub-systems are
     * initialized (VT cache, VLM-KV cache, projector, model). Use for
     * diagnostic UIs.
     */
    fun getMemoryStatsJson(): String? = GGUFNativeLib.nativeGetMemoryStatsJson()

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
     * Run only the vision encoder for [imageBytes] and store the resulting
     * embeddings in the VT cache under a key derived the same way
     * [computeVtKey] does. No LLM context is touched.
     *
     * Use this to pre-warm the VT cache in the background — e.g. as soon as
     * the user picks/imports an image, kick this off so the first actual
     * query against the image hits the cache and skips the ~9s ViT pass on
     * Snapdragon 7s Gen 3.
     *
     * Suspends on [Dispatchers.IO]; returns true on successful encode + store.
     * Requires: [load] succeeded, [loadVlmProjector] succeeded, [vtCacheInit]
     * succeeded.
     */
    suspend fun precomputeVisionEmbeddings(
        imageBytes: ByteArray,
        projectorPath: String,
        imageMaxTokens: Int,
        imageQuality: ImageQuality = ImageQuality.MEDIUM,
    ): Boolean = withContext(Dispatchers.IO) {
        val key = computeVtKey(imageBytes, projectorPath, imageMaxTokens)
        GGUFNativeLib.nativeVlmPrecomputeVisionEmbeddings(imageBytes, key, imageQuality.nativeValue)
    }

    /** Lower-level overload — caller supplies an already-derived 32-byte VT key. */
    suspend fun precomputeVisionEmbeddings(
        imageBytes: ByteArray,
        vtKey: ByteArray,
        imageQuality: ImageQuality = ImageQuality.MEDIUM,
    ): Boolean = withContext(Dispatchers.IO) {
        GGUFNativeLib.nativeVlmPrecomputeVisionEmbeddings(imageBytes, vtKey, imageQuality.nativeValue)
    }

    /**
     * Pre-warm the VLM-KV cache: encode the image AND run the LLM
     * image-prefill in the background, capture the post-image LLM state.
     * The next [generateVlmFlow] call with the same [vlmKvKey] hits the
     * cache and skips BOTH the ViT pass AND the image-prefill — the
     * literal first user prompt against this image gets sub-second TTFT.
     *
     * [messagesJson] is the canonical pre-warm message list; typically
     * `[{"role":"user","content":"<__image__>\n"}]`. Anything *after* the
     * image marker is decoded but its KV is captured in the saved blob,
     * so keep the suffix minimal.
     *
     * Pass [vtKey] (e.g. from [computeVtKey]) to populate the VT cache
     * as a side-effect; pass null to skip the VT-side write.
     *
     * Suspends on [Dispatchers.IO]. The LLM image-prefill is the slow
     * part (~5-10 s on Snapdragon 7s Gen 3 for Qwen3-VL-2B; ~1-2 s for
     * LFM2-VL-450M). Fire-and-forget right after the host knows which
     * image and which system prompt + chat template will be used —
     * usually as soon as the image lands.
     */
    suspend fun precomputeVlmKvState(
        messagesJson: String,
        imageBytes: ByteArray,
        vlmKvKey: ByteArray,
        vtKey: ByteArray? = null,
        imageQuality: ImageQuality = ImageQuality.MEDIUM,
    ): Boolean = withContext(Dispatchers.IO) {
        GGUFNativeLib.nativeVlmPrecomputeKvState(
            messagesJson, imageBytes, vtKey, vlmKvKey,
            imageQuality.nativeValue, /* callback = */ null,
        )
    }

    /**
     * Streaming variant of [precomputeVlmKvState] that emits per-stage events
     * (one per image/text chunk) so the host UI can show "Encoding tile 3/5",
     * "Decoding tile 3/5 (3.2 s)", etc. Same caching contract as the suspend
     * version — only the callback surface differs.
     */
    fun precomputeVlmKvStateFlow(
        messagesJson: String,
        imageBytes: ByteArray,
        vlmKvKey: ByteArray,
        vtKey: ByteArray? = null,
        imageQuality: ImageQuality = ImageQuality.MEDIUM,
    ): Flow<VlmPrewarmEvent> = callbackFlow {
        val cb = object : com.dark.gguf_lib.models.VlmPrewarmCallback {
            override fun onStarted(totalChunks: Int) {
                trySend(VlmPrewarmEvent.Started(totalChunks))
            }
            override fun onChunkStart(index: Int, total: Int, isImage: Boolean) {
                trySend(VlmPrewarmEvent.ChunkStart(index, total, isImage))
            }
            override fun onChunkDone(index: Int, total: Int, encodeMs: Float, decodeMs: Float) {
                trySend(VlmPrewarmEvent.ChunkDone(index, total, encodeMs, decodeMs))
            }
            override fun onStateStored(blobBytes: Long, nTokens: Int) {
                trySend(VlmPrewarmEvent.StateStored(blobBytes, nTokens))
            }
            override fun onDone(totalMs: Long, cached: Boolean) {
                trySend(VlmPrewarmEvent.Done(totalMs, cached))
                close()
            }
            override fun onError(message: String) {
                trySend(VlmPrewarmEvent.Error(message))
                close()
            }
        }
        val job = launch(Dispatchers.IO) {
            GGUFNativeLib.nativeVlmPrecomputeKvState(
                messagesJson, imageBytes, vtKey, vlmKvKey,
                imageQuality.nativeValue, cb,
            )
            // If the native side returned without firing onDone (shouldn't
            // happen on success, but defensive), close the flow anyway.
            close()
        }
        awaitClose { job.cancel() }
    }

    /**
     * Stream generation with text + images. The user message content should
     * contain the marker from [getVlmDefaultMarker] at each image's position.
     *
     * @param imageData Raw file bytes (JPEG/PNG) for each image.
     * @param vtKeys Optional 32-byte SHA256 keys, parallel to [imageData]. When
     *   non-null and the VT cache is initialised, cached embeddings short-circuit
     *   the vision encoder. Use [computeVtKey] to derive a canonical key.
     * @param vlmKvKey Optional single 32-byte key for the VLM-KV cache. Stronger
     *   than [vtKeys] — on hit, the LLM context state captured at the post-image
     *   boundary is restored, skipping BOTH the ViT pass AND the ~9s LLM
     *   image-prefill. TTFT drops from ~10s to a few hundred ms. The key must
     *   cover everything that goes into the cached prefix: see [computeVlmKvKey].
     */
    fun generateVlmFlow(
        messagesJson: String,
        imageData: List<ByteArray>,
        maxTokens: Int = 4096,
        vtKeys: List<ByteArray?>? = null,
        vlmKvKey: ByteArray? = null,
        imageQuality: ImageQuality = ImageQuality.MEDIUM,
    ): Flow<GenerationEvent> = callbackFlow {
        val cb = streamCallback(::trySend, ::close)
        val keysArray: Array<ByteArray>? = vtKeys?.let { keys ->
            // Replace null entries with zero-length arrays so JNI sees a stable
            // jobjectArray; native side checks length == 32 before using a slot.
            Array(keys.size) { i -> keys[i] ?: ByteArray(0) }
        }
        val job = launch(Dispatchers.IO) {
            GGUFNativeLib.nativeVlmGenerateStream(
                messagesJson, imageData.toTypedArray(), keysArray, vlmKvKey,
                imageQuality.nativeValue, maxTokens, cb,
            )
        }
        awaitClose {
            job.cancel()
            GGUFNativeLib.nativeStopGeneration()
        }
    }

    /**
     * Canonical VT-cache key for an image.
     *
     * SHA256 of: image bytes ∥ projector identity ∥ image_max_tokens.
     * Two different JPEG/PNG encodings of the same picture intentionally hit
     * different cache slots — caching is byte-content-addressed for simplicity.
     * If you want pixel-content addressing, decode + re-encode at a canonical
     * resolution before calling this.
     */
    fun computeVtKey(
        imageBytes: ByteArray,
        projectorPath: String,
        imageMaxTokens: Int,
    ): ByteArray {
        val md = java.security.MessageDigest.getInstance("SHA-256")
        md.update(imageBytes)
        md.update(projectorPath.toByteArray(Charsets.UTF_8))
        md.update(byteArrayOf(
            (imageMaxTokens shr 24).toByte(),
            (imageMaxTokens shr 16).toByte(),
            (imageMaxTokens shr 8).toByte(),
            imageMaxTokens.toByte(),
        ))
        return md.digest()
    }

    // ── VT cache management ──────────────────────────────────────────────

    /** Initialise the VT cache. Call once after [load] (or any time before generate). */
    fun vtCacheInit(dir: String, budgetBytes: Long = 200L * 1024L * 1024L): Boolean =
        GGUFNativeLib.nativeVtCacheInit(dir, budgetBytes)

    fun vtCacheRelease()                  = GGUFNativeLib.nativeVtCacheRelease()
    fun vtCacheClear()                    = GGUFNativeLib.nativeVtCacheClear()
    fun vtCacheSetBudget(bytes: Long)     = GGUFNativeLib.nativeVtCacheSetBudget(bytes)
    fun vtCacheStatsJson(): String        = GGUFNativeLib.nativeVtCacheStatsJson()
    fun vtCacheListEntriesJson(): String  = GGUFNativeLib.nativeVtCacheListEntriesJson()
    fun vtCacheRemove(hash: ByteArray): Boolean = GGUFNativeLib.nativeVtCacheRemove(hash)

    // ── VLM-KV cache management ──────────────────────────────────────────

    /**
     * Initialise the VLM-KV cache. Default budget 300 MB (entries are typically
     * 5–15 MB each). Open after [load] + [loadVlmProjector], close before unload.
     */
    fun vlmKvCacheInit(dir: String, budgetBytes: Long = 300L * 1024L * 1024L): Boolean =
        GGUFNativeLib.nativeVlmKvCacheInit(dir, budgetBytes)

    fun vlmKvCacheRelease()                      = GGUFNativeLib.nativeVlmKvCacheRelease()
    fun vlmKvCacheClear()                        = GGUFNativeLib.nativeVlmKvCacheClear()
    fun vlmKvCacheSetBudget(bytes: Long)         = GGUFNativeLib.nativeVlmKvCacheSetBudget(bytes)
    fun vlmKvCacheStatsJson(): String            = GGUFNativeLib.nativeVlmKvCacheStatsJson()
    fun vlmKvCacheListEntriesJson(): String      = GGUFNativeLib.nativeVlmKvCacheListEntriesJson()
    fun vlmKvCacheRemove(hash: ByteArray): Boolean = GGUFNativeLib.nativeVlmKvCacheRemove(hash)

    /**
     * JSON snapshot of registered ggml backends + devices. Diagnostic only —
     * the engine does not currently route ops to GPU; per-op routing is parked
     * pending upstream llama.cpp changes. See VLM.md "Per-op routing" for the
     * design notes.
     */
    fun listBackendsJson(): String = GGUFNativeLib.nativeListBackendsJson()

    /**
     * Canonical VLM-KV cache key.
     *
     * SHA256 of: image bytes ∥ projector path ∥ image_max_tokens ∥
     * model fingerprint ∥ system prompt ∥ chat-template prefix.
     *
     * The cached LLM state is bound to *all* of these inputs — change any and
     * the cached prefix is no longer valid. The host app is responsible for
     * passing a stable [chatTemplatePrefix] (typically the verbatim text the
     * user puts before the image marker, e.g. an empty string if the marker
     * is the first thing in the user message).
     */
    fun computeVlmKvKey(
        imageBytes: ByteArray,
        projectorPath: String,
        imageMaxTokens: Int,
        modelFingerprint: String,
        systemPrompt: String,
        chatTemplatePrefix: String,
    ): ByteArray {
        val md = java.security.MessageDigest.getInstance("SHA-256")
        md.update(imageBytes)
        md.update(byteArrayOf(0))                                 // separator
        md.update(projectorPath.toByteArray(Charsets.UTF_8))
        md.update(byteArrayOf(0))
        md.update(byteArrayOf(
            (imageMaxTokens shr 24).toByte(),
            (imageMaxTokens shr 16).toByte(),
            (imageMaxTokens shr 8).toByte(),
            imageMaxTokens.toByte(),
        ))
        md.update(byteArrayOf(0))
        md.update(modelFingerprint.toByteArray(Charsets.UTF_8))
        md.update(byteArrayOf(0))
        md.update(systemPrompt.toByteArray(Charsets.UTF_8))
        md.update(byteArrayOf(0))
        md.update(chatTemplatePrefix.toByteArray(Charsets.UTF_8))
        return md.digest()
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

private inline fun streamCallback(
    crossinline send: (GenerationEvent) -> Unit,
    crossinline close: () -> Unit,
): StreamCallback = object : StreamCallback {
    override fun onToken(token: String) { send(GenerationEvent.Token(token)) }
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
    override fun onVlmCacheStatus(hit: Boolean, nTokens: Int, nEmbd: Int) {
        send(GenerationEvent.VtCacheStatus(hit, nTokens, nEmbd))
    }
    override fun onVlmKvCacheStatus(hit: Boolean, nTokens: Int) {
        send(GenerationEvent.VlmKvCacheStatus(hit, nTokens))
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
