package com.dark.demon_system.ui.vlm

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dark.demon_system.data.VlmModelDownloader
import com.dark.demon_system.data.VlmModelSpec
import com.dark.gguf_lib.GGMLEngine
import com.dark.gguf_lib.models.GenerationEvent
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * VLM screen ViewModel — owns the downloader + [GGMLEngine] lifecycle.
 *
 * State flow: Idle → Downloading → LoadingModel → LoadingProjector → Ready
 *           → Generating → GenerationDone → Ready (loop)
 */
class VlmViewModel(app: Application) : AndroidViewModel(app) {

    val spec: VlmModelSpec = VlmModelSpec.LFM2_VL_450M
    private val downloader = VlmModelDownloader(app)
    private val engine = GGMLEngine()

    /** Set after [loadAndPrepare] succeeds; used to derive VT cache keys. */
    private var projectorPath: String = ""
    private val imageMaxTokens: Int = 256
    private var vtCacheReady: Boolean = false
    private var vlmKvCacheReady: Boolean = false
    private var modelFingerprint: String = ""

    /**
     * The chat-template prefix that sits between the system prompt and the
     * first character of the user's question. The VLM-KV cache key includes
     * this — change it and every cached entry is invalidated.
     *
     * In this test app, the user message is always "<__image__>\n{question}",
     * so the prefix between system+template and the question is just the
     * marker plus a newline. The host can extend this with sticky preamble.
     */
    private val chatTemplatePrefix: String = "<__image__>\n"

    /** UI-controllable image quality. Affects ViT compute, prefill batch size, and cache entry size. */
    private val _imageQuality = MutableStateFlow(com.dark.gguf_lib.ImageQuality.MEDIUM)
    val imageQuality: StateFlow<com.dark.gguf_lib.ImageQuality> = _imageQuality.asStateFlow()
    fun setImageQuality(q: com.dark.gguf_lib.ImageQuality) { _imageQuality.value = q }

    private val _state = MutableStateFlow<VlmState>(VlmState.Idle)
    val state: StateFlow<VlmState> = _state.asStateFlow()

    private val _prewarmState = MutableStateFlow<PrewarmState>(PrewarmState.Idle)
    val prewarmState: StateFlow<PrewarmState> = _prewarmState.asStateFlow()

    private var generationJob: Job? = null
    private var prewarmJob: Job? = null

    val modelDownloaded: Boolean get() = downloader.isFullyDownloaded(spec)
    val isLoaded: Boolean get() = engine.isLoaded
    val isVlmLoaded: Boolean get() = engine.isVlmLoaded

    fun startDownload() {
        if (modelDownloaded) {
            _state.value = VlmState.Idle
            return
        }
        viewModelScope.launch {
            downloader.download(spec).collectLatest { event ->
                when (event) {
                    is VlmModelDownloader.Event.FileProgress -> {
                        _state.value = VlmState.Downloading(
                            fileIndex = event.fileIndex,
                            fileName = event.fileName,
                            bytesDownloaded = event.bytesDownloaded,
                            totalBytes = event.totalBytes,
                            bytesPerSecond = event.bytesPerSecond,
                            overallPct = event.overallPct,
                        )
                    }
                    is VlmModelDownloader.Event.Done -> {
                        _state.value = VlmState.Idle
                    }
                    is VlmModelDownloader.Event.Failed -> {
                        _state.value = VlmState.DownloadFailed(event.message)
                    }
                }
            }
        }
    }

    fun loadAndPrepare() {
        if (engine.isLoaded && engine.isVlmLoaded) {
            _state.value = VlmState.Ready
            return
        }
        viewModelScope.launch {
            try {
                _state.value = VlmState.LoadingModel
                val textPath = downloader.textFile(spec).absolutePath
                val projPath = downloader.projFile(spec).absolutePath

                val ok = engine.load(
                    path = textPath,
                    contextSize = 4096,
                    threads = 0,
                    batchSize = 0,
                    flashAttn = true,
                    useMmap = true,
                    useMlock = false,
                    cacheTypeK = "q8_0",
                    cacheTypeV = "q8_0",
                    // opOffload=true gives per-op CPU/GPU routing (large ops
                    // → Vulkan, decode → CPU) but currently triggers
                    // vk::DeviceLostError on Adreno 810's driver during the
                    // 234-token image-prefill compute graph (kernel TDR).
                    // Disabled until we add per-device gating or chunk the
                    // image-prefill into smaller Vulkan dispatches.
                    opOffload = false,
                )
                if (!ok) {
                    _state.value = VlmState.Error("Failed to load text model")
                    return@launch
                }

                _state.value = VlmState.LoadingProjector
                val vlmOk = engine.loadVlmProjector(
                    path = projPath,
                    threads = 0,
                    imageMinTokens = -1,
                    imageMaxTokens = imageMaxTokens,
                )
                if (!vlmOk) {
                    _state.value = VlmState.Error("Failed to load VLM projector")
                    return@launch
                }
                projectorPath = projPath
                modelFingerprint = "${spec.repo}:${spec.textFilename}"

                // Open the VT cache. Survives app restarts; LRU-evicts at 200 MB.
                val cacheDir = java.io.File(getApplication<Application>().filesDir, "vt_cache").absolutePath
                vtCacheReady = engine.vtCacheInit(cacheDir, 200L * 1024L * 1024L)

                // Open the VLM-KV cache. Bigger entries (~5–15 MB each — full LLM
                // KV slice for the post-image prefix), so 300 MB budget by default.
                val kvCacheDir = java.io.File(
                    getApplication<Application>().filesDir, "vlm_kv_cache",
                ).absolutePath
                vlmKvCacheReady = engine.vlmKvCacheInit(kvCacheDir, 300L * 1024L * 1024L)

                _state.value = VlmState.Ready
            } catch (t: Throwable) {
                _state.value = VlmState.Error(t.message ?: t::class.java.simpleName)
            }
        }
    }

    fun generate(prompt: String, imageBytes: ByteArray) {
        if (!engine.isLoaded || !engine.isVlmLoaded) {
            _state.value = VlmState.Error("Model not loaded")
            return
        }
        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            val marker = engine.getVlmDefaultMarker()
            val messagesJson = JSONArray().apply {
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", "$marker\n${prompt.trim()}")
                })
            }.toString()

            val text = StringBuilder()
            var encodeMs: Float? = null
            var decodeMs: Float? = null
            var imgTokens: Int? = null
            var cacheHit:  Boolean? = null
            var vlmKvHit:  Boolean? = null

            // Derive cache keys. VT covers just the ViT pass; VLM-KV covers the
            // entire pre-question state and is the bigger TTFT win on hits.
            val vtKey: ByteArray? = if (vtCacheReady) {
                engine.computeVtKey(imageBytes, projectorPath, imageMaxTokens)
            } else null

            val vlmKvKey: ByteArray? = if (vlmKvCacheReady) {
                engine.computeVlmKvKey(
                    imageBytes         = imageBytes,
                    projectorPath      = projectorPath,
                    imageMaxTokens     = imageMaxTokens,
                    modelFingerprint   = modelFingerprint,
                    systemPrompt       = "",
                    chatTemplatePrefix = chatTemplatePrefix,
                )
            } else null

            _state.value = VlmState.Generating(text = "")

            try {
                engine.generateVlmFlow(
                    messagesJson = messagesJson,
                    imageData = listOf(imageBytes),
                    maxTokens = 512,
                    vtKeys = vtKey?.let { listOf(it) },
                    vlmKvKey = vlmKvKey,
                    imageQuality = _imageQuality.value,
                ).collect { event ->
                    when (event) {
                        is GenerationEvent.Token -> {
                            text.append(event.text)
                            _state.value = VlmState.Generating(
                                text = text.toString(),
                                vlmEncodeMs = encodeMs,
                                vlmDecodeMs = decodeMs,
                                imageTokens = imgTokens,
                            )
                        }
                        is GenerationEvent.Progress -> {
                            val cur = _state.value
                            if (cur is VlmState.Generating) {
                                _state.value = cur.copy(progress = event.progress)
                            }
                        }
                        is GenerationEvent.VlmStageMetrics -> {
                            encodeMs = event.vlmEncodeMs
                            decodeMs = event.vlmDecodeMs
                            imgTokens = event.imageTokens
                            val cur = _state.value
                            if (cur is VlmState.Generating) {
                                _state.value = cur.copy(
                                    vlmEncodeMs = encodeMs,
                                    vlmDecodeMs = decodeMs,
                                    imageTokens = imgTokens,
                                )
                            }
                        }
                        is GenerationEvent.VtCacheStatus -> {
                            cacheHit = event.hit
                            val cur = _state.value
                            if (cur is VlmState.Generating) {
                                _state.value = cur.copy(vtCacheHit = event.hit)
                            }
                        }
                        is GenerationEvent.VlmKvCacheStatus -> {
                            vlmKvHit = event.hit
                            val cur = _state.value
                            if (cur is VlmState.Generating) {
                                _state.value = cur.copy(vlmKvCacheHit = event.hit)
                            }
                        }
                        is GenerationEvent.Metrics -> {
                            _state.value = VlmState.GenerationDone(
                                text = text.toString(),
                                metrics = event.metrics,
                                vlmEncodeMs = encodeMs,
                                vlmDecodeMs = decodeMs,
                                imageTokens = imgTokens,
                                vtCacheHit = cacheHit,
                                vlmKvCacheHit = vlmKvHit,
                            )
                        }
                        is GenerationEvent.Done -> {
                            val cur = _state.value
                            if (cur !is VlmState.GenerationDone) {
                                _state.value = VlmState.GenerationDone(
                                    text = text.toString(),
                                    metrics = null,
                                    vlmEncodeMs = encodeMs,
                                    vlmDecodeMs = decodeMs,
                                    imageTokens = imgTokens,
                                    vtCacheHit = cacheHit,
                                    vlmKvCacheHit = vlmKvHit,
                                )
                            }
                        }
                        is GenerationEvent.Error -> {
                            _state.value = VlmState.Error(event.message)
                        }
                    }
                }
            } catch (t: Throwable) {
                _state.value = VlmState.Error(t.message ?: t::class.java.simpleName)
            }
        }
    }

    fun stopGeneration() {
        engine.stopGeneration()
        generationJob?.cancel()
    }

    /**
     * Fire-and-forget background pre-warm for [imageBytes]. Runs BOTH the
     * vision encoder AND the LLM image-prefill, populating both caches so
     * the first generate() against this image gets sub-second TTFT (no ViT,
     * no image-prefill — only the user-question text decode).
     *
     * The pre-warm uses a canonical message that matches the structure
     * generate() builds (`<__image__>\n`), so the cache key derivation
     * lines up. No-op if the engine isn't loaded or the caches aren't
     * initialised.
     */
    fun precomputeVisionFor(imageBytes: ByteArray) {
        if (!engine.isLoaded || !engine.isVlmLoaded) {
            _prewarmState.value = PrewarmState.Idle
            return
        }
        // Cancel any in-flight pre-warm — newer image takes precedence.
        prewarmJob?.cancel()
        prewarmJob = viewModelScope.launch(Dispatchers.IO) {
            val t0 = System.currentTimeMillis()
            _prewarmState.value = PrewarmState.InProgress(startedAt = t0)
            try {
                val marker = engine.getVlmDefaultMarker()
                val pre = JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "user")
                        put("content", "$marker\n")
                    })
                }.toString()

                val vtKey = if (vtCacheReady) {
                    engine.computeVtKey(imageBytes, projectorPath, imageMaxTokens)
                } else null

                if (!vlmKvCacheReady) {
                    // Fallback: ViT-only pre-warm if VLM-KV cache isn't open.
                    val ok = vtKey?.let {
                        engine.precomputeVisionEmbeddings(imageBytes, it, _imageQuality.value)
                    } ?: false
                    val dt = System.currentTimeMillis() - t0
                    _prewarmState.value = PrewarmState.Done(durationMs = dt, cached = ok)
                    return@launch
                }

                val vlmKvKey = engine.computeVlmKvKey(
                    imageBytes         = imageBytes,
                    projectorPath      = projectorPath,
                    imageMaxTokens     = imageMaxTokens,
                    modelFingerprint   = modelFingerprint,
                    systemPrompt       = "",
                    chatTemplatePrefix = chatTemplatePrefix,
                )

                var totalChunks = 0
                var blobBytes   = 0L
                var nTokens     = 0

                engine.precomputeVlmKvStateFlow(
                    messagesJson = pre,
                    imageBytes   = imageBytes,
                    vlmKvKey     = vlmKvKey,
                    vtKey        = vtKey,
                    imageQuality = _imageQuality.value,
                ).collect { ev ->
                    when (ev) {
                        is com.dark.gguf_lib.VlmPrewarmEvent.Started -> {
                            totalChunks = ev.totalChunks
                            _prewarmState.value = PrewarmState.InProgress(
                                startedAt   = t0,
                                stage       = "Starting (${ev.totalChunks} chunks)…",
                                totalChunks = ev.totalChunks,
                            )
                        }
                        is com.dark.gguf_lib.VlmPrewarmEvent.ChunkStart -> {
                            val verb = if (ev.isImage) "Encoding image" else "Decoding text"
                            _prewarmState.value = PrewarmState.InProgress(
                                startedAt    = t0,
                                stage        = "$verb chunk ${ev.index + 1}/${ev.total}…",
                                chunkIndex   = ev.index,
                                totalChunks  = ev.total,
                            )
                        }
                        is com.dark.gguf_lib.VlmPrewarmEvent.ChunkDone -> {
                            // Bump the index to "completed" so the bar advances.
                            _prewarmState.value = PrewarmState.InProgress(
                                startedAt    = t0,
                                stage        = "Decoding into LLM (chunk ${ev.index + 1}/${ev.total})…",
                                chunkIndex   = ev.index + 1,
                                totalChunks  = ev.total,
                                lastEncodeMs = ev.encodeMs.takeIf { it > 0f },
                                lastDecodeMs = ev.decodeMs.takeIf { it > 0f },
                            )
                        }
                        is com.dark.gguf_lib.VlmPrewarmEvent.StateStored -> {
                            blobBytes = ev.blobBytes
                            nTokens   = ev.nTokens
                            _prewarmState.value = PrewarmState.InProgress(
                                startedAt   = t0,
                                stage       = "Saving KV state (${ev.blobBytes / 1024} KB)…",
                                chunkIndex  = totalChunks,
                                totalChunks = totalChunks,
                            )
                        }
                        is com.dark.gguf_lib.VlmPrewarmEvent.Done -> {
                            _prewarmState.value = PrewarmState.Done(
                                durationMs  = ev.totalMs,
                                cached      = ev.cached,
                                totalChunks = totalChunks,
                                blobBytes   = blobBytes,
                                nTokens     = nTokens,
                            )
                        }
                        is com.dark.gguf_lib.VlmPrewarmEvent.Error -> {
                            _prewarmState.value = PrewarmState.Failed(ev.message)
                        }
                    }
                }
            } catch (t: Throwable) {
                _prewarmState.value = PrewarmState.Failed(t.message ?: t::class.java.simpleName)
            }
        }
    }

    /** Reset prewarm state — call from the UI when a fresh image is being picked. */
    fun resetPrewarm() { _prewarmState.value = PrewarmState.Idle }

    /** Drop every cached VT entry from disk. */
    fun clearVtCache() {
        if (vtCacheReady)    engine.vtCacheClear()
        if (vlmKvCacheReady) engine.vlmKvCacheClear()
    }

    /** Returns `null` if cache not initialised. */
    fun vtCacheStatsJson(): String? =
        if (vtCacheReady) engine.vtCacheStatsJson() else null

    fun vlmKvCacheStatsJson(): String? =
        if (vlmKvCacheReady) engine.vlmKvCacheStatsJson() else null

    override fun onCleared() {
        super.onCleared()
        // Release on a background thread; don't block the main thread on KV teardown.
        viewModelScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.IO) {
                runCatching { engine.vlmKvCacheRelease() }
                runCatching { engine.vtCacheRelease() }
                runCatching { engine.releaseVlmProjector() }
                runCatching { engine.unload() }
            }
        }
    }
}
