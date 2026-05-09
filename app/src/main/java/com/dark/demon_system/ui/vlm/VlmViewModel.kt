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

    val spec: VlmModelSpec = VlmModelSpec.QWEN3_VL_2B
    private val downloader = VlmModelDownloader(app)
    private val engine = GGMLEngine()

    /** Set after [loadAndPrepare] succeeds; used to derive VT cache keys. */
    private var projectorPath: String = ""
    private val imageMaxTokens: Int = 256
    private var vtCacheReady: Boolean = false

    private val _state = MutableStateFlow<VlmState>(VlmState.Idle)
    val state: StateFlow<VlmState> = _state.asStateFlow()

    private var generationJob: Job? = null

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

                // Open the VT cache. Survives app restarts; LRU-evicts at 200 MB.
                val cacheDir = java.io.File(getApplication<Application>().filesDir, "vt_cache").absolutePath
                vtCacheReady = engine.vtCacheInit(cacheDir, 200L * 1024L * 1024L)

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

            // Derive a canonical VT cache key for this image.
            val vtKey: ByteArray? = if (vtCacheReady) {
                engine.computeVtKey(imageBytes, projectorPath, imageMaxTokens)
            } else null

            _state.value = VlmState.Generating(text = "")

            try {
                engine.generateVlmFlow(
                    messagesJson = messagesJson,
                    imageData = listOf(imageBytes),
                    maxTokens = 512,
                    vtKeys = vtKey?.let { listOf(it) },
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
                        is GenerationEvent.Metrics -> {
                            _state.value = VlmState.GenerationDone(
                                text = text.toString(),
                                metrics = event.metrics,
                                vlmEncodeMs = encodeMs,
                                vlmDecodeMs = decodeMs,
                                imageTokens = imgTokens,
                                vtCacheHit = cacheHit,
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

    /** Drop every cached VT entry from disk. */
    fun clearVtCache() {
        if (vtCacheReady) engine.vtCacheClear()
    }

    /** Returns `null` if cache not initialised. */
    fun vtCacheStatsJson(): String? =
        if (vtCacheReady) engine.vtCacheStatsJson() else null

    override fun onCleared() {
        super.onCleared()
        // Release on a background thread; don't block the main thread on KV teardown.
        viewModelScope.launch(Dispatchers.IO) {
            withContext(Dispatchers.IO) {
                runCatching { engine.vtCacheRelease() }
                runCatching { engine.releaseVlmProjector() }
                runCatching { engine.unload() }
            }
        }
    }
}
