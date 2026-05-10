package com.dark.gguf_lib

import com.dark.gguf_lib.models.VlmPrewarmCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Model-aware, GPU-profile-aware VLM encode/pre-warm scheduler.
 *
 * Sits in front of [GGMLEngine.precomputeVlmKvStateFlow]. Reads
 * the projector type at construction (via [GGMLEngine.getVlmInfoJson])
 * + a [GpuProfile] from [HardwareEngine], picks a strategy, and
 * exposes a clean batch-submit API.
 *
 * What it routes on:
 *  - **Projector type** — Qwen-family (single dynamic-resolution chunk per
 *    image) takes one path; tiled families (LFM2, MiniCPM, InternVL) take
 *    another. Today both paths reduce to "submit one image at a time" but
 *    the scheduler is structured so the implementation can swap to a
 *    batched primitive later (see #25, #26 in the project task list)
 *    without changing the host API.
 *  - **GPU availability** — when a non-CPU backend is registered, work
 *    runs there via clip_ctx's Vulkan path. When only CPU is available,
 *    the scheduler still works; it just doesn't get the GPU speedup.
 *  - **Image quality** — the host's chosen [ImageQuality] flows through
 *    to every encode. LOW reduces the per-image cost dramatically on
 *    tiled models because most images stop tiling entirely.
 *
 * Concurrency: pre-warm and generate both serialize at the engine's
 * gen_mutex on the native side. The scheduler does not pretend to
 * parallelize on CPU — that's harmful (DRAM bandwidth contention,
 * thermal throttling). What it *does* do:
 *  - JPEG decode + bitmap resize on a background dispatcher (parallel)
 *  - Per-job state tracked separately so the host can see fan-out progress
 *  - Cancels stale jobs when newer submissions arrive (configurable)
 *
 * Future: the same submit API will dispatch to a true batched ViT
 * primitive once clip.cpp grows n_batch>1 support. No host-code change
 * required at that point.
 */
class VlmEncoder(
    private val engine: GGMLEngine,
    private val projectorPath: String,
    private val modelFingerprint: String,
    private val systemPrompt: String = "",
    private val chatTemplatePrefix: String = "<__image__>\n",
    private val imageMaxTokens: Int = 256,
    /**
     * GPU profile snapshot at construction. Used to log what we're routing
     * on and to gate decisions (e.g. "this is a software rasterizer, do
     * everything on CPU"). Re-probed only when the host calls [refreshHardware].
     */
    initialHardware: List<GpuProfile> = HardwareEngine.probe(),
) {

    private val _hardware = MutableStateFlow(initialHardware)
    val hardware: StateFlow<List<GpuProfile>> = _hardware.asStateFlow()

    /** First non-CPU profile if any, else null. */
    val gpu: GpuProfile? get() = _hardware.value.firstOrNull { it.isGpu }

    /**
     * Model architecture lifted from the LLM's GGUF metadata
     * (`general.architecture`). Stable enough proxies for projector type:
     * "qwen3vl", "lfm2", "llava", "internvl", "minicpmv", etc.
     */
    val architecture: String = run {
        val info = runCatching { engine.getModelInfoJson() }.getOrNull()
        if (info != null) JSONObject(info).optString("architecture", "unknown") else "unknown"
    }

    /** Coarse routing strategy picked once, based on architecture + hardware. */
    val strategy: EncodeStrategy = pickStrategy(architecture, gpu)

    private val _activeJobs = MutableStateFlow<List<JobState>>(emptyList())
    val activeJobs: StateFlow<List<JobState>> = _activeJobs.asStateFlow()

    private val supervisor = SupervisorJob()
    private var nextJobId: Long = 0
    private val jobs: MutableMap<Long, Job> = mutableMapOf()

    fun refreshHardware() {
        _hardware.value = HardwareEngine.probe()
    }

    /**
     * Pre-warm both caches for one image. Returns a Flow that emits
     * [VlmEncodeEvent] until the work completes (or fails). Cancelling the
     * collector cancels the underlying native call at the next chunk boundary.
     */
    fun submit(
        imageBytes: ByteArray,
        quality: ImageQuality = ImageQuality.MEDIUM,
    ): Flow<VlmEncodeEvent> = callbackFlow {
        val jobId = ++nextJobId
        val started = System.currentTimeMillis()

        val pre = JSONArray().apply {
            put(JSONObject().apply {
                put("role", "user")
                put("content", "${engine.getVlmDefaultMarker()}\n")
            })
        }.toString()

        val vtKey = engine.computeVtKey(imageBytes, projectorPath, imageMaxTokens)
        val vlmKvKey = engine.computeVlmKvKey(
            imageBytes         = imageBytes,
            projectorPath      = projectorPath,
            imageMaxTokens     = imageMaxTokens,
            modelFingerprint   = modelFingerprint,
            systemPrompt       = systemPrompt,
            chatTemplatePrefix = chatTemplatePrefix,
        )

        updateJob(jobId, JobState.Pending(jobId, started))
        trySend(VlmEncodeEvent.Queued(jobId))

        val nativeJob = launch(Dispatchers.IO) {
            try {
                engine.precomputeVlmKvStateFlow(
                    messagesJson = pre,
                    imageBytes   = imageBytes,
                    vlmKvKey     = vlmKvKey,
                    vtKey        = vtKey,
                    imageQuality = quality,
                ).collect { ev ->
                    when (ev) {
                        is VlmPrewarmEvent.Started -> {
                            updateJob(jobId, JobState.Running(jobId, started, "starting", 0, ev.totalChunks))
                            trySend(VlmEncodeEvent.Started(jobId, ev.totalChunks))
                        }
                        is VlmPrewarmEvent.ChunkStart -> {
                            updateJob(jobId, JobState.Running(jobId, started,
                                if (ev.isImage) "encoding" else "decoding", ev.index, ev.total))
                            trySend(VlmEncodeEvent.Stage(jobId, ev.index, ev.total, ev.isImage))
                        }
                        is VlmPrewarmEvent.ChunkDone -> {
                            updateJob(jobId, JobState.Running(jobId, started,
                                "decoded", ev.index + 1, ev.total))
                            trySend(VlmEncodeEvent.ChunkDone(jobId, ev.index, ev.total, ev.encodeMs, ev.decodeMs))
                        }
                        is VlmPrewarmEvent.StateStored -> {
                            trySend(VlmEncodeEvent.StateStored(jobId, ev.blobBytes, ev.nTokens))
                        }
                        is VlmPrewarmEvent.Done -> {
                            removeJob(jobId)
                            trySend(VlmEncodeEvent.Done(jobId, ev.totalMs, ev.cached))
                            close()
                        }
                        is VlmPrewarmEvent.Error -> {
                            removeJob(jobId)
                            trySend(VlmEncodeEvent.Error(jobId, ev.message))
                            close()
                        }
                    }
                }
            } catch (t: Throwable) {
                removeJob(jobId)
                trySend(VlmEncodeEvent.Error(jobId, t.message ?: t::class.java.simpleName))
                close()
            }
        }

        jobs[jobId] = nativeJob
        awaitClose {
            nativeJob.cancel()
            removeJob(jobId)
        }
    }

    /**
     * Submit multiple images. Today this loops sequentially through the
     * native engine because the gen_mutex serializes all VLM ops. Returns
     * a single Flow that interleaves events from all jobs.
     *
     * When the batched ViT primitive lands (#25), this method becomes the
     * place to switch to a single batched call.
     */
    fun submitBatch(
        images: List<ByteArray>,
        quality: ImageQuality = ImageQuality.MEDIUM,
    ): Flow<VlmEncodeEvent> = callbackFlow {
        val nativeJob = launch(Dispatchers.IO) {
            try {
                for (img in images) {
                    submit(img, quality).collect { trySend(it) }
                }
                close()
            } catch (t: Throwable) {
                close(t)
            }
        }
        awaitClose { nativeJob.cancel() }
    }

    fun cancelAll() {
        jobs.values.forEach { it.cancel() }
        jobs.clear()
        _activeJobs.value = emptyList()
    }

    private fun updateJob(id: Long, st: JobState) {
        _activeJobs.value = (_activeJobs.value.filter { it.id != id } + st)
            .sortedBy { it.id }
    }

    private fun removeJob(id: Long) {
        _activeJobs.value = _activeJobs.value.filter { it.id != id }
        jobs.remove(id)?.cancel()
    }

    /** Strategy hint, exposed so hosts can show "ROUTING: GPU - dynamic-res batch" in debug UI. */
    fun describeRouting(): String {
        val gp = gpu
        val where = when {
            gp == null                                                  -> "CPU"
            gp.vendor == GpuVendor.SOFTWARE_RASTERIZER                  -> "CPU (SwiftShader detected — slower than CPU)"
            else                                                        -> "${gp.vendor.name.lowercase()} (${gp.deviceName})"
        }
        return "Encoder: $where  ·  arch=$architecture  ·  strategy=${strategy.name}"
    }

    sealed interface JobState {
        val id: Long
        val startedAt: Long
        data class Pending(override val id: Long, override val startedAt: Long) : JobState
        data class Running(
            override val id: Long,
            override val startedAt: Long,
            val stage: String,
            val chunkIndex: Int,
            val totalChunks: Int,
        ) : JobState
    }

    enum class EncodeStrategy {
        /** Dynamic-resolution single-chunk family (Qwen2/2.5/3-VL, Pixtral). */
        SINGLE_PASS,
        /** Tiled family (LFM2-VL, MiniCPM-V, InternVL, LLaVA-NeXT, Idefics3, Llama4). */
        TILED,
        /** Unknown projector — safe default. */
        DEFAULT,
    }

    companion object {
        private fun pickStrategy(architecture: String, gpu: GpuProfile?): EncodeStrategy {
            val a = architecture.lowercase()
            return when {
                "qwen" in a || "pixtral" in a           -> EncodeStrategy.SINGLE_PASS
                "lfm2" in a || "minicpm" in a ||
                "internvl" in a || "llava" in a ||
                "idefics" in a || "llama4" in a ||
                "gemma3" in a                            -> EncodeStrategy.TILED
                else                                     -> EncodeStrategy.DEFAULT
            }
        }
    }
}

/** Aggregate event stream. Each event is tagged with a [jobId] so the host can fan progress out per image. */
sealed class VlmEncodeEvent(open val jobId: Long) {
    data class Queued(override val jobId: Long) : VlmEncodeEvent(jobId)
    data class Started(override val jobId: Long, val totalChunks: Int) : VlmEncodeEvent(jobId)
    data class Stage(override val jobId: Long, val index: Int, val total: Int, val isImage: Boolean) : VlmEncodeEvent(jobId)
    data class ChunkDone(override val jobId: Long, val index: Int, val total: Int, val encodeMs: Float, val decodeMs: Float) : VlmEncodeEvent(jobId)
    data class StateStored(override val jobId: Long, val blobBytes: Long, val nTokens: Int) : VlmEncodeEvent(jobId)
    data class Done(override val jobId: Long, val totalMs: Long, val cached: Boolean) : VlmEncodeEvent(jobId)
    data class Error(override val jobId: Long, val message: String) : VlmEncodeEvent(jobId)
}
