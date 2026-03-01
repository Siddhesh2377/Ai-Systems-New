package com.dark.gguf_lib

import android.content.Context
import android.net.Uri
import android.os.ParcelFileDescriptor
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

/**
 * GGMLEngine - Main interface for LLM inference on Android.
 *
 * Supports model loading via file path or Android SAF URI,
 * text generation with streaming, and full model introspection.
 */
class GGMLEngine(
    private val contextSize: Int = 2048,
    private val batchSize: Int = 512,
    private val threads: Int = 0,  // 0 = auto-detect
    private val useMmap: Boolean = true,
    private val flashAttn: Boolean = true,
) : AutoCloseable {

    private var nativeHandle: Long = 0L

    init {
        nativeHandle = nativeCreate(contextSize, batchSize, threads, useMmap, flashAttn)
    }

    val isLoaded: Boolean get() = nativeHandle != 0L && nativeIsLoaded(nativeHandle)

    val contextUsed: Int get() = if (nativeHandle != 0L) nativeContextUsed(nativeHandle) else 0

    val totalContextSize: Int get() = if (nativeHandle != 0L) nativeContextSize(nativeHandle) else 0

    // ---- Model Loading ----

    fun loadModel(path: String): EngineStatus {
        check(nativeHandle != 0L) { "Engine already destroyed" }
        return EngineStatus.fromCode(nativeLoadModel(nativeHandle, path))
    }

    fun loadModel(context: Context, uri: Uri): EngineStatus {
        check(nativeHandle != 0L) { "Engine already destroyed" }
        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            ?: return EngineStatus.LOAD_FAILED
        return try {
            EngineStatus.fromCode(nativeLoadModelFromFd(nativeHandle, pfd.fd))
        } finally {
            pfd.close()
        }
    }

    fun unloadModel() {
        if (nativeHandle != 0L) nativeUnloadModel(nativeHandle)
    }

    // ---- Model Info ----

    fun getModelInfoJson(): String {
        check(nativeHandle != 0L && isLoaded) { "No model loaded" }
        return nativeGetModelInfo(nativeHandle)
    }

    // ---- Generation ----

    suspend fun generate(
        prompt: String,
        params: SamplingParams = SamplingParams(),
    ): GenerationResult = withContext(Dispatchers.IO) {
        check(nativeHandle != 0L && isLoaded) { "No model loaded" }

        val status = nativeGenerate(
            nativeHandle, prompt,
            params.temperature, params.topK, params.topP,
            params.minP, params.repeatPenalty, params.repeatLastN,
            params.maxTokens, params.seed, null
        )

        val response = nativeGetResponse(nativeHandle)
        val perf = nativeGetPerf(nativeHandle)

        GenerationResult(
            status = EngineStatus.fromCode(status),
            text = response,
            perf = PerfMetrics(
                promptEvalMs = perf[0].toDouble(),
                generationMs = perf[1].toDouble(),
                promptTokens = perf[2].toInt(),
                generatedTokens = perf[3].toInt(),
                promptTokensPerSec = perf[4].toDouble(),
                generationTokensPerSec = perf[5].toDouble(),
            )
        )
    }

    fun generateStream(
        prompt: String,
        params: SamplingParams = SamplingParams(),
    ): Flow<String> = callbackFlow {
        val job = launch(Dispatchers.IO) {
            val callback = object : TokenCallback {
                override fun onToken(token: String): Boolean {
                    val result = trySend(token)
                    return result.isSuccess
                }
            }

            nativeGenerate(
                nativeHandle, prompt,
                params.temperature, params.topK, params.topP,
                params.minP, params.repeatPenalty, params.repeatLastN,
                params.maxTokens, params.seed, callback
            )

            channel.close()
        }

        awaitClose { job.cancel() }
    }

    fun cancel() {
        if (nativeHandle != 0L) nativeCancel(nativeHandle)
    }

    fun clearContext() {
        if (nativeHandle != 0L) nativeClearContext(nativeHandle)
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    // ---- Native methods ----

    private external fun nativeCreate(
        nCtx: Int, nBatch: Int, nThreads: Int, useMmap: Boolean, flashAttn: Boolean
    ): Long

    private external fun nativeDestroy(handle: Long)
    private external fun nativeLoadModel(handle: Long, path: String): Int
    private external fun nativeLoadModelFromFd(handle: Long, fd: Int): Int
    private external fun nativeUnloadModel(handle: Long)
    private external fun nativeIsLoaded(handle: Long): Boolean
    private external fun nativeGetModelInfo(handle: Long): String
    private external fun nativeGenerate(
        handle: Long, prompt: String,
        temperature: Float, topK: Int, topP: Float,
        minP: Float, repeatPenalty: Float, repeatLastN: Int,
        nPredict: Int, seed: Int, callback: TokenCallback?
    ): Int
    private external fun nativeCancel(handle: Long)
    private external fun nativeGetResponse(handle: Long): String
    private external fun nativeClearContext(handle: Long)
    private external fun nativeContextUsed(handle: Long): Int
    private external fun nativeContextSize(handle: Long): Int
    private external fun nativeGetPerf(handle: Long): FloatArray

    companion object {
        init {
            System.loadLibrary("gguf_lib")
        }
    }
}

// ---- Data classes ----

enum class EngineStatus(val code: Int) {
    OK(0),
    LOAD_FAILED(1),
    CONTEXT_FAIL(2),
    NO_MODEL(3),
    TOKENIZE_ERROR(4),
    DECODE_ERROR(5),
    CANCELLED(6),
    OUT_OF_MEMORY(7);

    companion object {
        fun fromCode(code: Int): EngineStatus =
            entries.find { it.code == code } ?: LOAD_FAILED
    }
}

data class SamplingParams(
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    val maxTokens: Int = 256,
    val seed: Int = -1,  // -1 = random
)

data class PerfMetrics(
    val promptEvalMs: Double,
    val generationMs: Double,
    val promptTokens: Int,
    val generatedTokens: Int,
    val promptTokensPerSec: Double,
    val generationTokensPerSec: Double,
)

data class GenerationResult(
    val status: EngineStatus,
    val text: String,
    val perf: PerfMetrics,
)

interface TokenCallback {
    fun onToken(token: String): Boolean
}
