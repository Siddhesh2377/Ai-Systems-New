package com.dark.ai_rmg

import com.dark.ai_rmg.models.DecodingMetrics
import com.dark.ai_rmg.models.GenerationEvent
import com.dark.ai_rmg.models.GenerationResult
import com.dark.ai_rmg.models.RmgDims
import com.dark.ai_rmg.models.RmgLogLevel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-level wrapper around a single rm-graph engine instance.
 *
 * Not thread-safe: callers must serialize forward / generate / reset on a
 * given instance. close() is idempotent; do not call it while another thread
 * is mid-call on the same engine.
 */
class RmgEngine : Closeable {

    @Volatile private var handle: Long = 0L
    private var _dims: RmgDims? = null
    private var _hasTokenizer: Boolean = false

    val isLoaded: Boolean get() = handle != 0L

    val dims: RmgDims? get() = _dims

    /** True when the loaded model embeds a tokenizer blob. */
    val hasTokenizer: Boolean get() = _hasTokenizer

    val seqPos: Int
        get() = if (handle != 0L) RmgNativeLib.nativeSeqPos(handle) else 0

    fun load(path: String) {
        check(handle == 0L) { "engine already loaded — call unload() first" }
        val h = RmgNativeLib.nativeOpen(path)
        if (h == 0L) error("failed to open rmg model: $path")
        val raw = RmgNativeLib.nativeGetDims(h) ?: run {
            RmgNativeLib.nativeClose(h)
            error("failed to read model dims")
        }
        _dims = RmgDims(
            dModel = raw[0],
            nLayers = raw[1],
            nHeads = raw[2],
            nKvHeads = raw[3],
            dHead = raw[4],
            dFf = raw[5],
            vocabSize = raw[6],
            maxSeq = raw[7],
            ropeTheta = Float.fromBits(raw[8]),
            rmsEps = Float.fromBits(raw[9]),
            ropeInterleaved = raw[10] != 0,
            tieWordEmbeddings = raw[11] != 0
        )
        _hasTokenizer = RmgNativeLib.nativeHasTokenizer(h)
        handle = h
    }

    fun unload() = close()

    fun reset() {
        RmgNativeLib.nativeReset(requireOpen())
    }

    /**
     * Single-token forward pass. Advances seqPos by 1 and writes
     * vocabSize floats into [logitsOut]. Caller samples and calls again.
     */
    fun forward(tokenId: Int, logitsOut: FloatArray) {
        val h = requireOpen()
        val d = _dims!!
        require(logitsOut.size >= d.vocabSize) {
            "logitsOut.size=${logitsOut.size} < vocabSize=${d.vocabSize}"
        }
        check(RmgNativeLib.nativeForward(h, tokenId, logitsOut)) {
            "engine_forward failed (tokenId=$tokenId)"
        }
    }

    /**
     * Native one-shot greedy decode. Resets the KV cache, prefills the prompt,
     * then samples up to [maxNew] tokens. Returns when the full result is
     * ready — no per-token callback. Use [generateFlow] for streaming.
     *
     * [stopId] = -1 disables early stop. Stop token is not included in result.
     * Sets [GenerationResult.text] to the decoded UTF-8 string when the model
     * has an embedded tokenizer; null otherwise.
     */
    suspend fun generate(
        promptIds: IntArray,
        maxNew: Int = 64,
        stopId: Int = -1
    ): GenerationResult = withContext(Dispatchers.IO) {
        val h = requireOpen()
        require(promptIds.isNotEmpty()) { "promptIds must not be empty" }
        require(maxNew >= 0) { "maxNew must be >= 0" }

        val t0 = System.nanoTime()
        val out = RmgNativeLib.nativeGenerate(h, promptIds, maxNew, stopId)
            ?: return@withContext GenerationResult(
                tokenIds = IntArray(0),
                text = null,
                success = false,
                error = "engine_generate returned null"
            )
        val totalMs = (System.nanoTime() - t0) / 1_000_000f
        val total = out.size + promptIds.size
        val tps = if (totalMs > 0f) total * 1000f / totalMs else 0f

        val text = if (_hasTokenizer && out.isNotEmpty()) {
            RmgNativeLib.nativeDecodeTokens(h, out)?.let { String(it, Charsets.UTF_8) }
        } else null

        GenerationResult(
            tokenIds = out,
            text = text,
            success = true,
            metrics = DecodingMetrics(
                tokensPerSecond = tps,
                totalTimeMs = totalMs,
                tokensEvaluated = promptIds.size,
                tokensPredicted = out.size
            )
        )
    }

    /**
     * Streaming greedy decode driven by the native per-token callback.
     * Emits Token(id, bytes) per generated token, terminal Metrics + Done.
     * On native failure emits Error + Done. Cancelling the collector aborts
     * the in-flight generation cleanly via a non-zero callback return.
     *
     * `bytes` is the token's raw byte form from the embedded tokenizer
     * (empty array if absent). For end-of-stream UTF-8 text use [decode] on
     * the accumulated ids, or accumulate bytes across tokens to handle
     * multi-byte chars that span token boundaries.
     */
    fun generateFlow(
        promptIds: IntArray,
        maxNew: Int = 64,
        stopId: Int = -1
    ): Flow<GenerationEvent> = callbackFlow {
        val h = requireOpen()
        require(promptIds.isNotEmpty()) { "promptIds must not be empty" }
        require(maxNew >= 0) { "maxNew must be >= 0" }

        val cancelled = AtomicBoolean(false)
        val cb = object : RmgTokenCallback {
            override fun onToken(tokenId: Int, bytes: ByteArray?): Boolean {
                if (cancelled.get()) return false
                trySend(GenerationEvent.Token(tokenId, bytes ?: ByteArray(0)))
                return true
            }
        }

        val t0 = System.nanoTime()
        val job = launch(Dispatchers.IO) {
            try {
                val n = RmgNativeLib.nativeGenerateStream(h, promptIds, maxNew, stopId, cb)
                if (n < 0) {
                    trySend(GenerationEvent.Error("engine_generate_stream failed (rc=$n)"))
                } else {
                    val totalMs = (System.nanoTime() - t0) / 1_000_000f
                    val total = n + promptIds.size
                    val tps = if (totalMs > 0f) total * 1000f / totalMs else 0f
                    trySend(
                        GenerationEvent.Metrics(
                            DecodingMetrics(
                                tokensPerSecond = tps,
                                totalTimeMs = totalMs,
                                tokensEvaluated = promptIds.size,
                                tokensPredicted = n
                            )
                        )
                    )
                }
            } catch (t: Throwable) {
                trySend(GenerationEvent.Error(t.message ?: t.javaClass.simpleName))
            } finally {
                trySend(GenerationEvent.Done)
                close()
            }
        }

        awaitClose {
            cancelled.set(true)
            job.cancel()
        }
    }.buffer(Channel.UNLIMITED)

    /**
     * Zero-copy view of a single token's raw bytes from the embedded
     * tokenizer. Returns null if the model has no tokenizer or [tokenId]
     * is out of range. The returned ByteArray is a fresh copy — safe to keep.
     */
    fun tokenBytes(tokenId: Int): ByteArray? =
        RmgNativeLib.nativeTokenBytes(requireOpen(), tokenId)

    /**
     * Decode a sequence of token ids into UTF-8 text using the embedded
     * tokenizer. Returns null if the model has no tokenizer.
     */
    fun decode(tokenIds: IntArray): String? {
        val h = requireOpen()
        if (!_hasTokenizer) return null
        if (tokenIds.isEmpty()) return ""
        val bytes = RmgNativeLib.nativeDecodeTokens(h, tokenIds) ?: return null
        return String(bytes, Charsets.UTF_8)
    }

    @Synchronized
    override fun close() {
        val h = handle
        if (h != 0L) {
            handle = 0L
            _dims = null
            _hasTokenizer = false
            RmgNativeLib.nativeClose(h)
        }
    }

    private fun requireOpen(): Long {
        val h = handle
        check(h != 0L) { "RmgEngine is not loaded" }
        return h
    }

    companion object {
        @JvmStatic
        fun setLogLevel(level: RmgLogLevel) {
            RmgNativeLib.nativeSetLogLevel(level.value)
        }
    }
}
