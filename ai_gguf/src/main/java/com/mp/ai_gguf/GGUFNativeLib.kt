package com.mp.ai_gguf

import android.util.Log
import com.mp.ai_gguf.models.StreamCallback

class GGUFNativeLib {

    companion object {
        private const val TAG = "GGUFNativeLib"

        init {
            System.loadLibrary("ai_gguf")
        }

        object LowEnd {
            const val N_CTX = 1024
            const val N_BATCH = 512
            const val N_THREADS = 0
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        object MidRange {
            const val N_CTX = 2048
            const val N_BATCH = 512
            const val N_THREADS = 0
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        object HighEnd {
            const val N_CTX = 4096
            const val N_BATCH = 1024
            const val N_THREADS = 0
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        fun recommendedContextSize(availableMemoryMB: Int, modelSizeMB: Int): Int {
            val free = availableMemoryMB - modelSizeMB
            return when {
                free < 1024 -> 512
                free < 2048 -> 1024
                free < 4096 -> 2048
                else -> 4096
            }
        }
    }

    // ── Model Lifecycle ──────────────────────────────────────

    external fun nativeLoadModel(
        path: String,
        nCtx: Int = 4096,
        nBatch: Int = 512,
        nUbatch: Int = 512,
        nThreads: Int = 0,
        flashAttn: Boolean = true,
        useMmap: Boolean = true,
        useMlock: Boolean = false,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
        backendPath: String = ""
    ): Boolean

    external fun nativeLoadModelFromFd(
        fd: Int,
        nCtx: Int = 4096,
        nBatch: Int = 512,
        nUbatch: Int = 512,
        nThreads: Int = 0,
        flashAttn: Boolean = true,
        cacheTypeK: String = "q8_0",
        cacheTypeV: String = "q8_0",
        backendPath: String = ""
    ): Boolean

    external fun nativeRelease(): Boolean
    external fun nativeIsLoaded(): Boolean

    // ── Sampling ─────────────────────────────────────────────

    external fun nativeSetSampling(
        temperature: Float = 0.8f,
        topK: Int = 40,
        topP: Float = 0.95f,
        minP: Float = 0.05f,
        repeatPenalty: Float = 1.0f,
        penaltyLastN: Int = 64,
        frequencyPenalty: Float = 0.0f,
        presencePenalty: Float = 0.0f,
        seed: Int = -1,
        dryMultiplier: Float = 0.0f,
        dryBase: Float = 1.75f,
        dryAllowedLength: Int = 2,
        dryPenaltyLastN: Int = -1,
        xtcProbability: Float = 0.0f,
        xtcThreshold: Float = 0.1f,
        mirostat: Int = 0,
        mirostatTau: Float = 5.0f,
        mirostatEta: Float = 0.1f
    )

    // ── Generation ───────────────────────────────────────────

    external fun nativeGenerateStream(
        prompt: String,
        maxTokens: Int = -1,
        callback: StreamCallback
    ): Boolean

    external fun nativeGenerateStreamMultiTurn(
        messagesJson: String,
        maxTokens: Int = -1,
        callback: StreamCallback
    ): Boolean

    external fun nativeStopGeneration()

    // ── Configuration ────────────────────────────────────────

    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeSetChatTemplate(template: String)
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetToolChoice(choice: String)
    external fun nativeEnableToolCalling(enabled: Boolean)
    external fun nativeIsToolCallingEnabled(): Boolean
    external fun nativeSetGrammarMode(mode: Int)
    external fun nativeSetGrammar(grammar: String)
    external fun nativeSetStopStrings(strings: Array<String>)

    // ── Interventions ────────────────────────────────────────

    external fun nativeSetLogitBias(tokenIds: IntArray, biases: FloatArray)
    external fun nativeSetHeadScales(scales: FloatArray)
    external fun nativeResetHeadScales()
    external fun nativeSetAttentionTemperatureProfile(temps: FloatArray)
    external fun nativeResetAttentionTemperatures()
    external fun nativeSetResidualGates(attnGates: FloatArray, ffnGates: FloatArray)
    external fun nativeResetResidualGates()
    external fun nativeSetNormOffsets(layer: Int, offsets: FloatArray)
    external fun nativeResetNormOffsets()
    external fun nativeSetAttentionBias(layer: Int, biases: FloatArray)
    external fun nativeClearAttentionBias()
    external fun nativeClearAllInterventions()

    // ── Personality Engine (Stubs — C++ bridges pending) ─────
    // These methods are called by ControlVectorManager for the
    // Character Intelligence Engine. Currently Kotlin stubs that
    // return failure/no-op. Will be replaced with JNI bridges
    // once the engine C++ layer wraps the llama.h functions.

    /** System A: Compute contrastive personality vectors from prompts. */
    fun nativeComputePersonalityVectors(
        promptsJson: String, strengthsJson: String, cacheDir: String
    ): Boolean {
        Log.d(TAG, "nativeComputePersonalityVectors: stub (C++ bridge pending)")
        return false
    }

    /** System A: Apply emotion-gated dimensional control vectors. */
    fun nativeApplyEmotionGatedVectors(
        strengthsJson: String, emotionsJson: String, cacheDir: String, scale: Float
    ): Boolean {
        Log.d(TAG, "nativeApplyEmotionGatedVectors: stub (C++ bridge pending)")
        return false
    }

    /** System A: Clear all active control vectors. */
    fun nativeClearControlVector() {
        Log.d(TAG, "nativeClearControlVector: stub → clearing all interventions")
        nativeClearAllInterventions()
    }

    /** System B: Set logit biases from JSON array of {token, bias} entries.
     *  Resolves text tokens to IDs via nativeTokenize(). */
    fun nativeSetLogitBias(biasJson: String) {
        try {
            val arr = org.json.JSONArray(biasJson)
            if (arr.length() == 0) {
                nativeSetLogitBias(IntArray(0), FloatArray(0))
                return
            }
            val tokenIds = mutableListOf<Int>()
            val biases = mutableListOf<Float>()
            for (i in 0 until arr.length()) {
                val entry = arr.getJSONObject(i)
                val tokenText = entry.optString("token", "")
                val bias = entry.optDouble("bias", 0.0).toFloat()
                if (tokenText.isEmpty() || bias == 0f) continue
                // Resolve text to token IDs
                val ids = nativeTokenize(tokenText, false)
                if (ids != null && ids.isNotEmpty()) {
                    for (id in ids) {
                        tokenIds.add(id)
                        biases.add(bias)
                    }
                }
            }
            if (tokenIds.isNotEmpty()) {
                nativeSetLogitBias(tokenIds.toIntArray(), biases.toFloatArray())
                Log.d(TAG, "Logit bias: ${tokenIds.size} token IDs resolved from ${arr.length()} entries")
            } else {
                nativeSetLogitBias(IntArray(0), FloatArray(0))
            }
        } catch (e: Exception) {
            Log.w(TAG, "nativeSetLogitBias(json) parse error: ${e.message}")
        }
    }

    /** System D: Probe head importance and set head scales from direction vectors. */
    fun nativeProbeAndSetHeadScales(strengthsJson: String, cacheDir: String): String {
        Log.d(TAG, "nativeProbeAndSetHeadScales: stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** System E: Set attention temperature profile from JSON (backward-compat overload). */
    fun nativeSetAttentionTemperatureProfile(profileJson: String) {
        try {
            val obj = org.json.JSONObject(profileJson)
            val early = obj.optDouble("early", 1.0).toFloat()
            val mid = obj.optDouble("mid", 1.0).toFloat()
            val late = obj.optDouble("late", 1.0).toFloat()
            val nLayers = nativeLayerCount()
            if (nLayers <= 0) return
            val temps = FloatArray(nLayers) { i ->
                val ratio = i.toFloat() / (nLayers - 1).coerceAtLeast(1)
                when {
                    ratio < 0.33f -> early
                    ratio < 0.66f -> mid
                    else -> late
                }
            }
            nativeSetAttentionTemperatureProfile(temps)
        } catch (e: Exception) {
            Log.w(TAG, "nativeSetAttentionTemperatureProfile(json) error: ${e.message}")
        }
    }

    /** Gated Residual: Set residual gates from JSON (backward-compat overload). */
    fun nativeSetResidualGates(gatesJson: String) {
        try {
            val obj = org.json.JSONObject(gatesJson)
            val attnArr = obj.optJSONArray("attn")
            val ffnArr = obj.optJSONArray("ffn")
            if (attnArr == null || ffnArr == null) return
            val attn = FloatArray(attnArr.length()) { attnArr.optDouble(it, 1.0).toFloat() }
            val ffn = FloatArray(ffnArr.length()) { ffnArr.optDouble(it, 1.0).toFloat() }
            nativeSetResidualGates(attn, ffn)
        } catch (e: Exception) {
            Log.w(TAG, "nativeSetResidualGates(json) error: ${e.message}")
        }
    }

    /** System G: Set norm offsets from JSON (backward-compat overload). */
    fun nativeSetNormOffsets(strengthsJson: String, cacheDir: String, scale: Float): String {
        Log.d(TAG, "nativeSetNormOffsets(json): stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** System F: Initialize fast weight associative memory. */
    fun nativeFastWeightInit(dim: Int, gamma: Float, eta: Float, inject: Float): Boolean {
        Log.d(TAG, "nativeFastWeightInit: stub (C++ bridge pending)")
        return false
    }

    /** System F: Update fast weight memory with current activations. */
    fun nativeFastWeightUpdate() {
        // no-op until C++ bridge
    }

    /** System F: Reset fast weight memory. */
    fun nativeFastWeightReset() {
        // no-op until C++ bridge
    }

    /** System F: Get fast weight memory state as JSON. */
    fun nativeFastWeightGetState(): String {
        return """{"initialized":false}"""
    }

    /** System P5: Initialize sparse masks (all neurons active). */
    fun nativeInitSparseMasks(keepRatio: Float): Boolean {
        Log.d(TAG, "nativeInitSparseMasks: stub (C++ bridge pending)")
        return false
    }

    /** System P5: Update sparse masks from recent text activations. */
    fun nativeUpdateSparseMasks(text: String, keepRatio: Float, momentum: Float): String {
        Log.d(TAG, "nativeUpdateSparseMasks: stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** System P5: Reset all sparse masks. */
    fun nativeResetSparseMasks() {
        // no-op until C++ bridge
    }

    /** System P4: Initialize hypernetwork from cached direction vectors. */
    fun nativeInitHypernetworkFromDirections(
        strengthsJson: String, cacheDir: String, rank: Int, strength: Float
    ): String {
        Log.d(TAG, "nativeInitHypernetworkFromDirections: stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** System P4: Reset hypernetwork (remove all LoRA matrices). */
    fun nativeResetHypernetwork() {
        // no-op until C++ bridge
    }

    /** System P6: Initialize KAN-lite learnable activation overlay. */
    fun nativeInitKan(alpha: Float): Boolean {
        Log.d(TAG, "nativeInitKan: stub (C++ bridge pending)")
        return false
    }

    /** System P6: Reset KAN overlay. */
    fun nativeResetKan() {
        // no-op until C++ bridge
    }

    /** System P7: Forward-only learning step (SPSA perturbation). */
    fun nativeForwardLearnStep(
        text: String, learningRate: Float, noiseScale: Float, maxTokens: Int
    ): String {
        Log.d(TAG, "nativeForwardLearnStep: stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** Activation capture: Enable/disable hidden state capture for emotion probing. */
    fun nativeSetCaptureEnabled(enabled: Boolean) {
        Log.d(TAG, "nativeSetCaptureEnabled($enabled): stub (C++ bridge pending)")
    }

    /** Emotion probing: Probe residual stream for emotional state. */
    fun nativeProbeEmotionAxes(cacheDir: String): String {
        Log.d(TAG, "nativeProbeEmotionAxes: stub (C++ bridge pending)")
        return """{"success":false,"error":"C++ bridge not yet implemented"}"""
    }

    /** State persistence: Save learnable intervention state to file. */
    fun nativeSaveInterventionState(path: String): Boolean {
        Log.d(TAG, "nativeSaveInterventionState: stub (C++ bridge pending)")
        return false
    }

    /** State persistence: Load learnable intervention state from file. */
    fun nativeLoadInterventionState(path: String): Boolean {
        Log.d(TAG, "nativeLoadInterventionState: stub (C++ bridge pending)")
        return false
    }

    /** Check if model supports tool calling (backward-compat stub). */
    fun nativeIsToolCallingSupported(): Boolean {
        // Any loaded model can do tool calling via grammar constraints
        return nativeIsLoaded()
    }

    /** Set typed grammar enforcement (backward-compat stub). */
    fun nativeSetTypedGrammar(enabled: Boolean) {
        Log.d(TAG, "nativeSetTypedGrammar($enabled): stub (now uses nativeSetGrammarMode)")
    }

    /** Update sampler params from JSON (backward-compat stub). */
    fun nativeUpdateSamplerParams(paramsJson: String): Boolean {
        Log.d(TAG, "nativeUpdateSamplerParams: stub (use nativeSetSampling instead)")
        return false
    }

    /** Load control vectors from JSON (backward-compat stub). */
    fun nativeLoadControlVectors(vectorsJson: String): Boolean {
        Log.d(TAG, "nativeLoadControlVectors: stub (C++ bridge pending)")
        return false
    }

    /** Get KV cache state size in bytes (backward-compat). */
    fun nativeGetStateSize(): Long {
        Log.d(TAG, "nativeGetStateSize: stub")
        return 0L
    }

    /** Save KV cache state to file (backward-compat alias). */
    fun nativeStateSaveToFile(path: String): Boolean = nativeSaveState(path)

    /** Load KV cache state from file (backward-compat alias). */
    fun nativeStateLoadFromFile(path: String): Boolean = nativeLoadState(path)

    /** Get model info as JSON (backward-compat alias). */
    fun nativeGetModelInfo(): String = nativeModelInfo()

    /** Encode text to embeddings with callback (backward-compat). */
    fun nativeEncodeText(text: String, normalize: Boolean, callback: com.mp.ai_gguf.models.EmbeddingCallback): Boolean {
        return try {
            val result = nativeEmbed(text)
            if (result != null) {
                callback.onComplete(com.mp.ai_gguf.models.EmbeddingResult(embeddings = result))
                true
            } else {
                callback.onError("Embedding returned null")
                false
            }
        } catch (e: Exception) {
            callback.onError(e.message ?: "Embedding failed")
            false
        }
    }

    // ── LoRA ─────────────────────────────────────────────────

    external fun nativeApplyLora(path: String, scale: Float = 1.0f): Boolean
    external fun nativeRemoveLora(path: String)
    external fun nativeClearLora()

    // ── State ────────────────────────────────────────────────

    external fun nativeSaveState(path: String): Boolean
    external fun nativeLoadState(path: String): Boolean

    // ── Cache ────────────────────────────────────────────────

    external fun nativeClearCache()

    // ── Embeddings ───────────────────────────────────────────

    external fun nativeLoadEmbeddingModel(
        path: String,
        nCtx: Int = 512,
        nThreads: Int = 0,
        backendPath: String = ""
    ): Boolean

    external fun nativeLoadEmbeddingModelFromFd(
        fd: Int,
        nCtx: Int = 512,
        nThreads: Int = 0,
        backendPath: String = ""
    ): Boolean

    external fun nativeEmbed(text: String): FloatArray?
    external fun nativeReleaseEmbeddingModel()
    external fun nativeGetEmbeddingModelInfo(): String

    // ── Info ─────────────────────────────────────────────────

    external fun nativeModelInfo(): String
    external fun nativeBackendInfo(): String
    external fun nativeContextSize(): Int
    external fun nativeVocabSize(): Int
    external fun nativeLayerCount(): Int

    // ── Benchmark ────────────────────────────────────────────

    external fun nativeBench(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 1): String

    // ── Tokenization ─────────────────────────────────────────

    external fun nativeTokenize(text: String, addSpecial: Boolean = true): IntArray?
    external fun nativeDetokenize(tokens: IntArray): String

    // ── Thermal ──────────────────────────────────────────────

    external fun nativeGetThermalState(): String
    external fun nativeGetThermalLevel(): Int

    // ── Backend Config ───────────────────────────────────────

    external fun nativeSetOpenCLCacheDir(path: String)
    external fun nativeSetGPUCacheDir(path: String)

}
