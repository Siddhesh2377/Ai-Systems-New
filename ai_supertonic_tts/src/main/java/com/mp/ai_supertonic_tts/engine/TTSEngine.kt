package com.mp.ai_supertonic_tts.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.mp.ai_supertonic_tts.SupertonicNativeLib
import com.mp.ai_supertonic_tts.callback.TTSCallback
import com.mp.ai_supertonic_tts.models.SynthesisResult
import com.mp.ai_supertonic_tts.models.TTSConfig
import com.mp.ai_supertonic_tts.models.VoiceStyle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * ONNX inference engine for Supertonic TTS.
 *
 * Orchestrates the 4-model pipeline:
 * 1. Duration Predictor → predicts speech duration
 * 2. Text Encoder → encodes text into embeddings
 * 3. Vector Estimator → iterative flow-matching denoising
 * 4. Vocoder → converts latents to audio waveform
 */
class TTSEngine(private val nativeLib: SupertonicNativeLib) {

    private var environment: OrtEnvironment? = null
    private var dpSession: OrtSession? = null
    private var teSession: OrtSession? = null
    private var veSession: OrtSession? = null
    private var vocSession: OrtSession? = null

    private var textProcessor: TextProcessor? = null
    private val voiceStyles = mutableMapOf<String, VoiceStyle>()

    // Model config from tts.json
    private var sampleRate: Int = 44100
    private var baseChunkSize: Int = 512
    private var chunkCompressFactor: Int = 6
    private var latentDim: Int = 24

    private val latentDimTotal: Int get() = latentDim * chunkCompressFactor

    var lastError: String? = null
        private set

    fun isLoaded(): Boolean = dpSession != null && teSession != null &&
            veSession != null && vocSession != null && textProcessor != null

    /**
     * Load Supertonic ONNX models from a directory.
     *
     * Expected directory structure:
     * ```
     * modelDir/
     *   onnx/
     *     duration_predictor.onnx
     *     text_encoder.onnx
     *     vector_estimator.onnx
     *     vocoder.onnx
     *     tts.json
     *     unicode_indexer.json
     *   voice_styles/
     *     F1.json, F2.json, ..., M1.json, M2.json, ...
     * ```
     *
     * @param modelDir Root directory containing onnx/ and voice_styles/
     * @param useNNAPI Enable NNAPI acceleration
     * @return true if all models loaded successfully
     */
    fun loadModel(modelDir: String, useNNAPI: Boolean = false): Boolean {
        try {
            release()

            val onnxDir = "$modelDir/onnx"
            val voiceDir = "$modelDir/voice_styles"

            // Verify required files exist
            val requiredFiles = listOf(
                "$onnxDir/duration_predictor.onnx",
                "$onnxDir/text_encoder.onnx",
                "$onnxDir/vector_estimator.onnx",
                "$onnxDir/vocoder.onnx",
                "$onnxDir/tts.json",
                "$onnxDir/unicode_indexer.json"
            )
            for (f in requiredFiles) {
                if (!File(f).exists()) {
                    lastError = "Missing required file: $f"
                    return false
                }
            }

            // Load config
            loadConfig("$onnxDir/tts.json")

            // Load unicode indexer
            val indexer = TextProcessor.loadUnicodeIndexer("$onnxDir/unicode_indexer.json")
            textProcessor = TextProcessor(indexer)

            // Create ONNX environment and sessions
            environment = OrtEnvironment.getEnvironment()
            val env = environment!!

            val opts = OrtSession.SessionOptions()
            if (useNNAPI) {
                try {
                    opts.addNnapi()
                } catch (_: Exception) {
                    // NNAPI not available, fall back to CPU
                }
            }

            dpSession = env.createSession("$onnxDir/duration_predictor.onnx", opts)
            teSession = env.createSession("$onnxDir/text_encoder.onnx", opts)
            veSession = env.createSession("$onnxDir/vector_estimator.onnx", opts)
            vocSession = env.createSession("$onnxDir/vocoder.onnx", opts)

            // Load voice styles
            loadVoiceStyles(voiceDir)

            lastError = null
            return true
        } catch (e: Exception) {
            lastError = "Failed to load model: ${e.message}"
            release()
            return false
        }
    }

    /**
     * Get list of available voice style names.
     */
    fun getAvailableVoices(): List<String> = voiceStyles.keys.sorted()

    /**
     * Synthesize speech from text.
     *
     * @param text Input text to synthesize
     * @param config Synthesis configuration
     * @param callback Optional progress callback
     * @return SynthesisResult with audio data
     */
    suspend fun synthesize(
        text: String,
        config: TTSConfig,
        callback: TTSCallback? = null
    ): SynthesisResult = withContext(Dispatchers.Default) {
        if (!isLoaded()) {
            throw IllegalStateException("Model not loaded. Call loadModel() first.")
        }

        val style = voiceStyles[config.voice]
            ?: throw IllegalArgumentException("Voice '${config.voice}' not found. Available: ${getAvailableVoices()}")

        val startTime = System.currentTimeMillis()

        // Chunk text if needed
        val chunks = if (config.chunkingEnabled) {
            TextChunker.chunk(text, config.language)
        } else {
            listOf(text)
        }

        callback?.onSynthesisStart(text.length, chunks.size)

        val audioSegments = mutableListOf<FloatArray>()
        var totalSamples = 0
        var totalDuration = 0f
        val silenceSamples = (config.chunkSilenceMs * sampleRate / 1000)

        for ((index, chunk) in chunks.withIndex()) {
            if (chunk.isBlank()) continue

            val chunkAudio = synthesizeChunk(chunk, config, style)
            val chunkDur = chunkAudio.size.toFloat() / sampleRate

            // Add silence gap between chunks (not before first audio segment)
            if (audioSegments.isNotEmpty() && config.chunkSilenceMs > 0) {
                val silence = FloatArray(silenceSamples)
                audioSegments.add(silence)
                totalSamples += silenceSamples
                totalDuration += config.chunkSilenceMs / 1000f
            }

            audioSegments.add(chunkAudio)
            totalSamples += chunkAudio.size
            totalDuration += chunkDur

            callback?.onChunkProgress(index + 1, chunks.size)
        }

        // Concatenate all segments into one array
        val audioArray = FloatArray(totalSamples)
        var offset = 0
        for (segment in audioSegments) {
            System.arraycopy(segment, 0, audioArray, offset, segment.size)
            offset += segment.size
        }

        // Clip audio to [-1, 1]
        nativeLib.nativeClipAudio(audioArray)

        val synthesisTime = System.currentTimeMillis() - startTime
        val durationMs = (totalDuration * 1000).toLong()

        val result = SynthesisResult(
            audioData = audioArray,
            sampleRate = sampleRate,
            channels = 1,
            durationMs = durationMs,
            synthesisTimeMs = synthesisTime
        )

        callback?.onAudioReady(result)
        result
    }

    /**
     * Synthesize a single chunk of text (no chunking).
     */
    private fun synthesizeChunk(text: String, config: TTSConfig, style: VoiceStyle): FloatArray {
        val env = environment ?: throw IllegalStateException("Environment not initialized")

        // 1. Process text
        val processed = textProcessor!!.process(text, config.language)
        val seqLen = processed.sequenceLength

        // Create tensors
        val textIdsTensor = createLongTensor(
            processed.textIds, longArrayOf(1, seqLen.toLong()), env
        )
        val textMaskTensor = createFloatTensor(
            processed.textMask, longArrayOf(1, 1, seqLen.toLong()), env
        )
        val styleDpTensor = createFloatTensor(
            style.styleDp, style.styleDpShape, env
        )
        val styleTtlTensor = createFloatTensor(
            style.styleTtl, style.styleTtlShape, env
        )

        try {
            // 2. Duration prediction
            val dpInputs = mapOf(
                "text_ids" to textIdsTensor,
                "style_dp" to styleDpTensor,
                "text_mask" to textMaskTensor
            )
            val dpResult = dpSession!!.run(dpInputs)
            val durationRaw = extractFloatOutput(dpResult)
            dpResult.close()

            val duration = durationRaw / config.speed

            // 3. Text encoding
            val teInputs = mapOf(
                "text_ids" to textIdsTensor,
                "style_ttl" to styleTtlTensor,
                "text_mask" to textMaskTensor
            )
            val teResult = teSession!!.run(teInputs)
            val textEmbTensor = teResult.get(0) as OnnxTensor

            // 4. Initialize noisy latent
            val wavLen = (duration * sampleRate).toLong()
            val chunkSize = baseChunkSize.toLong() * chunkCompressFactor.toLong()
            val latentLen = ((wavLen + chunkSize - 1) / chunkSize).toInt()

            val noisyLatent = generateGaussianNoise(latentDimTotal * latentLen)
            val latentMask = FloatArray(latentLen) { 1.0f }

            // Apply mask to noisy latent
            for (d in 0 until latentDimTotal) {
                for (t in 0 until latentLen) {
                    noisyLatent[d * latentLen + t] *= latentMask[t]
                }
            }

            // 5. Denoising loop (Euler integration)
            var xt = noisyLatent
            val totalSteps = config.steps
            val totalStepArr = floatArrayOf(totalSteps.toFloat())
            val totalStepTensor = OnnxTensor.createTensor(env, totalStepArr)

            for (step in 0 until totalSteps) {
                val currentStepArr = floatArrayOf(step.toFloat())
                val currentStepTensor = OnnxTensor.createTensor(env, currentStepArr)
                val xtTensor = createFloatTensor(
                    xt, longArrayOf(1, latentDimTotal.toLong(), latentLen.toLong()), env
                )
                val latentMaskTensor2 = createFloatTensor(
                    latentMask, longArrayOf(1, 1, latentLen.toLong()), env
                )
                // Recreate textMask tensor for vector estimator
                val textMask2 = createFloatTensor(
                    processed.textMask, longArrayOf(1, 1, seqLen.toLong()), env
                )

                val veInputs = mapOf(
                    "noisy_latent" to xtTensor,
                    "text_emb" to textEmbTensor,
                    "style_ttl" to styleTtlTensor,
                    "latent_mask" to latentMaskTensor2,
                    "text_mask" to textMask2,
                    "current_step" to currentStepTensor,
                    "total_step" to totalStepTensor
                )

                val veResult = veSession!!.run(veInputs)
                xt = extractFlatFloatOutput(veResult, latentDimTotal * latentLen)
                veResult.close()

                currentStepTensor.close()
                xtTensor.close()
                latentMaskTensor2.close()
                textMask2.close()
            }

            totalStepTensor.close()
            teResult.close()

            // 6. Vocoding
            val latentTensor = createFloatTensor(
                xt, longArrayOf(1, latentDimTotal.toLong(), latentLen.toLong()), env
            )
            val vocInputs = mapOf("latent" to latentTensor)
            val vocResult = vocSession!!.run(vocInputs)

            val wavTensor = vocResult.get(0) as OnnxTensor
            val wavInfo = wavTensor.info as ai.onnxruntime.TensorInfo
            val wavTotalSize = wavInfo.shape.fold(1L) { acc, v -> acc * v }.toInt()
            val wavBuf = wavTensor.floatBuffer
            val wav = FloatArray(wavTotalSize)
            wavBuf.get(wav)

            vocResult.close()
            latentTensor.close()

            // Trim to actual audio length
            val actualLen = min(wav.size, (duration * sampleRate).toInt())
            return wav.copyOf(actualLen)

        } finally {
            textIdsTensor.close()
            textMaskTensor.close()
            styleDpTensor.close()
            styleTtlTensor.close()
        }
    }

    /**
     * Release all ONNX resources.
     */
    fun release() {
        dpSession?.close(); dpSession = null
        teSession?.close(); teSession = null
        veSession?.close(); veSession = null
        vocSession?.close(); vocSession = null
        environment = null
        textProcessor = null
        voiceStyles.clear()
    }

    // ========================================================================
    // PRIVATE HELPERS
    // ========================================================================

    private fun loadConfig(path: String) {
        val json = JSONObject(File(path).readText())
        val ae = json.getJSONObject("ae")
        sampleRate = ae.getInt("sample_rate")
        baseChunkSize = ae.getInt("base_chunk_size")

        val ttl = json.getJSONObject("ttl")
        chunkCompressFactor = ttl.getInt("chunk_compress_factor")
        latentDim = ttl.getInt("latent_dim")
    }

    private fun loadVoiceStyles(voiceDir: String) {
        val dir = File(voiceDir)
        if (!dir.exists() || !dir.isDirectory) return

        dir.listFiles()?.filter { it.extension == "json" }?.forEach { file ->
            try {
                val style = VoiceStyle.loadFromJson(file.absolutePath)
                voiceStyles[style.name] = style
            } catch (_: Exception) {
                // Skip invalid voice files
            }
        }
    }

    /**
     * Generate Gaussian noise using Box-Muller transform.
     */
    private fun generateGaussianNoise(size: Int): FloatArray {
        val result = FloatArray(size)
        val rng = Random()
        var i = 0
        while (i < size - 1) {
            val u1 = maxOf(1e-7f, rng.nextFloat())
            val u2 = rng.nextFloat()
            val mag = sqrt(-2.0f * ln(u1))
            result[i] = mag * cos(2.0f * PI.toFloat() * u2)
            result[i + 1] = mag * sin(2.0f * PI.toFloat() * u2)
            i += 2
        }
        if (size % 2 == 1) {
            val u1 = maxOf(1e-7f, rng.nextFloat())
            val u2 = rng.nextFloat()
            result[size - 1] = sqrt(-2.0f * ln(u1)) * cos(2.0f * PI.toFloat() * u2)
        }
        return result
    }

    /**
     * Extract the first scalar float from an ONNX result.
     */
    private fun extractFloatOutput(result: OrtSession.Result): Float {
        val tensor = result.get(0) as OnnxTensor
        val value = tensor.value
        return when (value) {
            is FloatArray -> value[0]
            is Array<*> -> {
                @Suppress("UNCHECKED_CAST")
                val arr = value as Array<FloatArray>
                arr[0][0]
            }
            is Float -> value
            else -> throw IllegalStateException("Unexpected output type: ${value?.javaClass}")
        }
    }

    /**
     * Extract a flat float array from an ONNX tensor result.
     */
    private fun extractFlatFloatOutput(result: OrtSession.Result, expectedSize: Int): FloatArray {
        val tensor = result.get(0) as OnnxTensor
        val buf = tensor.floatBuffer
        val arr = FloatArray(expectedSize)
        buf.get(arr)
        return arr
    }

    /**
     * Create an ONNX float tensor from a flat array + shape.
     */
    private fun createFloatTensor(
        data: FloatArray, shape: LongArray, env: OrtEnvironment
    ): OnnxTensor {
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
    }

    /**
     * Create an ONNX long tensor from a flat array + shape.
     */
    private fun createLongTensor(
        data: LongArray, shape: LongArray, env: OrtEnvironment
    ): OnnxTensor {
        return OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
    }
}
