package com.dark.ai_sd

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Main manager class for Stable Diffusion operations.
 *
 * Manages the full lifecycle: runtime setup, model loading, image generation,
 * and resource cleanup. Uses JNI native calls directly (no HTTP server).
 *
 * Thread safety: all native calls are protected by mutexes in C++.
 */
class DiffusionManager(private val context: Context) {

    companion object {
        private const val TAG = "DiffusionManager"
        private const val RUNTIME_DIR = "runtime_libs"
        private const val SAFETY_CHECKER_FILE = "safety_checker/safety_checker.mnn"

        @SuppressLint("StaticFieldLeak")
        @Volatile
        private var instance: DiffusionManager? = null

        fun getInstance(context: Context): DiffusionManager {
            return instance ?: synchronized(this) {
                instance ?: DiffusionManager(context.applicationContext).also { instance = it }
            }
        }
    }

    private val nativeLib = SDNativeLib()
    private val safetyCheckerFile = File(context.filesDir, SAFETY_CHECKER_FILE)

    // State management — all updates go through synchronized helpers to prevent
    // inconsistent observations from concurrent threads (Bug 13 fix)
    private val stateLock = Any()
    private val _backendState = MutableStateFlow<DiffusionBackendState>(DiffusionBackendState.Idle)
    val diffusionBackendState: StateFlow<DiffusionBackendState> = _backendState.asStateFlow()

    private val _generationState = MutableStateFlow<DiffusionGenerationState>(DiffusionGenerationState.Idle)
    val diffusionGenerationState: StateFlow<DiffusionGenerationState> = _generationState.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private fun updateBackendState(state: DiffusionBackendState) {
        synchronized(stateLock) { _backendState.value = state }
    }

    private fun updateGenerationState(state: DiffusionGenerationState) {
        synchronized(stateLock) { _generationState.value = state }
    }

    private val _runtimeSetupState = MutableStateFlow<RuntimeSetupState>(RuntimeSetupState.Idle)
    val runtimeSetupState: StateFlow<RuntimeSetupState> = _runtimeSetupState.asStateFlow()

    private fun updateSetupState(state: RuntimeSetupState) {
        synchronized(stateLock) { _runtimeSetupState.value = state }
    }

    // LoRA state
    private val _loraState = MutableStateFlow<LoRAState>(LoRAState.None)
    val loraState: StateFlow<LoRAState> = _loraState.asStateFlow()
    private val activeLoras = mutableListOf<LoRAConfig>()

    // Runtime configuration
    private lateinit var runtimeDir: File
    private var currentModel: DiffusionModelConfig? = null
    private var isRuntimePrepared = false

    /**
     * Initialize the runtime environment.
     * Extracts QNN libraries from assets and sets up ADSP_LIBRARY_PATH.
     * Must be called before loading models.
     */
    suspend fun setupRuntimeAsync(config: DiffusionRuntimeConfig = DiffusionRuntimeConfig(RUNTIME_DIR)) {
        if (isRuntimePrepared) {
            Log.i(TAG, "Runtime already prepared")
            return
        }

        updateBackendState(DiffusionBackendState.Starting)

        withContext(Dispatchers.IO) {
            try {
                prepareRuntimeDirectory(config)

                if (config.safetyCheckerEnabled) {
                    updateSetupState(RuntimeSetupState.CopyingSafetyChecker)
                    prepareSafetyChecker(config)
                }

                // Initialize QNN runtime with the extracted library directory
                updateSetupState(RuntimeSetupState.InitializingRuntime)
                val success = nativeLib.nativeInitRuntime(runtimeDir.absolutePath)
                if (!success) {
                    throw RuntimeException("Native runtime initialization failed")
                }

                isRuntimePrepared = true
                updateSetupState(RuntimeSetupState.Complete)
                updateBackendState(DiffusionBackendState.Idle)
                Log.i(TAG, "Runtime setup completed successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Runtime setup failed", e)
                updateSetupState(RuntimeSetupState.Error("Runtime setup failed: ${e.message}"))
                updateBackendState(DiffusionBackendState.Error("Runtime setup failed: ${e.message}"))
                throw RuntimeException("Failed to setup runtime environment", e)
            }
        }
    }

    /**
     * Load a model via JNI native calls.
     * @return true if successful
     */
    fun loadModel(diffusionModelConfig: DiffusionModelConfig, width: Int = 512, height: Int = 512): Boolean {
        if (!isRuntimePrepared) {
            Log.e(TAG, "Runtime not prepared. Call setupRuntimeAsync() first")
            updateBackendState(DiffusionBackendState.Error("Runtime not prepared"))
            return false
        }

        // Release existing model if loaded
        if (_backendState.value is DiffusionBackendState.Running) {
            Log.i(TAG, "Stopping existing model before loading new one")
            stopBackend()
        }

        updateBackendState(DiffusionBackendState.Starting)

        try {
            val model = diffusionModelConfig
            val modelsDir = File(model.modelDir)

            // Determine file paths based on CPU/GPU mode
            val clipFilename = when {
                model.runOnCpu -> "clip.mnn"
                model.useCpuClip -> "clip.mnn"
                else -> "clip.bin"
            }

            val unetFilename = if (model.runOnCpu) "unet.mnn" else "unet.bin"
            val vaeDecoderFilename = if (model.runOnCpu) "vae_decoder.mnn" else "vae_decoder.bin"
            val vaeEncoderFilename = if (model.runOnCpu) "vae_encoder.mnn" else "vae_encoder.bin"

            // Early validation: fail fast if UNET file is missing
            val unetFile = File(modelsDir, unetFilename)
            if (!unetFile.exists()) {
                val mode = if (model.runOnCpu) "CPU/MNN" else "QNN"
                val msg = "UNET file not found: ${unetFile.absolutePath} ($mode mode expects $unetFilename in modelDir)"
                Log.e(TAG, msg)
                synchronized(stateLock) {
                    _backendState.value = DiffusionBackendState.Error(msg)
                }
                return false
            }

            val vaeEncoderFile = File(modelsDir, vaeEncoderFilename)
            val vaeEncoderPath = if (vaeEncoderFile.exists()) vaeEncoderFile.absolutePath else ""

            // Resolution patch
            val patchPath = if (width != 512 || height != 512) {
                findPatchFile(modelsDir, width, height)?.absolutePath ?: ""
            } else ""

            // Safety checker
            val safetyPath = if (model.safetyMode && safetyCheckerFile.exists()) {
                safetyCheckerFile.absolutePath
            } else ""

            // QNN backend paths (empty for CPU mode)
            val qnnBackendPath = if (!model.runOnCpu) {
                File(runtimeDir, "libQnnHtp.so").absolutePath
            } else ""

            val qnnSystemLibPath = if (!model.runOnCpu) {
                File(runtimeDir, "libQnnSystem.so").absolutePath
            } else ""

            Log.i(TAG, "Loading model: ${model.name}, CPU=${model.runOnCpu}")

            val success = nativeLib.nativeLoadModel(
                clipPath = File(modelsDir, clipFilename).absolutePath,
                unetPath = File(modelsDir, unetFilename).absolutePath,
                vaeDecoderPath = File(modelsDir, vaeDecoderFilename).absolutePath,
                vaeEncoderPath = vaeEncoderPath,
                tokenizerPath = File(modelsDir, "tokenizer.json").absolutePath,
                safetyCheckerPath = safetyPath,
                patchPath = patchPath,
                modelDir = modelsDir.absolutePath,
                qnnBackendPath = qnnBackendPath,
                qnnSystemLibPath = qnnSystemLibPath,
                textEmbeddingSize = model.textEmbeddingSize,
                runOnCpu = model.runOnCpu,
                useCpuClip = model.useCpuClip,
                isPony = model.isPony,
                useSafetyChecker = model.safetyMode
            )

            if (success) {
                currentModel = model
                updateBackendState(DiffusionBackendState.Running)
                Log.i(TAG, "Model loaded successfully")
            } else {
                updateBackendState(DiffusionBackendState.Error("Failed to load model"))
            }

            return success
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model", e)
            updateBackendState(DiffusionBackendState.Error("Model load failed: ${e.message}"))
            return false
        }
    }

    /**
     * Generate an image. Progress and results delivered via state flows.
     */
    fun generateImage(params: DiffusionGenerationParams) {
        // Input validation — fail fast before hitting JNI
        if (params.width % 8 != 0 || params.height % 8 != 0) {
            updateGenerationState(DiffusionGenerationState.Error(
                "Width and height must be divisible by 8 (got ${params.width}x${params.height})"))
            return
        }
        if (params.steps !in 1..150) {
            updateGenerationState(DiffusionGenerationState.Error(
                "Steps must be 1-150 (got ${params.steps})"))
            return
        }
        if (params.cfgScale < 0f) {
            updateGenerationState(DiffusionGenerationState.Error(
                "CFG scale must be >= 0 (got ${params.cfgScale})"))
            return
        }
        if (params.prompt.isBlank()) {
            updateGenerationState(DiffusionGenerationState.Error("Prompt cannot be empty"))
            return
        }

        if (!_isGenerating.compareAndSet(false, true)) {
            Log.w(TAG, "Generation already in progress")
            return
        }

        Thread({
            try {
                updateGenerationState(DiffusionGenerationState.Progress(0f))

                // Convert input image to RGB bytes if needed
                val inputImageBytes: ByteArray? = params.inputImage?.let { base64Str ->
                    try {
                        java.util.Base64.getDecoder().decode(base64Str)
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to decode input image", e)
                        null
                    }
                }

                val maskBytes: ByteArray? = params.mask?.let { base64Str ->
                    try {
                        java.util.Base64.getDecoder().decode(base64Str)
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to decode mask", e)
                        null
                    }
                }

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {
                        val progress = step.toFloat() / totalSteps
                        updateGenerationState(DiffusionGenerationState.Progress(
                            progress = progress,
                            currentStep = step,
                            totalSteps = totalSteps
                        ))
                    }

                    override fun onImageProgress(
                        step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int
                    ) {
                        val progress = step.toFloat() / totalSteps
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        updateGenerationState(DiffusionGenerationState.Progress(
                            progress = progress,
                            currentStep = step,
                            totalSteps = totalSteps,
                            intermediateImage = bitmap
                        ))
                    }

                    override fun onComplete(
                        rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int
                    ) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        updateGenerationState(DiffusionGenerationState.Complete(
                            bitmap = bitmap,
                            seed = seed,
                            width = width,
                            height = height
                        ))
                    }

                    override fun onError(message: String) {
                        updateGenerationState(DiffusionGenerationState.Error(message))
                    }
                }

                nativeLib.nativeGenerate(
                    prompt = params.prompt,
                    negativePrompt = params.negativePrompt,
                    steps = params.steps,
                    cfgScale = params.cfgScale,
                    seed = params.seed ?: 0L,
                    width = params.width,
                    height = params.height,
                    scheduler = params.scheduler,
                    useOpenCL = params.useOpenCL,
                    inputImage = inputImageBytes,
                    mask = maskBytes,
                    denoiseStrength = params.denoiseStrength,
                    showProcess = params.showDiffusionProcess,
                    showStride = params.showDiffusionStride,
                    callback = callback
                )
            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                updateGenerationState(DiffusionGenerationState.Error(e.message ?: "Unknown error"))
            } finally {
                _isGenerating.value = false
            }
        }, "SD-Generate").apply {
            isDaemon = true
            start()
        }
    }

    /**
     * Generate image synchronously (suspending).
     */
    suspend fun generateImageSync(params: DiffusionGenerationParams): DiffusionGenerationResult =
        withContext(Dispatchers.IO) {
            if (!_isGenerating.compareAndSet(false, true)) {
                return@withContext DiffusionGenerationResult.Failure("Generation already in progress")
            }
            try {
                updateGenerationState(DiffusionGenerationState.Progress(0f))

                val inputImageBytes: ByteArray? = params.inputImage?.let { base64Str ->
                    runCatching { java.util.Base64.getDecoder().decode(base64Str) }
                        .onFailure { Log.e(TAG, "Failed to decode input image", it) }
                        .getOrNull()
                }
                val maskBytes: ByteArray? = params.mask?.let { base64Str ->
                    runCatching { java.util.Base64.getDecoder().decode(base64Str) }
                        .onFailure { Log.e(TAG, "Failed to decode mask", it) }
                        .getOrNull()
                }

                var result: DiffusionGenerationResult = DiffusionGenerationResult.Failure("No result")

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {
                        val progress = step.toFloat() / totalSteps
                        updateGenerationState(DiffusionGenerationState.Progress(
                            progress = progress, currentStep = step, totalSteps = totalSteps
                        ))
                    }

                    override fun onImageProgress(
                        step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int
                    ) {
                        val progress = step.toFloat() / totalSteps
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        updateGenerationState(DiffusionGenerationState.Progress(
                            progress = progress, currentStep = step, totalSteps = totalSteps,
                            intermediateImage = bitmap
                        ))
                    }

                    override fun onComplete(
                        rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int
                    ) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        result = DiffusionGenerationResult.Success(bitmap, seed, width, height)
                        updateGenerationState(DiffusionGenerationState.Complete(bitmap, seed, width, height))
                    }

                    override fun onError(message: String) {
                        result = DiffusionGenerationResult.Failure(message)
                        updateGenerationState(DiffusionGenerationState.Error(message))
                    }
                }

                nativeLib.nativeGenerate(
                    prompt = params.prompt,
                    negativePrompt = params.negativePrompt,
                    steps = params.steps,
                    cfgScale = params.cfgScale,
                    seed = params.seed ?: 0L,
                    width = params.width,
                    height = params.height,
                    scheduler = params.scheduler,
                    useOpenCL = params.useOpenCL,
                    inputImage = inputImageBytes,
                    mask = maskBytes,
                    denoiseStrength = params.denoiseStrength,
                    showProcess = params.showDiffusionProcess,
                    showStride = params.showDiffusionStride,
                    callback = callback
                )

                result
            } catch (e: Exception) {
                Log.e(TAG, "Synchronous generation failed", e)
                DiffusionGenerationResult.Failure(e.message ?: "Unknown error")
            } finally {
                _isGenerating.value = false
            }
        }

    /**
     * Cancel ongoing generation.
     */
    fun cancelGeneration() {
        nativeLib.nativeStopGeneration()
        _isGenerating.value = false
    }

    /**
     * Restart the backend with the same model.
     */
    fun restartBackend(): Boolean {
        val model = currentModel ?: run {
            Log.e(TAG, "Cannot restart: no model loaded")
            return false
        }
        Log.i(TAG, "Restarting with model: ${model.name}")
        stopBackend()
        return loadModel(model, 512, 512)
    }

    /**
     * Stop the backend and release model resources.
     */
    fun stopBackend() {
        Log.i(TAG, "Stopping backend")
        nativeLib.nativeRelease()
        currentModel = null
        updateBackendState(DiffusionBackendState.Idle)
    }

    /**
     * Get SoC hardware info from native level.
     * Returns JSON with soc_id, machine, family, revision, htp_version, has_qnn_htp.
     */
    fun getSocInfo(): String = nativeLib.nativeGetSocInfo()

    fun getCurrentModel(): DiffusionModelConfig? = currentModel

    fun isBackendRunning(): Boolean = _backendState.value is DiffusionBackendState.Running

    fun resetGenerationState() {
        if (!_isGenerating.value) {
            updateGenerationState(DiffusionGenerationState.Idle)
        }
    }

    /**
     * Cleanup all resources.
     */
    fun cleanup() {
        cancelGeneration()
        clearLora()
        stopBackend()
        isRuntimePrepared = false
    }

    // ========================================================================
    // LoRA
    // ========================================================================

    /**
     * Apply a LoRA to the currently loaded model.
     * Requires MNN/CPU mode — QNN mode does not support runtime LoRA.
     *
     * @param config LoRA configuration (path + weight)
     * @return true if successfully applied
     */
    fun applyLora(config: LoRAConfig): Boolean {
        if (_backendState.value !is DiffusionBackendState.Running) {
            synchronized(stateLock) {
                _loraState.value = LoRAState.Error("No model loaded")
            }
            return false
        }
        if (currentModel?.runOnCpu != true) {
            synchronized(stateLock) {
                _loraState.value = LoRAState.Error("LoRA requires CPU/MNN mode")
            }
            return false
        }
        if (!File(config.path).exists()) {
            synchronized(stateLock) {
                _loraState.value = LoRAState.Error("LoRA file not found: ${config.path}")
            }
            return false
        }
        if (!config.path.endsWith(".safetensors")) {
            synchronized(stateLock) {
                _loraState.value = LoRAState.Error("LoRA must be a .safetensors file")
            }
            return false
        }
        if (config.weight !in -2f..2f) {
            synchronized(stateLock) {
                _loraState.value = LoRAState.Error("LoRA weight must be between -2.0 and 2.0")
            }
            return false
        }

        synchronized(stateLock) { _loraState.value = LoRAState.Applying }

        val success = nativeLib.nativeApplyLora(config.path, config.weight)
        if (success) {
            activeLoras.add(config)
            synchronized(stateLock) {
                _loraState.value = LoRAState.Applied(activeLoras.toList())
            }
            Log.i(TAG, "LoRA applied: ${config.path} (weight=${config.weight})")
        } else {
            synchronized(stateLock) {
                _loraState.value = if (activeLoras.isEmpty()) LoRAState.None
                    else LoRAState.Applied(activeLoras.toList())
            }
            Log.e(TAG, "Failed to apply LoRA: ${config.path}")
        }
        return success
    }

    /**
     * Remove all active LoRAs and restore original model weights.
     */
    fun clearLora() {
        if (activeLoras.isEmpty()) return
        nativeLib.nativeClearLora()
        activeLoras.clear()
        synchronized(stateLock) { _loraState.value = LoRAState.None }
        Log.i(TAG, "All LoRAs cleared")
    }

    // ========================================================================
    // Upscaler (Phase 5.1)
    // ========================================================================

    private val _upscaleState = MutableStateFlow<UpscaleState>(UpscaleState.Idle)
    val upscaleState: StateFlow<UpscaleState> = _upscaleState.asStateFlow()

    /**
     * Load a 4x upscaler model.
     *
     * @param modelPath Path to upscaler model file (.bin for QNN, .mnn for MNN)
     * @param useMnn Use MNN backend (CPU/GPU). If false, uses QNN (NPU).
     * @param useOpenCL Use OpenCL acceleration for MNN backend
     * @return true if loaded successfully
     */
    fun loadUpscaler(modelPath: String, useMnn: Boolean = true, useOpenCL: Boolean = false): Boolean {
        return nativeLib.nativeLoadUpscaler(modelPath, useMnn, useOpenCL)
    }

    /**
     * Upscale a bitmap 4x. Result delivered via [upscaleState] flow.
     *
     * @param inputBitmap Bitmap to upscale
     */
    fun upscaleImage(inputBitmap: Bitmap) {
        Thread({
            try {
                synchronized(stateLock) { _upscaleState.value = UpscaleState.Processing }

                val width = inputBitmap.width
                val height = inputBitmap.height
                val rgbBytes = bitmapToRgbBytes(inputBitmap)

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        synchronized(stateLock) {
                            _upscaleState.value = UpscaleState.Complete(bitmap, width, height, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _upscaleState.value = UpscaleState.Error(message) }
                    }
                }

                nativeLib.nativeUpscaleImage(rgbBytes, width, height, callback)

            } catch (e: Exception) {
                Log.e(TAG, "Upscale failed", e)
                synchronized(stateLock) { _upscaleState.value = UpscaleState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-Upscale").start()
    }

    /**
     * Release upscaler model resources.
     */
    fun releaseUpscaler() {
        nativeLib.nativeReleaseUpscaler()
    }

    // ========================================================================
    // Segmenter (Phase 5.3 — MobileSAM)
    // ========================================================================

    private val _segmenterState = MutableStateFlow<SegmenterState>(SegmenterState.Idle)
    val segmenterState: StateFlow<SegmenterState> = _segmenterState.asStateFlow()

    fun loadSegmenter(encoderPath: String, decoderPath: String, useOpenCL: Boolean = false): Boolean {
        return nativeLib.nativeLoadSegmenter(encoderPath, decoderPath, useOpenCL)
    }

    fun segmenterEncodeImage(inputBitmap: Bitmap): Boolean {
        val width = inputBitmap.width
        val height = inputBitmap.height
        val rgbBytes = bitmapToRgbBytes(inputBitmap)
        return nativeLib.nativeSegmenterEncodeImage(rgbBytes, width, height)
    }

    fun segmentAtPoint(x: Float, y: Float) {
        Thread({
            try {
                synchronized(stateLock) { _segmenterState.value = SegmenterState.Processing }
                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val mask = createMaskBitmap(rgbData, width, height)
                        val score = seed / 10000f  // Score was packed into seed field
                        synchronized(stateLock) {
                            _segmenterState.value = SegmenterState.Complete(mask, score, width, height, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _segmenterState.value = SegmenterState.Error(message) }
                    }
                }
                nativeLib.nativeSegmentAtPoint(x, y, callback)
            } catch (e: Exception) {
                Log.e(TAG, "Segment at point failed", e)
                synchronized(stateLock) { _segmenterState.value = SegmenterState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-SegmentPoint").start()
    }

    fun segmentWithBox(x1: Float, y1: Float, x2: Float, y2: Float) {
        Thread({
            try {
                synchronized(stateLock) { _segmenterState.value = SegmenterState.Processing }
                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val mask = createMaskBitmap(rgbData, width, height)
                        val score = seed / 10000f
                        synchronized(stateLock) {
                            _segmenterState.value = SegmenterState.Complete(mask, score, width, height, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _segmenterState.value = SegmenterState.Error(message) }
                    }
                }
                nativeLib.nativeSegmentWithBox(x1, y1, x2, y2, callback)
            } catch (e: Exception) {
                Log.e(TAG, "Segment with box failed", e)
                synchronized(stateLock) { _segmenterState.value = SegmenterState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-SegmentBox").start()
    }

    fun releaseSegmenter() {
        nativeLib.nativeReleaseSegmenter()
        synchronized(stateLock) { _segmenterState.value = SegmenterState.Idle }
    }

    // ========================================================================
    // LaMa Inpainter (Phase 5.4)
    // ========================================================================

    private val _lamaState = MutableStateFlow<LamaState>(LamaState.Idle)
    val lamaState: StateFlow<LamaState> = _lamaState.asStateFlow()

    fun loadLamaInpainter(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return nativeLib.nativeLoadLamaInpainter(modelPath, useOpenCL)
    }

    fun lamaInpaint(inputBitmap: Bitmap, maskBitmap: Bitmap) {
        Thread({
            try {
                synchronized(stateLock) { _lamaState.value = LamaState.Processing }
                val width = inputBitmap.width
                val height = inputBitmap.height
                val rgbBytes = bitmapToRgbBytes(inputBitmap)
                val maskBytes = bitmapToRgbBytes(maskBitmap)

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        synchronized(stateLock) {
                            _lamaState.value = LamaState.Complete(bitmap, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _lamaState.value = LamaState.Error(message) }
                    }
                }
                nativeLib.nativeLamaInpaint(rgbBytes, maskBytes, width, height, callback)
            } catch (e: Exception) {
                Log.e(TAG, "LaMa inpaint failed", e)
                synchronized(stateLock) { _lamaState.value = LamaState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-LamaInpaint").start()
    }

    fun releaseLamaInpainter() {
        nativeLib.nativeReleaseLamaInpainter()
        synchronized(stateLock) { _lamaState.value = LamaState.Idle }
    }

    // ========================================================================
    // Depth Estimator (Phase 5.5)
    // ========================================================================

    private val _depthState = MutableStateFlow<DepthState>(DepthState.Idle)
    val depthState: StateFlow<DepthState> = _depthState.asStateFlow()

    fun loadDepthEstimator(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return nativeLib.nativeLoadDepthEstimator(modelPath, useOpenCL)
    }

    fun estimateDepth(inputBitmap: Bitmap) {
        Thread({
            try {
                synchronized(stateLock) { _depthState.value = DepthState.Processing }
                val width = inputBitmap.width
                val height = inputBitmap.height
                val rgbBytes = bitmapToRgbBytes(inputBitmap)

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        synchronized(stateLock) {
                            _depthState.value = DepthState.Complete(bitmap, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _depthState.value = DepthState.Error(message) }
                    }
                }
                nativeLib.nativeEstimateDepthColorized(rgbBytes, width, height, callback)
            } catch (e: Exception) {
                Log.e(TAG, "Depth estimation failed", e)
                synchronized(stateLock) { _depthState.value = DepthState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-Depth").start()
    }

    fun releaseDepthEstimator() {
        nativeLib.nativeReleaseDepthEstimator()
        synchronized(stateLock) { _depthState.value = DepthState.Idle }
    }

    // ========================================================================
    // Style Transfer (Phase 5.6)
    // ========================================================================

    private val _styleState = MutableStateFlow<StyleState>(StyleState.Idle)
    val styleState: StateFlow<StyleState> = _styleState.asStateFlow()

    fun loadStyleTransfer(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return nativeLib.nativeLoadStyleTransfer(modelPath, useOpenCL)
    }

    fun stylize(contentBitmap: Bitmap, styleBitmap: Bitmap, strength: Float = 1.0f) {
        Thread({
            try {
                synchronized(stateLock) { _styleState.value = StyleState.Processing }
                val contentRgb = bitmapToRgbBytes(contentBitmap)
                val styleRgb = bitmapToRgbBytes(styleBitmap)

                val callback = object : SDCallback {
                    override fun onProgress(step: Int, totalSteps: Int) {}
                    override fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int) {}
                    override fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int) {
                        val bitmap = createBitmapFromRgb(rgbData, width, height)
                        synchronized(stateLock) {
                            _styleState.value = StyleState.Complete(bitmap, generationTimeMs)
                        }
                    }
                    override fun onError(message: String) {
                        synchronized(stateLock) { _styleState.value = StyleState.Error(message) }
                    }
                }
                nativeLib.nativeStylize(
                    contentRgb, contentBitmap.width, contentBitmap.height,
                    styleRgb, styleBitmap.width, styleBitmap.height,
                    strength, callback
                )
            } catch (e: Exception) {
                Log.e(TAG, "Style transfer failed", e)
                synchronized(stateLock) { _styleState.value = StyleState.Error(e.message ?: "Unknown error") }
            }
        }, "SD-Style").start()
    }

    fun releaseStyleTransfer() {
        nativeLib.nativeReleaseStyleTransfer()
        synchronized(stateLock) { _styleState.value = StyleState.Idle }
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    private fun bitmapToRgbBytes(bitmap: Bitmap): ByteArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val rgbBytes = ByteArray(width * height * 3)
        for (i in pixels.indices) {
            rgbBytes[i * 3] = ((pixels[i] shr 16) and 0xFF).toByte()
            rgbBytes[i * 3 + 1] = ((pixels[i] shr 8) and 0xFF).toByte()
            rgbBytes[i * 3 + 2] = (pixels[i] and 0xFF).toByte()
        }
        return rgbBytes
    }

    private fun createMaskBitmap(maskData: ByteArray, width: Int, height: Int): Bitmap {
        val bitmap = createBitmap(width, height)
        val pixels = IntArray(width * height)
        for (i in 0 until width * height) {
            if (i < maskData.size) {
                val v = maskData[i].toInt() and 0xFF
                pixels[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    private fun prepareRuntimeDirectory(config: DiffusionRuntimeConfig) {
        runtimeDir = File(context.filesDir, config.runtimeDir).apply {
            if (!exists()) mkdirs()
        }

        try {
            val markerFile = File(runtimeDir, ".extracted")

            if (markerFile.exists() && runtimeDir.listFiles()?.isNotEmpty() == true) {
                Log.i(TAG, "QNN libraries already exist, skipping extraction")
                runtimeDir.listFiles()?.forEach { file ->
                    file.setReadable(true, true)
                    file.setExecutable(true, true)
                }
                runtimeDir.setReadable(true, true)
                runtimeDir.setExecutable(true, true)
                updateSetupState(RuntimeSetupState.Complete)
                return
            }

            val bufferSize = getAdaptiveBufferSize(context)
            Log.i(TAG, "Adaptive buffer size: ${bufferSize / 1024}KB")

            val tarXzFile = File(context.cacheDir, "qnnlibs.tar.xz")

            // Phase 1: Get tar.xz — from pre-downloaded file or bundled assets
            val externalSource = config.tarXzSourcePath?.let { File(it) }
            if (externalSource != null && externalSource.exists()) {
                // Use pre-downloaded file (e.g. from HuggingFace)
                Log.i(TAG, "Using pre-downloaded QNN archive: ${externalSource.absolutePath}")
                updateSetupState(RuntimeSetupState.CopyingAsset(externalSource.length(), externalSource.length()))
                if (externalSource.absolutePath != tarXzFile.absolutePath) {
                    externalSource.copyTo(tarXzFile, overwrite = true)
                }
            } else {
                // Fall back to bundled assets
                val tarXzAssetPath = "${config.qnnLibsAssetPath}/qnnlibs.tar.xz"
                Log.i(TAG, "Copying QNN libraries from assets")

                val totalAssetBytes = try {
                    context.assets.openFd(tarXzAssetPath).use { it.length }
                } catch (_: Exception) {
                    -1L
                }

                context.assets.open(tarXzAssetPath).use { input ->
                    FileOutputStream(tarXzFile).use { output ->
                        input.copyToWithProgress(output, bufferSize, totalAssetBytes) { written, total ->
                            updateSetupState(RuntimeSetupState.CopyingAsset(written, total))
                        }
                    }
                }
            }
            Log.i(TAG, "Archive ready: ${tarXzFile.length()} bytes")

            // Phase 2: Extract tar.xz with progress
            Log.i(TAG, "Extracting QNN libraries from tar.xz")
            extractTarXzWithCommonsCompress(tarXzFile, runtimeDir, bufferSize) { filesExtracted, currentFile ->
                updateSetupState(RuntimeSetupState.Extracting(filesExtracted, currentFile))
            }

            markerFile.createNewFile()
            tarXzFile.delete()

            runtimeDir.listFiles()?.forEach { file ->
                file.setReadable(true, true)
                file.setExecutable(true, true)
            }
            runtimeDir.setReadable(true, true)
            runtimeDir.setExecutable(true, true)

            Log.i(TAG, "QNN libraries extracted: ${runtimeDir.list()?.joinToString()}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prepare QNN libraries", e)
            updateSetupState(RuntimeSetupState.Error("Failed to prepare QNN libraries: ${e.message}"))
            throw RuntimeException("Failed to prepare QNN libraries", e)
        }
    }

    private fun prepareSafetyChecker(config: DiffusionRuntimeConfig) {
        try {
            if (safetyCheckerFile.exists()) {
                Log.i(TAG, "Safety checker already exists, skipping")
                return
            }

            safetyCheckerFile.parentFile?.let { parent ->
                if (!parent.exists()) parent.mkdirs()
            }

            val externalSource = config.safetyCheckerSourcePath?.let { File(it) }
            if (externalSource != null && externalSource.exists()) {
                // Use pre-downloaded file
                Log.i(TAG, "Using pre-downloaded safety checker: ${externalSource.absolutePath}")
                externalSource.copyTo(safetyCheckerFile, overwrite = true)
            } else {
                // Fall back to bundled asset
                context.assets.open(config.safetyCheckerAssetPath).use { input ->
                    safetyCheckerFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            safetyCheckerFile.setReadable(true, true)
            Log.i(TAG, "Safety checker ready at: ${safetyCheckerFile.absolutePath}")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to prepare safety checker model", e)
            throw RuntimeException("Failed to prepare safety checker model", e)
        }
    }

    private fun findPatchFile(modelsDir: File, width: Int, height: Int): File? {
        val patchFile = if (width == height) {
            val squarePatch = File(modelsDir, "${width}.patch")
            if (squarePatch.exists()) squarePatch else File(modelsDir, "${width}x${height}.patch")
        } else {
            File(modelsDir, "${width}x${height}.patch")
        }

        return if (patchFile.exists()) {
            Log.i(TAG, "Using patch file: ${patchFile.name}")
            patchFile
        } else {
            Log.w(TAG, "Patch file not found: ${patchFile.absolutePath}")
            null
        }
    }

    // Cached pixel scratch reused across preview steps. At 768² this is
    // ~2.3 MB; the previous per-call alloc was producing ~65 MB of GC
    // churn over a 28-step gen, which was the actual trigger for the
    // Bitmap finalizer race (RenderThread still drawing while GC ran).
    @Volatile private var pixelScratch: IntArray? = null

    private fun createBitmapFromRgb(imageBytes: ByteArray, width: Int, height: Int): Bitmap {
        val pixelCount = width * height
        val expectedLen = pixelCount * 3
        require(imageBytes.size >= expectedLen) {
            "createBitmapFromRgb: expected $expectedLen RGB bytes for ${width}x${height}, got ${imageBytes.size}"
        }

        val pixels = pixelScratch?.takeIf { it.size == pixelCount }
            ?: IntArray(pixelCount).also { pixelScratch = it }

        for (i in 0 until pixelCount) {
            val idx = i * 3
            val r = imageBytes[idx].toInt() and 0xFF
            val g = imageBytes[idx + 1].toInt() and 0xFF
            val b = imageBytes[idx + 2].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        // Software ARGB_8888 bitmap. The previous HARDWARE copy was added
        // to dodge a 1024² RenderThread SIGSEGV, but the underlying cause
        // was the JNI callback cache crossing methodIDs across SDCallback
        // subclasses — fixed separately. HARDWARE bitmaps brought their
        // own GraphicBuffer/finalizer race observed in 768² previews:
        // worker-thread GPU upload + Bitmap finalizer releasing the GPU
        // texture while RenderThread was still drawing. Software bitmaps'
        // pixel buffer is reference-counted by RenderThread for the
        // duration of any pending draw, so finalization waits.
        val bm = createBitmap(width, height)
        bm.setPixels(pixels, 0, width, 0, 0, width, height)
        return bm
    }
}
