package com.dark.ai_sd

import android.graphics.Bitmap
import android.content.Context
import kotlinx.coroutines.flow.StateFlow

/**
 * Unified facade for Stable Diffusion operations.
 *
 * Delegates to [DiffusionManager] which handles both model management
 * and image generation via JNI native calls.
 *
 * Usage:
 * ```
 * val sdManager = StableDiffusionManager.getInstance(context)
 *
 * // Initialize runtime (extracts QNN libs)
 * sdManager.initialize()
 *
 * // Load model
 * sdManager.loadModel(modelConfig)
 *
 * // Generate image
 * sdManager.generateImage(generationParams)
 *
 * // Observe states
 * lifecycleScope.launch {
 *     sdManager.diffusionGenerationState.collect { state ->
 *         when (state) {
 *             is DiffusionGenerationState.Progress -> updateProgress(state.progress)
 *             is DiffusionGenerationState.Complete -> showResult(state.bitmap)
 *             is DiffusionGenerationState.Error -> showError(state.message)
 *             else -> {}
 *         }
 *     }
 * }
 * ```
 */
class StableDiffusionManager private constructor(context: Context) {

    companion object {
        @Volatile
        private var instance: StableDiffusionManager? = null

        fun getInstance(context: Context): StableDiffusionManager {
            return instance ?: synchronized(this) {
                instance ?: StableDiffusionManager(context.applicationContext).also {
                    instance = it
                }
            }
        }
    }

    private val diffusionManager = DiffusionManager.getInstance(context)

    // Expose state flows
    val diffusionBackendState: StateFlow<DiffusionBackendState> = diffusionManager.diffusionBackendState
    val diffusionGenerationState: StateFlow<DiffusionGenerationState> = diffusionManager.diffusionGenerationState
    val isGenerating: StateFlow<Boolean> = diffusionManager.isGenerating
    val upscaleState: StateFlow<UpscaleState> = diffusionManager.upscaleState
    val runtimeSetupState: StateFlow<RuntimeSetupState> = diffusionManager.runtimeSetupState

    /**
     * Initialize the runtime environment.
     * Must be called before any other operations.
     */
    suspend fun initialize(config: DiffusionRuntimeConfig = DiffusionRuntimeConfig("runtime_libs")) {
        diffusionManager.setupRuntimeAsync(config)
    }

    /**
     * Load a model.
     * @return true if successful
     */
    fun loadModel(diffusionModelConfig: DiffusionModelConfig, width: Int = 512, height: Int = 512): Boolean {
        return diffusionManager.loadModel(diffusionModelConfig, width, height)
    }

    /**
     * Restart the backend with the current model.
     */
    fun restartBackend(): Boolean {
        return diffusionManager.restartBackend()
    }

    /**
     * Stop the backend and release model resources.
     */
    fun stopBackend() {
        diffusionManager.stopBackend()
    }

    /**
     * Generate an image asynchronously.
     * Monitor progress through [diffusionGenerationState] flow.
     */
    fun generateImage(params: DiffusionGenerationParams) {
        diffusionManager.generateImage(params)
    }

    /**
     * Generate an image synchronously (suspending function).
     */
    suspend fun generateImageSync(params: DiffusionGenerationParams): DiffusionGenerationResult {
        return diffusionManager.generateImageSync(params)
    }

    /**
     * Cancel ongoing generation.
     */
    fun cancelGeneration() {
        diffusionManager.cancelGeneration()
    }

    /**
     * Reset generation state to idle.
     */
    fun resetGenerationState() {
        diffusionManager.resetGenerationState()
    }

    /**
     * Get SoC hardware info from native C++ level.
     * Returns JSON: {"soc_id","machine","family","revision","htp_version","has_qnn_htp"}
     */
    fun getSocInfo(): String {
        return diffusionManager.getSocInfo()
    }

    /**
     * Get the currently loaded model.
     */
    fun getCurrentModel(): DiffusionModelConfig? {
        return diffusionManager.getCurrentModel()
    }

    /**
     * Check if the backend is running.
     */
    fun isBackendRunning(): Boolean {
        return diffusionManager.isBackendRunning()
    }

    /**
     * Cleanup all resources.
     */
    fun cleanup() {
        diffusionManager.cleanup()
    }

    // ========================================================================
    // Upscaler (Phase 5.1)
    // ========================================================================

    /**
     * Load a 4x upscaler model.
     */
    fun loadUpscaler(modelPath: String, useMnn: Boolean = true, useOpenCL: Boolean = false): Boolean {
        return diffusionManager.loadUpscaler(modelPath, useMnn, useOpenCL)
    }

    /**
     * Upscale a bitmap 4x. Monitor via [upscaleState] flow.
     */
    fun upscaleImage(inputBitmap: Bitmap) {
        diffusionManager.upscaleImage(inputBitmap)
    }

    /**
     * Release upscaler model resources.
     */
    fun releaseUpscaler() {
        diffusionManager.releaseUpscaler()
    }

    // ========================================================================
    // Segmenter (Phase 5.3 — MobileSAM)
    // ========================================================================

    val segmenterState: StateFlow<SegmenterState> = diffusionManager.segmenterState

    fun loadSegmenter(encoderPath: String, decoderPath: String, useOpenCL: Boolean = false): Boolean {
        return diffusionManager.loadSegmenter(encoderPath, decoderPath, useOpenCL)
    }

    fun segmenterEncodeImage(inputBitmap: Bitmap): Boolean {
        return diffusionManager.segmenterEncodeImage(inputBitmap)
    }

    fun segmentAtPoint(x: Float, y: Float) {
        diffusionManager.segmentAtPoint(x, y)
    }

    fun segmentWithBox(x1: Float, y1: Float, x2: Float, y2: Float) {
        diffusionManager.segmentWithBox(x1, y1, x2, y2)
    }

    fun releaseSegmenter() {
        diffusionManager.releaseSegmenter()
    }

    // ========================================================================
    // LaMa Inpainter (Phase 5.4)
    // ========================================================================

    val lamaState: StateFlow<LamaState> = diffusionManager.lamaState

    fun loadLamaInpainter(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return diffusionManager.loadLamaInpainter(modelPath, useOpenCL)
    }

    fun lamaInpaint(inputBitmap: Bitmap, maskBitmap: Bitmap) {
        diffusionManager.lamaInpaint(inputBitmap, maskBitmap)
    }

    fun releaseLamaInpainter() {
        diffusionManager.releaseLamaInpainter()
    }

    // ========================================================================
    // Depth Estimator (Phase 5.5)
    // ========================================================================

    val depthState: StateFlow<DepthState> = diffusionManager.depthState

    fun loadDepthEstimator(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return diffusionManager.loadDepthEstimator(modelPath, useOpenCL)
    }

    fun estimateDepth(inputBitmap: Bitmap) {
        diffusionManager.estimateDepth(inputBitmap)
    }

    fun releaseDepthEstimator() {
        diffusionManager.releaseDepthEstimator()
    }

    // ========================================================================
    // Style Transfer (Phase 5.6)
    // ========================================================================

    val styleState: StateFlow<StyleState> = diffusionManager.styleState

    fun loadStyleTransfer(modelPath: String, useOpenCL: Boolean = false): Boolean {
        return diffusionManager.loadStyleTransfer(modelPath, useOpenCL)
    }

    fun stylize(contentBitmap: Bitmap, styleBitmap: Bitmap, strength: Float = 1.0f) {
        diffusionManager.stylize(contentBitmap, styleBitmap, strength)
    }

    fun releaseStyleTransfer() {
        diffusionManager.releaseStyleTransfer()
    }
}
