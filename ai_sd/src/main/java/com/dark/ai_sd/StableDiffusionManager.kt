package com.dark.ai_sd

import android.content.Context
import kotlinx.coroutines.flow.StateFlow

/**
 * Unified facade for Stable Diffusion operations
 * Combines DiffusionManager (backend/model management) and GenerationManager (image generation)
 * 
 * Usage:
 * ```
 * val sdManager = StableDiffusionManager.getInstance(context)
 * 
 * // Initialize
 * sdManager.initialize()
 * 
 * // Load model
 * val model = ModelConfig(...)
 * sdManager.loadModel(model)
 * 
 * // Generate image
 * val params = GenerationParams(
 *     prompt = "a beautiful landscape",
 *     steps = 28
 * )
 * sdManager.generateImage(params)
 * 
 * // Observe states
 * lifecycleScope.launch {
 *     sdManager.generationState.collect { state ->
 *         when (state) {
 *             is GenerationState.Progress -> updateProgress(state.progress)
 *             is GenerationState.Complete -> showResult(state.bitmap)
 *             is GenerationState.Error -> showError(state.message)
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
    private val generationManager = GenerationManager.getInstance(context)

    // Expose state flows
    val backendState: StateFlow<BackendState> = diffusionManager.backendState
    val generationState: StateFlow<GenerationState> = generationManager.generationState
    val isGenerating: StateFlow<Boolean> = generationManager.isGenerating

    /**
     * Initialize the runtime environment
     * Must be called before any other operations
     */
    fun initialize(config: RuntimeConfig = RuntimeConfig("runtime_libs")) {
        diffusionManager.setupRuntime(config)
    }

    /**
     * Load a model and start the backend server
     * @return true if successful, false otherwise
     */
    fun loadModel(modelConfig: ModelConfig, width: Int = 512, height: Int = 512): Boolean {
        return diffusionManager.loadModel(modelConfig, width, height)
    }

    /**
     * Restart the backend with the current model
     * @return true if successful, false otherwise
     */
    fun restartBackend(): Boolean {
        return diffusionManager.restartBackend()
    }

    /**
     * Stop the backend server
     */
    fun stopBackend() {
        diffusionManager.stopBackend()
    }

    /**
     * Generate an image asynchronously
     * Monitor progress through generationState flow
     */
    fun generateImage(params: GenerationParams) {
        generationManager.generateImage(params)
    }

    /**
     * Generate an image synchronously (suspending function)
     * @return GenerationResult containing the bitmap or error
     */
    suspend fun generateImageSync(params: GenerationParams): GenerationResult {
        return generationManager.generateImageSync(params)
    }

    /**
     * Cancel ongoing generation
     */
    fun cancelGeneration() {
        generationManager.cancelGeneration()
    }

    /**
     * Reset generation state to idle
     */
    fun resetGenerationState() {
        generationManager.resetState()
    }

    /**
     * Get the currently loaded model
     */
    fun getCurrentModel(): ModelConfig? {
        return diffusionManager.getCurrentModel()
    }

    /**
     * Check if the backend is running
     */
    fun isBackendRunning(): Boolean {
        return diffusionManager.isBackendRunning()
    }

    /**
     * Cleanup all resources
     * Should be called when the app is being destroyed
     */
    fun cleanup() {
        generationManager.cleanup()
        diffusionManager.cleanup()
    }
}