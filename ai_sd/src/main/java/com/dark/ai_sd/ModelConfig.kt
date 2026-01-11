package com.dark.ai_sd

import android.graphics.Bitmap

/**
 * Configuration for a Stable Diffusion model
 */
data class ModelConfig(
    val name: String,
    val modelDir: String,
    val textEmbeddingSize: Int = 768,
    val runOnCpu: Boolean = false,
    val useCpuClip: Boolean = false,
    val isPony: Boolean = false,
    val httpPort: Int = 8081
)

/**
 * Parameters for image generation
 */
data class GenerationParams(
    val prompt: String,
    val negativePrompt: String = "",
    val steps: Int = 28,
    val cfgScale: Float = 7f,
    val seed: Long? = null,
    val width: Int = 512,
    val height: Int = 512,
    val scheduler: String = "dpm",
    val useOpenCL: Boolean = false,
    
    // Img2Img specific
    val inputImage: String? = null,
    val mask: String? = null,
    val denoiseStrength: Float = 0.6f,
    
    // Process visualization
    val showDiffusionProcess: Boolean = false,
    val showDiffusionStride: Int = 1
)

/**
 * Sealed class representing the state of the backend service
 */
sealed class BackendState {
    object Idle : BackendState()
    object Starting : BackendState()
    object Running : BackendState()
    data class Error(val message: String) : BackendState()
}

/**
 * Sealed class representing the state of image generation
 */
sealed class GenerationState {
    object Idle : GenerationState()
    data class Progress(
        val progress: Float,
        val currentStep: Int = 0,
        val totalSteps: Int = 0,
        val intermediateImage: Bitmap? = null
    ) : GenerationState()
    data class Complete(
        val bitmap: Bitmap,
        val seed: Long?,
        val width: Int,
        val height: Int
    ) : GenerationState()
    data class Error(val message: String) : GenerationState()
}

/**
 * Result of a generation operation
 */
sealed class GenerationResult {
    data class Success(
        val bitmap: Bitmap,
        val seed: Long?,
        val width: Int,
        val height: Int
    ) : GenerationResult()
    data class Failure(val error: String) : GenerationResult()
}

/**
 * Configuration for the runtime environment
 */
data class RuntimeConfig(
    val runtimeDir: String,
    val executableName: String = "libstable_diffusion_core.so",
    val qnnLibsAssetPath: String = "qnnlibs",
    val safetyCheckerEnabled: Boolean = false,
    val safetyCheckerPath: String? = null
)