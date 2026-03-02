package com.dark.ai_sd

import android.graphics.Bitmap

/**
 * Configuration for a Stable Diffusion model
 */
data class DiffusionModelConfig(
    val name: String,
    val modelDir: String,
    val textEmbeddingSize: Int = 768,
    val runOnCpu: Boolean = false,
    val useCpuClip: Boolean = false,
    val isPony: Boolean = false,
    val safetyMode: Boolean = false
)

/**
 * Parameters for image generation
 */
data class DiffusionGenerationParams(
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
sealed class DiffusionBackendState {
    object Idle : DiffusionBackendState()
    object Starting : DiffusionBackendState()
    object Running : DiffusionBackendState()
    data class Error(val message: String) : DiffusionBackendState()
}

/**
 * Sealed class representing the state of image generation
 */
sealed class DiffusionGenerationState {
    object Idle : DiffusionGenerationState()
    data class Progress(
        val progress: Float,
        val currentStep: Int = 0,
        val totalSteps: Int = 0,
        val intermediateImage: Bitmap? = null
    ) : DiffusionGenerationState()
    data class Complete(
        val bitmap: Bitmap,
        val seed: Long?,
        val width: Int,
        val height: Int
    ) : DiffusionGenerationState()
    data class Error(val message: String) : DiffusionGenerationState()
}

/**
 * Result of a generation operation
 */
sealed class DiffusionGenerationResult {
    data class Success(
        val bitmap: Bitmap,
        val seed: Long?,
        val width: Int,
        val height: Int
    ) : DiffusionGenerationResult()
    data class Failure(val error: String) : DiffusionGenerationResult()
}

/**
 * Sealed class representing the state of an upscale operation
 */
sealed class UpscaleState {
    object Idle : UpscaleState()
    object Processing : UpscaleState()
    data class Complete(
        val bitmap: Bitmap,
        val width: Int,
        val height: Int,
        val timeMs: Int
    ) : UpscaleState()
    data class Error(val message: String) : UpscaleState()
}

/**
 * Configuration for the runtime environment
 */
data class DiffusionRuntimeConfig(
    val runtimeDir: String,
    val qnnLibsAssetPath: String = "qnnlibs",
    val safetyCheckerEnabled: Boolean = true,
    val safetyCheckerPath: String = "assets/safety_checker.mnn"
)

/**
 * Sealed class representing runtime setup progress
 */
sealed class RuntimeSetupState {
    object Idle : RuntimeSetupState()
    data class CopyingAsset(val bytesWritten: Long, val totalBytes: Long) : RuntimeSetupState()
    data class Extracting(val filesExtracted: Int, val currentFile: String) : RuntimeSetupState()
    object CopyingSafetyChecker : RuntimeSetupState()
    object InitializingRuntime : RuntimeSetupState()
    object Complete : RuntimeSetupState()
    data class Error(val message: String) : RuntimeSetupState()
}
