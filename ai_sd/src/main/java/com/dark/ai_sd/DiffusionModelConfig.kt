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
 * Sealed class representing the state of a segmentation operation
 */
sealed class SegmenterState {
    object Idle : SegmenterState()
    object Processing : SegmenterState()
    data class Complete(
        val mask: Bitmap,
        val score: Float,
        val width: Int,
        val height: Int,
        val timeMs: Int
    ) : SegmenterState()
    data class Error(val message: String) : SegmenterState()
}

/**
 * Sealed class representing the state of a LaMa inpainting operation
 */
sealed class LamaState {
    object Idle : LamaState()
    object Processing : LamaState()
    data class Complete(
        val bitmap: Bitmap,
        val timeMs: Int
    ) : LamaState()
    data class Error(val message: String) : LamaState()
}

/**
 * Sealed class representing the state of a depth estimation operation
 */
sealed class DepthState {
    object Idle : DepthState()
    object Processing : DepthState()
    data class Complete(
        val depthMap: Bitmap,
        val timeMs: Int
    ) : DepthState()
    data class Error(val message: String) : DepthState()
}

/**
 * Sealed class representing the state of a style transfer operation
 */
sealed class StyleState {
    object Idle : StyleState()
    object Processing : StyleState()
    data class Complete(
        val bitmap: Bitmap,
        val timeMs: Int
    ) : StyleState()
    data class Error(val message: String) : StyleState()
}

/**
 * Configuration for the runtime environment.
 *
 * When [tarXzSourcePath] or [safetyCheckerSourcePath] are set, the manager
 * copies from those local files instead of extracting from bundled assets.
 * This supports downloading files externally (e.g. from HuggingFace) before
 * initializing the runtime.
 */
data class DiffusionRuntimeConfig(
    val runtimeDir: String,
    val qnnLibsAssetPath: String = "qnnlibs",
    val safetyCheckerEnabled: Boolean = true,
    val safetyCheckerAssetPath: String = "safety_checker.mnn",
    val tarXzSourcePath: String? = null,
    val safetyCheckerSourcePath: String? = null
)

/**
 * Sealed class representing runtime setup progress
 */
sealed class RuntimeSetupState {
    object Idle : RuntimeSetupState()
    data class Downloading(val bytesDownloaded: Long, val totalBytes: Long, val fileName: String) : RuntimeSetupState()
    data class CopyingAsset(val bytesWritten: Long, val totalBytes: Long) : RuntimeSetupState()
    data class Extracting(val filesExtracted: Int, val currentFile: String) : RuntimeSetupState()
    object CopyingSafetyChecker : RuntimeSetupState()
    object InitializingRuntime : RuntimeSetupState()
    object Complete : RuntimeSetupState()
    data class Error(val message: String) : RuntimeSetupState()
}
