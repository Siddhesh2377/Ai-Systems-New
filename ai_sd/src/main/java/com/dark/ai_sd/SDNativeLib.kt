package com.dark.ai_sd

/**
 * JNI wrapper for the native Stable Diffusion library.
 *
 * Follows the same pattern as ai_gguf/GGUFNativeLib.kt:
 * - System.loadLibrary() in companion init block
 * - External fun declarations matching C++ JNI functions
 * - Direct native calls (no HTTP, no subprocess)
 */
class SDNativeLib {

    /**
     * Initialize QNN runtime environment.
     * Sets ADSP_LIBRARY_PATH for Hexagon DSP library resolution.
     * Must be called once before [nativeLoadModel].
     *
     * @param qnnLibDir Directory containing extracted QNN .so libraries
     * @return true on success
     */
    external fun nativeInitRuntime(qnnLibDir: String): Boolean

    /**
     * Load Stable Diffusion model components.
     *
     * @param clipPath Path to CLIP model (.bin for QNN, .mnn for CPU)
     * @param unetPath Path to UNet model
     * @param vaeDecoderPath Path to VAE decoder model
     * @param vaeEncoderPath Path to VAE encoder model (empty string if not available)
     * @param tokenizerPath Path to tokenizer.json
     * @param safetyCheckerPath Path to safety checker .mnn (empty string to disable)
     * @param patchPath Path to resolution patch file (empty string if none)
     * @param modelDir Model directory root path
     * @param qnnBackendPath Path to libQnnHtp.so (empty for CPU mode)
     * @param qnnSystemLibPath Path to libQnnSystem.so (empty for CPU mode)
     * @param textEmbeddingSize CLIP embedding dimension (768 or 1024)
     * @param runOnCpu Use CPU-only inference via MNN
     * @param useCpuClip Use MNN for CLIP even when UNet runs on QNN
     * @param isPony Pony v5.5 model flag (uses v_prediction)
     * @param useSafetyChecker Enable NSFW safety filter
     * @return true on success
     */
    external fun nativeLoadModel(
        clipPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        vaeEncoderPath: String,
        tokenizerPath: String,
        safetyCheckerPath: String,
        patchPath: String,
        modelDir: String,
        qnnBackendPath: String,
        qnnSystemLibPath: String,
        textEmbeddingSize: Int,
        runOnCpu: Boolean,
        useCpuClip: Boolean,
        isPony: Boolean,
        useSafetyChecker: Boolean
    ): Boolean

    /**
     * Run image generation.
     *
     * This is a blocking call that runs on the calling thread.
     * Progress and results are delivered via the [callback].
     *
     * @param prompt Positive prompt text
     * @param negativePrompt Negative prompt text
     * @param steps Number of diffusion steps
     * @param cfgScale Classifier-free guidance scale
     * @param seed Random seed (0 for auto-generated)
     * @param width Output image width (must be divisible by 8)
     * @param height Output image height (must be divisible by 8)
     * @param scheduler Scheduler type: "dpm" or "euler_a"
     * @param useOpenCL Use OpenCL acceleration for MNN
     * @param inputImage Raw RGB bytes for img2img input (null for txt2img)
     * @param mask Raw RGB mask bytes for inpainting (null if no mask)
     * @param denoiseStrength Denoising strength for img2img (0.0-1.0)
     * @param showProcess Enable intermediate image callbacks
     * @param showStride Callback every N-th step for intermediate images
     * @param callback [SDCallback] for progress, completion, and error notifications
     * @return true if generation completed successfully
     */
    external fun nativeGenerate(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        cfgScale: Float,
        seed: Long,
        width: Int,
        height: Int,
        scheduler: String,
        useOpenCL: Boolean,
        inputImage: ByteArray?,
        mask: ByteArray?,
        denoiseStrength: Float,
        showProcess: Boolean,
        showStride: Int,
        callback: SDCallback
    ): Boolean

    /**
     * Stop ongoing generation.
     * Sets an atomic flag checked each diffusion step.
     * The current step will complete before stopping.
     */
    external fun nativeStopGeneration()

    /**
     * Release all model resources and free memory.
     *
     * @return true on success
     */
    external fun nativeRelease(): Boolean

    /**
     * Get information about the loaded model.
     *
     * @return JSON string with model info, or "{}" if no model loaded
     */
    external fun nativeGetModelInfo(): String

    /**
     * Apply a LoRA to the current model.
     *
     * @param loraPath Path to LoRA .safetensors file
     * @param weight LoRA strength multiplier (typically 0.5-1.0)
     * @return true on success
     */
    external fun nativeApplyLora(loraPath: String, weight: Float): Boolean

    /**
     * Clear all applied LoRA weights, reverting to base model.
     */
    external fun nativeClearLora()

    companion object {
        init {
            System.loadLibrary("ai_sd")
        }
    }
}
