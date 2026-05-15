package com.dark.ai_sd

/**
 * JNI wrapper for the native Stable Diffusion library.
 *
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

    // =========================================================================
    // Upscaler (Phase 5.1 — dead code revival)
    // =========================================================================

    /**
     * Load a 4x upscaler model.
     *
     * @param modelPath Path to upscaler model (.bin for QNN, .mnn for MNN)
     * @param useMnn Use MNN backend (CPU/GPU). If false, uses QNN (NPU).
     * @param useOpenCL Use OpenCL acceleration for MNN backend
     * @return true if loaded successfully
     */
    external fun nativeLoadUpscaler(modelPath: String, useMnn: Boolean, useOpenCL: Boolean): Boolean

    /**
     * Upscale an image 4x using the loaded upscaler model.
     * Result delivered via [SDCallback.onComplete].
     *
     * @param inputRgb Raw RGB bytes of input image
     * @param width Input image width
     * @param height Input image height
     * @param callback Callback for completion/error
     * @return true if upscaling started successfully
     */
    external fun nativeUpscaleImage(inputRgb: ByteArray, width: Int, height: Int, callback: SDCallback): Boolean

    /**
     * Release upscaler model resources.
     */
    external fun nativeReleaseUpscaler()

    /**
     * Get SoC hardware info from sysfs at native level.
     * Returns JSON: {"soc_id","machine","family","revision","htp_version","has_qnn_htp"}
     */
    external fun nativeGetSocInfo(): String

    /**
     * Enumerate every (width, height) the model in [modelDir] can run at.
     * Scans the directory for `.patch` files (`<N>.patch` for square or
     * `<W>x<H>.patch` for rectangular) and combines with [baseWidth] x
     * [baseHeight] (the resolution baked into the UNet `.bin` — usually
     * 512×512 for SD-1.5 `min` variants, often visible in the parent dir
     * name like `output_512`).
     *
     * @return flat int array `[w0, h0, w1, h1, ...]` sorted by total
     *         pixel count ascending. Empty if [modelDir] doesn't exist
     *         and [baseWidth]/[baseHeight] are non-positive.
     */
    external fun nativeGetSupportedResolutions(
        modelDir: String,
        baseWidth: Int,
        baseHeight: Int
    ): IntArray

    // ── Segmenter (Phase 5.3 — MobileSAM) ──

    external fun nativeLoadSegmenter(encoderPath: String, decoderPath: String, useOpenCL: Boolean): Boolean
    external fun nativeSegmenterEncodeImage(rgbBytes: ByteArray, width: Int, height: Int): Boolean
    external fun nativeSegmentAtPoint(x: Float, y: Float, callback: SDCallback): Boolean
    external fun nativeSegmentWithBox(x1: Float, y1: Float, x2: Float, y2: Float, callback: SDCallback): Boolean
    external fun nativeReleaseSegmenter()

    // ── LaMa Inpainter (Phase 5.4) ──

    external fun nativeLoadLamaInpainter(modelPath: String, useOpenCL: Boolean): Boolean
    external fun nativeLamaInpaint(rgbBytes: ByteArray, maskBytes: ByteArray, width: Int, height: Int, callback: SDCallback): Boolean
    external fun nativeReleaseLamaInpainter()

    // ── Depth Estimator (Phase 5.5) ──

    external fun nativeLoadDepthEstimator(modelPath: String, useOpenCL: Boolean): Boolean
    external fun nativeEstimateDepthColorized(rgbBytes: ByteArray, width: Int, height: Int, callback: SDCallback): Boolean
    external fun nativeReleaseDepthEstimator()

    // ── Style Transfer (Phase 5.6) ──

    external fun nativeLoadStyleTransfer(modelPath: String, useOpenCL: Boolean): Boolean
    external fun nativeStylize(contentRgb: ByteArray, contentW: Int, contentH: Int, styleRgb: ByteArray, styleW: Int, styleH: Int, strength: Float, callback: SDCallback): Boolean
    external fun nativeReleaseStyleTransfer()

    companion object {
        /** Whether the native library loaded successfully (false on unsupported ABIs like x86_64). */
        val isAvailable: Boolean

        init {
            isAvailable = try {
                // tn_security must be resolvable when libai_sd.so links it. The
                // dynamic linker normally handles transitive deps via DT_NEEDED,
                // but loading it explicitly first guarantees its JNI_OnLoad +
                // process-wide sink state are initialized before ai_sd issues
                // its first tn_sec_log call.
                runCatching { System.loadLibrary("tn_security") }
                System.loadLibrary("ai_sd")
                true
            } catch (_: UnsatisfiedLinkError) {
                false
            }
        }
    }
}
