package com.dark.ai_sd

/**
 * Callback interface for Stable Diffusion generation progress and results.
 *
 * Called from native C++ via JNI on the generation thread.
 * Implementations should be thread-safe (e.g., updating MutableStateFlow).
 */
interface SDCallback {

    /**
     * Called after each diffusion step.
     *
     * @param step Current step number (1-based)
     * @param totalSteps Total number of steps
     */
    fun onProgress(step: Int, totalSteps: Int)

    /**
     * Called with an intermediate image during diffusion (if showDiffusionProcess is enabled).
     *
     * @param step Current step number
     * @param totalSteps Total number of steps
     * @param rgbData Raw RGB byte array of the intermediate image
     * @param width Image width in pixels
     * @param height Image height in pixels
     */
    fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int)

    /**
     * Called when generation completes successfully.
     *
     * @param rgbData Raw RGB byte array of the final image
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param seed The seed used for this generation
     * @param generationTimeMs Total generation time in milliseconds
     */
    fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int)

    /**
     * Called when an error occurs during generation.
     *
     * @param message Error description
     */
    fun onError(message: String)
}
