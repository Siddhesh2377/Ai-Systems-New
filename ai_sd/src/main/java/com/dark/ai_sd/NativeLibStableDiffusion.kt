package com.dark.ai_sd

class NativeLibStableDiffusion {

    /**
     * Initialize the Stable Diffusion native library
     * @param clipPath Path to CLIP model
     * @param unetPath Path to UNet model
     * @param vaeDecoderPath Path to VAE Decoder model
     * @param vaeEncoderPath Path to VAE Encoder model (can be null)
     * @param tokenizerPath Path to tokenizer file
     * @param backendPath Path to QNN backend library (can be null)
     * @param systemLibraryPath Path to QNN system library (can be null)
     * @param safetyCheckerPath Path to safety checker model (can be null)
     * @param patchPath Path to patch file (can be null)
     * @param port Server port number
     * @param useMnn Use MNN instead of QNN
     * @param useMnnClip Use MNN for CLIP model
     * @param ponyv55 Enable Pony v5.5 mode
     * @param useSafetyChecker Enable safety checker
     * @param upscalerMode Enable upscaler mode
     * @return 0 on success, non-zero on failure
     */
    external fun initialize(
        clipPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        vaeEncoderPath: String?,
        tokenizerPath: String,
        backendPath: String?,
        systemLibraryPath: String?,
        safetyCheckerPath: String?,
        patchPath: String?,
        port: Int,
        useMnn: Boolean,
        useMnnClip: Boolean,
        ponyv55: Boolean,
        useSafetyChecker: Boolean,
        upscalerMode: Boolean
    ): Int

    /**
     * Cleanup and release all native resources
     */
    external fun cleanup()

    /**
     * Start the HTTP server (blocking call - run in separate thread)
     */
    external fun startServer()

    companion object {
        init {
            System.loadLibrary("ai_sd")
        }
    }
}