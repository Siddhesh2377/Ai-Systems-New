package com.mp.ai_gguf

import com.mp.ai_gguf.models.StreamCallback

/**
 * Native library interface for GGUF model inference
 *
 * Optimizations for low-end devices:
 * - Configurable batch sizes (lower = less memory)
 * - Thread count auto-detection
 * - Memory-mapped model loading
 * - Grammar caching for tool calls
 */
class GGUFNativeLib {

    external fun nativeRelease(): Boolean
    external fun nativeSetChatTemplate(template: String)
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeGetModelInfo(): String
    external fun nativeStopGeneration()
    external fun nativeClearMemory()
    external fun llamaPrintTimings()

    external fun nativeGenerateStream(
        prompt: String,
        maxTokens: Int,
        callback: StreamCallback
    ): Boolean

    /**
     * Load a GGUF model with full configuration
     *
     * @param path Path to the GGUF model file
     * @param threads Number of threads (0 = auto-detect physical cores)
     * @param ctxSize Context window size (2048 recommended for low-end)
     * @param temp Temperature (0.0 = greedy, 0.7 = balanced, 1.0+ = creative)
     * @param topK Top-K filtering (40 typical)
     * @param topP Nucleus sampling threshold (0.9 typical)
     * @param minP Minimum probability filter (0.05 typical)
     * @param mirostat Mirostat mode (0 = off, 1 or 2 = enabled)
     * @param mirostatTau Target entropy for mirostat
     * @param mirostatEta Learning rate for mirostat
     * @param seed Random seed (-1 = random)
     */
    external fun nativeLoadModel(
        path: String,
        threads: Int,
        ctxSize: Int,
        temp: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        mirostat: Int,
        mirostatTau: Float,
        mirostatEta: Float,
        seed: Int
    ): Boolean

    companion object {
        init {
            System.loadLibrary("ai_gguf")
        }

        /**
         * Recommended settings for low-end devices (< 4GB RAM)
         */
        object LowEndDefaults {
            const val CONTEXT_SIZE = 1024
            const val BATCH_SIZE = 256
            const val THREADS = 0  // Auto-detect
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        /**
         * Recommended settings for mid-range devices (4-8GB RAM)
         */
        object MidRangeDefaults {
            const val CONTEXT_SIZE = 2048
            const val BATCH_SIZE = 512
            const val THREADS = 0  // Auto-detect
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        /**
         * Recommended settings for high-end devices (> 8GB RAM)
         */
        object HighEndDefaults {
            const val CONTEXT_SIZE = 4096
            const val BATCH_SIZE = 512
            const val THREADS = 0  // Auto-detect
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        /**
         * Get recommended context size based on available memory
         * @param availableMemoryMB Available RAM in MB
         * @param modelSizeMB Approximate model size in MB
         * @return Recommended context size
         */
        fun recommendedContextSize(availableMemoryMB: Int, modelSizeMB: Int): Int {
            val freeAfterModel = availableMemoryMB - modelSizeMB
            return when {
                freeAfterModel < 1024 -> 512   // Very constrained
                freeAfterModel < 2048 -> 1024  // Low-end
                freeAfterModel < 4096 -> 2048  // Mid-range
                else -> 4096                    // High-end
            }
        }
    }
}