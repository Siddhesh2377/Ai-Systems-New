package com.mp.ai_gguf

import com.mp.ai_gguf.models.StreamCallback
import com.mp.ai_gguf.models.EmbeddingCallback

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

    external fun nativeLoadModelFromFd(
        fd: Int,
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

    external fun nativeRelease(): Boolean
    external fun nativeSetChatTemplate(template: String)
    external fun nativeSetToolsJson(toolsJson: String)
    external fun nativeSetSystemPrompt(prompt: String)
    external fun nativeGetModelInfo(): String
    external fun nativeStopGeneration()

    /**
     * Set custom stop strings for generation.
     *
     * Stop strings are checked during token generation to detect when the
     * model's turn has ended. This is critical for small/quantized models
     * that emit turn markers (e.g. `<end_of_turn>`, `<|im_end|>`) as
     * regular text tokens instead of the special EOT token.
     *
     * By default, stop strings are auto-detected from the model's chat
     * template when the model is loaded. Use this function to override
     * with custom stop strings, or pass an empty array to disable.
     *
     * @param strings Array of stop strings. Generation stops when any of
     *                these is detected in the output. The stop string
     *                itself is not included in the output.
     */
    external fun nativeSetStopStrings(strings: Array<String>)
    external fun nativeClearMemory()
    external fun llamaPrintTimings()

    external fun nativeGenerateStream(
        prompt: String,
        maxTokens: Int,
        callback: StreamCallback
    ): Boolean

    /**
     * Multi-turn generation: processes a full conversation history and generates the next response.
     *
     * Used by the ToolCallManager orchestrator for multi-turn tool calling.
     * Each call clears the KV cache and re-encodes the full conversation.
     * This is intentional: prefill runs at 100-300 t/s on CPU so re-encoding
     * 500-1000 tokens costs ~2-5s, which is acceptable for interactive tool flows.
     *
     * @param messagesJson JSON array of {role, content} message objects
     * @param maxTokens Maximum tokens to generate this turn
     * @param callback StreamCallback for tokens, tool calls, metrics, done/error
     * @return true if generation completed (check callback for results)
     */
    external fun nativeGenerateStreamMultiTurn(
        messagesJson: String,
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

    // ========================================================================
    // EMBEDDING MODEL FUNCTIONS
    // ========================================================================

    /**
     * Load an embedding model from file path
     *
     * This loads the model in a separate thread/state from the main generation model,
     * so you can have both a generation model and embedding model loaded simultaneously.
     *
     * @param path Path to the embedding model file (must be in app directory)
     * @param threads Number of threads (0 = auto-detect physical cores)
     * @param contextSize Context size for the embedding model (512 typical for embeddings)
     * @return true if model loaded successfully
     */
    external fun nativeLoadEmbeddingModel(
        path: String,
        threads: Int,
        contextSize: Int
    ): Boolean

    /**
     * Encode text into embeddings
     *
     * @param text The text to encode
     * @param normalize Whether to L2-normalize the output embeddings (recommended for similarity)
     * @param callback Callback for progress and results
     * @return true if encoding started successfully
     */
    external fun nativeEncodeText(
        text: String,
        normalize: Boolean,
        callback: EmbeddingCallback
    ): Boolean

    /**
     * Release the embedding model and free resources
     *
     * @return true if released successfully
     */
    external fun nativeReleaseEmbeddingModel(): Boolean

    /**
     * Get embedding model info (architecture, dimensions, etc.)
     *
     * @return JSON string with model info, or empty object if no model loaded
     */
    external fun nativeGetEmbeddingModelInfo(): String

    /**
     * Load an embedding model from file descriptor (for SAF compatibility)
     *
     * @param fd File descriptor from ContentResolver
     * @param threads Number of threads (0 = auto-detect physical cores)
     * @param ctxSize Context size for embeddings
     * @return true if model loaded successfully
     */
    external fun nativeLoadEmbeddingModelFromFd(
        fd: Int,
        threads: Int,
        ctxSize: Int
    ): Boolean

    // ========================================================================
    // TOOL CALLING SDK FUNCTIONS
    // ========================================================================

    /**
     * Get the architecture of the loaded model
     *
     * @return Model architecture (e.g., "qwen2", "llama", etc.) or empty string if no model
     */
    external fun nativeGetModelArchitecture(): String

    /**
     * Check if the currently loaded model supports tool calling.
     *
     * Returns true for any model with a chat template. Grammar enforcement
     * ensures valid JSON output regardless of model architecture.
     *
     * @return true if model has a chat template and can support tool calling
     */
    external fun nativeIsToolCallingSupported(): Boolean

    /**
     * Enable tool calling mode for the current model.
     *
     * Sets the tools JSON and initializes the grammar sampler.
     * System prompt and chat template should be set separately via
     * [nativeSetSystemPrompt] and [nativeSetChatTemplate].
     *
     * @param toolsJson OpenAI-compatible tools JSON array
     * @return true if tool calling was enabled successfully
     */
    external fun nativeEnableToolCalling(toolsJson: String): Boolean

    /**
     * Disable tool calling and revert to default model behavior
     *
     * This clears:
     * - Tools JSON
     * - System prompt
     * - Chat template override
     * - Tool calling state
     */
    external fun nativeDisableToolCalling()

    /**
     * Check if tool calling is currently enabled
     *
     * @return true if tool calling is enabled
     */
    external fun nativeIsToolCallingEnabled(): Boolean

    /**
     * Set the grammar enforcement mode for tool calling.
     *
     * @param mode 0 = STRICT (grammar active from first token, forces JSON output),
     *             1 = LAZY (grammar activates on "{" trigger, model chooses tool vs text)
     */
    external fun nativeSetGrammarMode(mode: Int)

    /**
     * Enable/disable parameter-aware typed grammar.
     *
     * When enabled, the grammar enforces exact parameter names, types, and
     * enum values per tool. When disabled, uses a generic JSON object grammar.
     *
     * @param enabled true to use typed grammar, false for generic
     */
    external fun nativeSetTypedGrammar(enabled: Boolean)

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