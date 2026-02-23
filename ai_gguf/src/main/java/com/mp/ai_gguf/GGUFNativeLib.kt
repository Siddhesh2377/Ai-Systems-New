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
        seed: Int,
        flashAttn: Boolean = true,
        cacheTypeK: Int = 9,  // GGML_TYPE_Q8_0
        cacheTypeV: Int = 9   // GGML_TYPE_Q8_0
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
     * @param flashAttn Enable flash attention (reduces memory bandwidth, major CPU speedup)
     * @param cacheTypeK GGML type for KV cache keys (9=Q8_0, 2=Q4_0, 1=F16, 0=F32)
     * @param cacheTypeV GGML type for KV cache values (9=Q8_0, 2=Q4_0, 1=F16, 0=F32)
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
        seed: Int,
        flashAttn: Boolean = true,
        cacheTypeK: Int = 9,  // GGML_TYPE_Q8_0
        cacheTypeV: Int = 9   // GGML_TYPE_Q8_0
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

    // ========================================================================
    // PERSONA ENGINE: Dynamic Sampling + Logit Bias + Control Vectors
    // ========================================================================

    /**
     * Update sampler parameters at runtime without reloading the model.
     *
     * Accepts a JSON object with any subset of sampler params.
     * Missing keys keep the current values (merge semantics).
     *
     * Supported keys:
     * - Base: topK, topP, temperature, minP, mirostat, mirostatTau, mirostatEta, seed
     * - Penalties: repeatPenalty, frequencyPenalty, presencePenalty, penaltyLastN
     * - DRY: dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN
     * - XTC: xtcProbability, xtcThreshold
     *
     * @param paramsJson JSON object with sampler parameters
     * @return true if sampler rebuilt successfully
     */
    external fun nativeUpdateSamplerParams(paramsJson: String): Boolean

    /**
     * Set per-token logit biases to suppress or boost specific tokens.
     *
     * Use to suppress AI-speak words like "certainly", "delve", "Moreover".
     * bias=-100 = hard suppression, -5 = soft discouragement, +5 = boost.
     * Pass "[]" to clear all biases.
     *
     * @param biasJson JSON array: [{"token": "certainly", "bias": -5.0}, ...]
     * @return true if biases applied successfully
     */
    external fun nativeSetLogitBias(biasJson: String): Boolean

    /**
     * Load one or more control vectors (steering vectors) from GGUF files.
     *
     * Control vectors shift the model's personality along specific axes
     * (warmth, energy, formality, etc.) without fine-tuning.
     * Multiple vectors are accumulated (summed with scaling) before applying.
     * Pass "[]" to clear all control vectors.
     *
     * @param vectorsJson JSON array: [{"path": "/path/warmth.gguf", "strength": 0.8}, ...]
     * @return true if vectors applied successfully
     */
    external fun nativeLoadControlVectors(vectorsJson: String): Boolean

    /**
     * Clear all control vectors, returning model to baseline behavior.
     *
     * @return true if cleared successfully
     */
    external fun nativeClearControlVector(): Boolean

    // ========================================================================
    // RUNTIME BEHAVIOR INTERVENTION — Personality Engine
    // ========================================================================

    /**
     * Compute personality control vectors from contrastive text prompts at runtime.
     *
     * Creates a lightweight probe context (shares model weights ~3MB), runs forward
     * passes through positive/negative prompts, extracts per-layer hidden states,
     * computes direction vectors, and applies them. Cached per model hash + axis.
     *
     * First call: ~24s (0.5B) to ~72s (3B). Subsequent calls: instant (cached).
     *
     * @param promptsJson JSON: {"warmth": {"positive": ["I care!"], "negative": ["Noted."]}, ...}
     * @param axisStrengthsJson JSON: {"warmth": 0.7, "energy": -0.3, ...}
     * @param cacheDir Directory to store cached direction vectors
     * @return true on success
     */
    external fun nativeComputePersonalityVectors(
        promptsJson: String,
        axisStrengthsJson: String,
        cacheDir: String
    ): Boolean

    /**
     * Apply emotion-conditioned dimensional gating to control vectors.
     *
     * Modulates WHICH dimensions of the control vector offset are active based on
     * the current emotional state. Different emotions activate different embedding dimensions.
     *
     * When emotionStrengths is empty, applies plain control vectors (no gating).
     * When provided, applies: result = sigmoid(scale * gate_signal) * base_direction
     *
     * @param personaStrengthsJson JSON: {"warmth": 0.7, ...} — persona baseline strengths
     * @param emotionStrengthsJson JSON: {"warmth": 0.5, ...} — current emotional state (or "{}" to disable gating)
     * @param cacheDir             Cache directory with per-axis direction vectors
     * @param gateScale            Sigmoid sharpness (3.0 = moderate, 5.0 = sharp). 0 = no gating.
     * @return true on success
     */
    external fun nativeApplyEmotionGatedVectors(
        personaStrengthsJson: String,
        emotionStrengthsJson: String,
        cacheDir: String,
        gateScale: Float
    ): Boolean

    /**
     * Set per-head attention output scales (head rescaling).
     * scale=1.0 (default), 0.0 (ablate), 2.0 (amplify), -1.0 (reverse).
     * Disables flash attention for affected layers.
     *
     * @param scalesJson JSON array: [{"layer": 0, "head": 0, "scale": 1.5}, ...]
     *                   Pass "[]" to clear all scales.
     */
    external fun nativeSetHeadScales(scalesJson: String): Boolean

    /**
     * Reset all head scales to default (1.0).
     */
    external fun nativeResetHeadScales(): Boolean

    /**
     * Set per-head attention temperatures using a layer-range profile.
     * T<1.0 = sharper (focused), T>1.0 = flatter (broader attention).
     * Disables flash attention for affected layers.
     *
     * @param profileJson JSON: {"early": 1.3, "mid": 1.0, "late": 0.8}
     *                    "early" = layers 0-30%, "mid" = 30-60%, "late" = 60-100%
     *                    Pass "{}" to reset.
     */
    external fun nativeSetAttentionTemperatureProfile(profileJson: String): Boolean

    /**
     * Reset all attention temperatures to default (1.0).
     */
    external fun nativeResetAttentionTemperatures(): Boolean

    /**
     * Set per-layer scalar gates on attention and FFN residual outputs.
     * Values in [0, 2]: 0 = skip layer, 1 = default (no change), 2 = amplify.
     *
     * @param gatesJson JSON: {"attn": [1.0, 0.8, ...], "ffn": [1.0, 0.9, ...]}
     *                  Either key may be omitted to leave unchanged.
     *                  Pass "{}" to reset all gates.
     */
    external fun nativeSetResidualGates(gatesJson: String): Boolean

    /**
     * Reset all residual gates to default (1.0, no scaling).
     */
    external fun nativeResetResidualGates(): Boolean

    // ========================================================================
    // SPECULATIVE DECODING (Self-Speculative Early Exit)
    // ========================================================================

    /**
     * Enable self-speculative decoding for faster generation.
     *
     * Uses the model's own early layers as a draft model. Each generation step:
     *   1. Drafts [numDraft] tokens through the first [exitLayer] transformer blocks
     *   2. Verifies all draft tokens in one batch through the full model
     *   3. Accepts matching tokens via greedy argmax comparison
     *
     * Speedup is ~1.3-2.0x depending on model depth and acceptance rate.
     * Output is greedy (always argmax from full model), which may differ
     * from sampled output when temperature > 0.
     *
     * @param exitLayer Number of transformer layers for draft model (e.g., 6 for a 24-layer model).
     *                  Lower = faster drafts but lower acceptance rate. Recommended: 25% of total layers.
     * @param numDraft  Number of draft tokens per speculative iteration (4-8 typical).
     *                  Higher = more potential speedup but more wasted work on rejection.
     * @return true if speculative decoding was enabled
     */
    external fun nativeEnableSpeculativeDecode(exitLayer: Int, numDraft: Int): Boolean

    /**
     * Disable speculative decoding and return to standard autoregressive generation.
     */
    external fun nativeDisableSpeculativeDecode(): Boolean

    // ========================================================================
    // EMOTION STATE MACHINE: Residual Stream Probing
    // ========================================================================

    /**
     * Enable/disable layer activation capture for emotion probing.
     *
     * When enabled, each decode stores the last token's per-layer activations.
     * Overhead: ~86KB/token for a 24-layer, 896-dim model (negligible).
     * Must be enabled before generation for [nativeProbeEmotionAxes] to work.
     *
     * @param enabled true to enable capture, false to disable
     */
    external fun nativeSetCaptureEnabled(enabled: Boolean): Boolean

    /**
     * Probe the model's internal emotional state via residual stream analysis.
     *
     * Computes dot products of captured layer activations with cached direction
     * vectors at 3 strategic layers (40%, 60%, 80% depth). The result indicates
     * how strongly the model is expressing each personality axis.
     *
     * Scores are tanh-squashed to [-1, +1]:
     *   Positive = model is expressing this axis
     *   Negative = model is suppressing this axis
     *   Near zero = neutral
     *
     * Requires [nativeSetCaptureEnabled] to be true and at least one decode
     * to have occurred. Call after each generation turn.
     *
     * @param cacheDir Directory containing cached direction vectors (same as personality cache)
     * @return JSON: {"warmth": 0.35, "energy": -0.12, ...} or {"error": "reason"}
     */
    external fun nativeProbeEmotionAxes(cacheDir: String): String

    /**
     * Set attention score bias for a token position range.
     * Boosts attention to persona/system prompt tokens.
     * bias > 0 boosts (exp(bias) multiplier), bias < 0 suppresses.
     *
     * @param startPos Start token position (inclusive)
     * @param endPos End token position (exclusive)
     * @param bias Log-space bias value (e.g., +2.0 = 7.4x boost)
     * @param layerStart First layer (inclusive)
     * @param layerEnd Last layer (exclusive, -1 = all)
     */
    external fun nativeSetAttentionBias(
        startPos: Int, endPos: Int, bias: Float,
        layerStart: Int, layerEnd: Int
    ): Boolean

    /**
     * Clear all attention biases.
     */
    external fun nativeClearAttentionBias(): Boolean

    // ========================================================================
    // PART G: LAYERNORM AFFINE SHIFT
    // ========================================================================

    /**
     * Compute and apply LayerNorm affine shift offsets from cached direction vectors.
     * Reads the same cached vectors as Part A/D, scales them down for normalization use.
     * Cheapest personality modification — zero flash-attention penalty.
     *
     * @param axisStrengthsJson JSON: {"warmth": 0.7, "energy": -0.3, ...}
     * @param cacheDir Same cache dir used by [nativeComputePersonalityVectors]
     * @param scaleFactor How much to scale direction vectors for norm use (0.01-0.05 typical)
     * @return JSON: {"success": true, "n_layers_set": 24, "axes_loaded": 3}
     */
    external fun nativeSetNormOffsets(axisStrengthsJson: String, cacheDir: String, scaleFactor: Float): String

    /**
     * Reset all LayerNorm offsets (remove all norm shifts).
     */
    external fun nativeResetNormOffsets(): Boolean

    // ========================================================================
    // INTERVENTION STATE PERSISTENCE
    // ========================================================================

    /**
     * Save all learnable intervention state (KAN coefficients, sparse masks) to a binary file.
     * Call after generation turns or on app pause to persist P7 learning progress.
     * File is compact (~1-2MB) and versioned for backward compatibility.
     *
     * @param path Full path to save the state file
     */
    external fun nativeSaveInterventionState(path: String): Boolean

    /**
     * Load learnable intervention state from a previously saved file.
     * Call after model load and before applyPersonality() to restore learned parameters.
     * Returns false if file doesn't exist (first run) or on error.
     *
     * @param path Full path to the state file
     */
    external fun nativeLoadInterventionState(path: String): Boolean

    // ========================================================================
    // PART P4: HYPERNETWORK FFN LoRA
    // ========================================================================

    /**
     * Initialize hypernetwork with rank-4 LoRA for middle FFN layers.
     * A matrices init with small random values, B matrices init to zeros (net zero effect).
     *
     * @param rank LoRA rank (typically 4)
     * @param layerStart First target layer (-1 = auto: 37% of model depth)
     * @param layerEnd One past last target layer (-1 = auto: 70%)
     * @param strength Global strength multiplier (0 = disabled)
     */
    external fun nativeInitHypernetwork(rank: Int, layerStart: Int, layerEnd: Int, strength: Float): Boolean

    /**
     * Set LoRA A and/or B matrices for a specific target layer index.
     * Target index is relative to first target layer (0-based, NOT model layer index).
     *
     * @param targetIdx Target layer index (0 = first target layer)
     * @param loraA FloatArray of rank*n_embd values, or null to skip
     * @param loraB FloatArray of n_ff*rank values, or null to skip
     */
    external fun nativeSetHypernetworkLora(targetIdx: Int, loraA: FloatArray?, loraB: FloatArray?): Boolean

    /** Set hypernetwork global strength multiplier. */
    external fun nativeSetHypernetworkStrength(strength: Float)

    /** Reset hypernetwork (clear all LoRA matrices, disable). */
    external fun nativeResetHypernetwork(): Boolean

    /**
     * Initialize hypernetwork from cached control vector direction vectors.
     * Uses personality axis directions to warm-start LoRA A matrices — much faster
     * convergence than random initialization when combined with P7 learning.
     *
     * @param strengthsJson JSON: {"warmth": 0.7, "energy": 0.3, ...}
     * @param cacheDir Directory containing cached direction vectors
     * @param rank LoRA rank (typically 4)
     * @param strength Global strength multiplier
     * @return JSON: {"success": true, "n_target_layers": 8}
     */
    external fun nativeInitHypernetworkFromDirections(
        strengthsJson: String,
        cacheDir: String,
        rank: Int,
        strength: Float
    ): String

    // ========================================================================
    // PART P5: DYNAMIC SPARSE MASKS
    // ========================================================================

    /**
     * Initialize sparse masks for all layers with all neurons active.
     * If keepRatio < 1.0, randomly disables (1-keepRatio) fraction of neurons.
     *
     * @param keepRatio Fraction of neurons to keep active (0.0-1.0, 1.0 = all active)
     */
    external fun nativeInitSparseMasks(keepRatio: Float): Boolean

    /**
     * Set sparse mask for a specific layer.
     * @param layer Target layer index
     * @param mask FloatArray of n_ff values in [0,1]: 0=disabled, 1=active
     */
    external fun nativeSetSparseMask(layer: Int, mask: FloatArray): Boolean

    /** Reset all sparse masks (all neurons fully active). */
    external fun nativeResetSparseMasks(): Boolean

    /**
     * Update sparse masks based on activation magnitude analysis.
     * Runs a probe forward pass on sample text, analyzes activations,
     * and updates masks with momentum smoothing.
     *
     * @param text Sample text to analyze activations on
     * @param keepRatio Fraction of neurons to keep (e.g., 0.9 = keep 90%)
     * @param momentum Smoothing factor (0.9 = gradual transition)
     * @return JSON: {"success": true, "avg_sparsity": 0.15}
     */
    external fun nativeUpdateSparseMasks(text: String, keepRatio: Float, momentum: Float): String

    // ========================================================================
    // PART P6: KAN-LITE LEARNABLE ACTIVATION OVERLAY
    // ========================================================================

    /**
     * Initialize KAN-lite overlay for all layers with zero coefficients.
     * Sets alpha (strength multiplier). Coefficients start at 0 = identity (no effect).
     * Call once after model load; coefficients are then tuned by forward-only learning (P7).
     *
     * @param alpha Strength multiplier (0 = disabled, 0.1 = gentle overlay, 1.0 = full strength)
     */
    external fun nativeInitKan(alpha: Float): Boolean

    /**
     * Set spline coefficients for a specific layer.
     * Coefficients define a piecewise-linear function on [-4, 3] with 8 knots.
     *
     * @param layer Target layer index (0-based)
     * @param coefficients Array of exactly 8 floats (knot values)
     */
    external fun nativeSetKanLayerCoefficients(layer: Int, coefficients: FloatArray): Boolean

    /**
     * Set the global KAN strength multiplier.
     * 0 = disabled (default), positive values scale the spline output.
     */
    external fun nativeSetKanAlpha(alpha: Float)

    /** Reset all KAN coefficients and disable the overlay. */
    external fun nativeResetKan(): Boolean

    /** Get KAN state as JSON for debugging/UI. */
    external fun nativeGetKanState(): String

    // ========================================================================
    // PART P7: FORWARD-ONLY LEARNING (SPSA)
    // ========================================================================

    /**
     * Run one forward-only learning step on the given text.
     * Uses SPSA (Simultaneous Perturbation Stochastic Approximation) to tune
     * KAN spline coefficients without backpropagation. Creates a temporary
     * probe context (~3MB) for the 2 forward passes.
     *
     * Call between conversation turns with the last assistant response text.
     *
     * @param text The text to learn from (last generated response)
     * @param learningRate Step size (typical: 0.001-0.01)
     * @param noiseScale Perturbation magnitude (typical: 0.01-0.1)
     * @param maxTokens Maximum tokens to use (more = better gradient estimate, slower)
     * @return JSON: {"success": true, "improvement": 0.05, "n_tokens": 64}
     */
    external fun nativeForwardLearnStep(
        text: String,
        learningRate: Float,
        noiseScale: Float,
        maxTokens: Int
    ): String

    // ========================================================================
    // PART D (EXTENDED): HEAD PROBING + AUTO-SCALING
    // ========================================================================

    /**
     * Probe head importance from cached direction vectors and apply head scales.
     * Reads direction vectors cached by [nativeComputePersonalityVectors], computes
     * per-layer importance (L2 norm), and scales heads at personality-relevant layers.
     * Layers where personality is NOT encoded keep scale=1.0 (preserves flash attention).
     *
     * @return JSON string: {"success": true, "n_scaled": 8, "layer_importance": [...]}
     */
    external fun nativeProbeAndSetHeadScales(axisStrengthsJson: String, cacheDir: String): String

    // ========================================================================
    // PART F: FAST WEIGHT MEMORY (BEYOND TRANSFORMER)
    // ========================================================================

    /**
     * Initialize the fast weight memory system. This creates a Hopfield-style
     * associative memory that auto-updates each token via outer product rule.
     * The memory has FIXED size regardless of conversation length.
     *
     * @param dReduced Reduced dimension for the fast weight matrix (64-256, lower = faster)
     * @param gamma Decay factor (0.99-0.999). Controls memory horizon.
     * @param eta Learning rate for writes (0.001-0.05).
     * @param injectStrength How strongly memory readout affects generation (0.01-0.5).
     */
    external fun nativeFastWeightInit(dReduced: Int, gamma: Float, eta: Float, injectStrength: Float): Boolean

    /** Update fast weight memory with latest activation. Call after each generated token. */
    external fun nativeFastWeightUpdate(): Boolean

    /** Read from fast weight memory and prepare injection. */
    external fun nativeFastWeightInject(): Boolean

    /** Reset fast weight memory (clear all stored associations). */
    external fun nativeFastWeightReset()

    /** Get fast weight state info as JSON. */
    external fun nativeFastWeightGetState(): String

    // ========================================================================
    // KV CACHE STATE PERSISTENCE
    // ========================================================================

    /**
     * Get the size in bytes needed to serialize the full KV cache state.
     *
     * Use this to estimate disk space before saving, or to check if
     * there's meaningful state to save (returns 0 if no context loaded).
     *
     * @return Size in bytes, or 0 if no context is loaded
     */
    external fun nativeGetStateSize(): Long

    /**
     * Save the current KV cache state to a file.
     *
     * Persists the full KV cache + prompt tokens to disk so a conversation
     * can be resumed instantly without re-processing the chat history.
     * The file includes model compatibility checks — loading into a
     * different model architecture will fail safely.
     *
     * @param path Absolute path to the output file
     * @return true if saved successfully
     */
    external fun nativeStateSaveToFile(path: String): Boolean

    /**
     * Load a previously saved KV cache state from a file.
     *
     * Restores the KV cache and prompt token cache, enabling instant
     * conversation resumption. The file must have been saved from the
     * same model architecture (layer count, embedding dimensions).
     *
     * After loading, multi-turn generation will use incremental KV reuse
     * from the restored state — only new tokens need to be decoded.
     *
     * @param path Absolute path to the state file
     * @return true if loaded successfully (false = model mismatch or corrupt file)
     */
    external fun nativeStateLoadFromFile(path: String): Boolean

    companion object {
        init {
            System.loadLibrary("ai_gguf")
        }

        /**
         * Recommended settings for low-end devices (< 4GB RAM)
         *
         * Thread count 0 = auto-detect performance cores (big.LITTLE aware).
         * Batch sizes tuned for 4 performance cores.
         */
        object LowEndDefaults {
            const val CONTEXT_SIZE = 1024
            const val BATCH_SIZE = 512
            const val THREADS = 0  // Auto-detect perf cores
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        /**
         * Recommended settings for mid-range devices (4-8GB RAM)
         *
         * Thread count 0 = auto-detect performance cores (big.LITTLE aware).
         * Batch sizes scaled for 4-6 performance cores.
         */
        object MidRangeDefaults {
            const val CONTEXT_SIZE = 2048
            const val BATCH_SIZE = 512
            const val THREADS = 0  // Auto-detect perf cores
            const val TEMPERATURE = 0.7f
            const val TOP_K = 40
            const val TOP_P = 0.9f
            const val MIN_P = 0.05f
        }

        /**
         * Recommended settings for high-end devices (> 8GB RAM)
         *
         * Thread count 0 = auto-detect performance cores (big.LITTLE aware).
         * Batch sizes scaled for 6+ performance cores (auto-tuned in native).
         */
        object HighEndDefaults {
            const val CONTEXT_SIZE = 4096
            const val BATCH_SIZE = 1024
            const val THREADS = 0  // Auto-detect perf cores
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