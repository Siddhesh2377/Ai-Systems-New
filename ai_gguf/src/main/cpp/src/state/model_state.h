#pragma once

/**
 * Model State Manager for llama.cpp JNI bindings
 *
 * Updated for llama.cpp b7400+ (January 2026):
 * - Uses llama_memory_* API instead of deprecated llama_kv_cache_*
 * - Optimized detokenization for streaming
 * - Efficient UTF-8 handling
 * - Grammar caching for tool calls
 * - Configurable batch sizes for low-end devices
 * - Multi-turn tool calling with lazy grammar support
 */

#include "llama.h"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <jni.h>

/**
 * Memory usage metrics for monitoring
 */
struct MemoryMetrics {
    size_t model_size_bytes = 0;      // Approximate model memory
    size_t context_size_bytes = 0;    // KV cache and context memory
    size_t peak_memory_bytes = 0;     // Peak observed memory
    float memory_usage_percent = 0.0f; // Percentage of available memory
};

/**
 * Grammar mode for tool calling
 */
enum class GrammarMode {
    STRICT,  // Grammar active from first token (forces tool call output)
    LAZY     // Grammar activates only on trigger pattern (model chooses tool vs text)
};

/**
 * Cached sampler parameters for multi-turn rebuilds
 */
struct SamplerParams {
    int topK = 40;
    float topP = 0.9f;
    float temp = 0.7f;
    float minP = 0.05f;
    int mirostat = 0;
    float mirostatTau = 5.0f;
    float mirostatEta = 0.1f;
    int seed = -1;
};

/**
 * Progress callback for model loading
 */
using LoadProgressCallback = std::function<void(float progress)>;

class ModelState {
public:
    // Core llama.cpp state
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    llama_sampler* grammar_sampler = nullptr;

    // Configuration
    int32_t ctx_size = 0;
    int32_t batch_size = 512;
    int32_t ubatch_size = 256;  // Micro-batch size for low-end devices

    // Chat/Tool state
    std::string system_prompt;
    std::string chat_template_override;
    std::string tools_json;
    bool tools_enabled = false;

    // Grammar configuration
    GrammarMode grammar_mode = GrammarMode::STRICT;
    bool use_typed_grammar = true;  // Use parameter-aware GBNF

    // Grammar caching for tool calls
    std::string cached_tools_json;     // Last tools JSON used to build grammar
    bool grammar_needs_rebuild = true;  // Flag to trigger grammar rebuild

    // Cached sampler params for multi-turn rebuilds
    SamplerParams cached_sampler_params;

    // UTF-8 carry buffer for incomplete sequences (legacy)
    std::string utf8_carry_buffer;

    // Memory tracking
    MemoryMetrics memory_metrics;

    // ========================================================================
    // CORE METHODS
    // ========================================================================

    /**
     * Check if model is ready for generation
     */
    bool is_ready() const {
        return model != nullptr && ctx != nullptr && sampler != nullptr;
    }

    /**
     * Release all resources
     */
    void release();

    /**
     * Prepare for new generation (clears KV cache and sampler state)
     */
    void prepare_for_generation();

    /**
     * Rebuild sampler with new parameters
     */
    void rebuild_sampler(
            int topK,
            float topP,
            float temp,
            float minP,
            int mirostat,
            float mirostatTau,
            float mirostatEta,
            int seed
    );

    /**
     * Rebuild sampler using cached parameters (for multi-turn)
     * Each call creates a fresh grammar clone in the sampler chain
     */
    void rebuild_sampler_cached();

    // ========================================================================
    // GRAMMAR MANAGEMENT (Optimized for low-end devices)
    // ========================================================================

    /**
     * Initialize or update grammar sampler for tool calls
     * Only rebuilds if tools_json has changed (caching)
     * Respects grammar_mode (STRICT vs LAZY) and use_typed_grammar
     */
    void update_grammar_if_needed();

    /**
     * Reset grammar sampler state for reuse across turns
     */
    void reset_grammar_sampler();

    /**
     * Check if grammar needs to be rebuilt
     */
    bool needs_grammar_rebuild() const {
        return grammar_needs_rebuild || (tools_json != cached_tools_json);
    }

    /**
     * Force grammar rebuild on next generation
     */
    void invalidate_grammar() {
        grammar_needs_rebuild = true;
    }

    // ========================================================================
    // TOKENIZATION
    // ========================================================================

    /**
     * Tokenize text into tokens
     */
    std::vector<llama_token> tokenize(const std::string& text) const;

    /**
     * Detokenize a single token to string
     * This is the optimized version - returns raw bytes
     */
    std::string detokenize_single(llama_token t) const;

    /**
     * Detokenize with UTF-8 buffering (legacy)
     * Handles incomplete UTF-8 sequences
     */
    std::string detokenize_buffered(llama_token t);

    /**
     * Flush UTF-8 buffer (legacy)
     */
    std::string flush_utf8_buffer();

    /**
     * Get space token for edge cases
     */
    llama_token space_token() const;

    // ========================================================================
    // INFERENCE
    // ========================================================================

    /**
     * Decode prompt tokens (prefill phase)
     */
    bool decode_prompt(const std::vector<llama_token>& toks) const;

    /**
     * Warm up context
     */
    void warmup_context() const;

    // ========================================================================
    // MEMORY MANAGEMENT
    // ========================================================================

    /**
     * Update memory metrics
     */
    void update_memory_metrics();

    /**
     * Get current memory metrics
     */
    const MemoryMetrics& get_memory_metrics() const { return memory_metrics; }

    /**
     * Estimate memory needed for given context size
     */
    static size_t estimate_context_memory(int32_t ctx_size, int32_t n_embd, int32_t n_layer);

    // ========================================================================
    // STATE PERSISTENCE
    // ========================================================================

    jlong get_state_size() const;
    void* get_state_data(void* buffer, size_t size) const;
    bool load_state_data(const void* data, size_t size) const;
};

// Global model state instance
extern ModelState g_state;