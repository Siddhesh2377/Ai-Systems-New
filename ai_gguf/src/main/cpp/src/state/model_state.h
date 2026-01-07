#pragma once

/**
 * Model State Manager for llama.cpp JNI bindings
 *
 * Updated for llama.cpp b7400+ (January 2026):
 * - Uses llama_memory_* API instead of deprecated llama_kv_cache_*
 * - Optimized detokenization for streaming
 * - Efficient UTF-8 handling
 */

#include "llama.h"
#include <string>
#include <vector>
#include <cstdint>
#include <jni.h>

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

    // Chat/Tool state
    std::string system_prompt;
    std::string chat_template_override;
    std::string tools_json;
    bool tools_enabled = false;

    // UTF-8 carry buffer for incomplete sequences (legacy)
    std::string utf8_carry_buffer;

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
    // STATE PERSISTENCE
    // ========================================================================

    jlong get_state_size() const;
    void* get_state_data(void* buffer, size_t size) const;
    bool load_state_data(const void* data, size_t size) const;
};

// Global model state instance
extern ModelState g_state;