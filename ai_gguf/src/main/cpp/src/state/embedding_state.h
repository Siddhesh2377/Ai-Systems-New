#pragma once

/**
 * Embedding Model State Manager for llama.cpp JNI bindings
 *
 * Handles text encoding to vector embeddings using llama.cpp models
 * Supports various pooling strategies (mean, CLS, last token, max)
 */

#include "llama.h"
#include <string>
#include <vector>
#include <cstdint>
#include <functional>

/**
 * Pooling strategy for combining token embeddings
 */
enum class PoolingType {
    NONE = 0,   // No pooling (return all token embeddings)
    MEAN = 1,   // Average pooling across all tokens
    CLS = 2,    // Use [CLS] token embedding only
    LAST = 3,   // Use last token embedding
    MAX = 4     // Max pooling across tokens
};

/**
 * Output from text encoding operation
 */
struct EmbeddingOutput {
    std::vector<float> embeddings;  // The embedding vector
    int32_t dimension = 0;           // Embedding dimension
    PoolingType pooling = PoolingType::MEAN;  // Pooling type used
    int32_t num_tokens = 0;          // Number of tokens processed
    int64_t time_ms = 0;             // Time taken in milliseconds
};

/**
 * Progress callback for embedding generation
 * Parameters: (progress 0.0-1.0, current_token, total_tokens)
 */
using EmbeddingProgressCallback = std::function<void(float, int32_t, int32_t)>;

class EmbeddingState {
public:
    // Core llama.cpp state
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;

    // Configuration
    int32_t ctx_size = 512;          // Context size for embeddings
    int32_t batch_size = 512;        // Batch size for encoding
    int32_t n_threads = 4;           // Number of threads
    int32_t n_embd = 0;              // Embedding dimension

    // Pooling configuration
    PoolingType pooling_type = PoolingType::MEAN;

    // ========================================================================
    // CORE METHODS
    // ========================================================================

    /**
     * Check if embedding model is ready
     */
    bool is_ready() const {
        return model != nullptr && ctx != nullptr;
    }

    /**
     * Release all resources
     */
    void release();

    /**
     * Get embedding dimension from model
     */
    int32_t get_embedding_dimension() const;

    /**
     * Detect pooling type from model metadata
     */
    PoolingType detect_pooling_type() const;

    // ========================================================================
    // ENCODING
    // ========================================================================

    /**
     * Encode text to embedding vector
     *
     * @param text Text to encode
     * @param normalize Whether to L2-normalize the output vector
     * @param progress_callback Optional progress callback
     * @return EmbeddingOutput containing the embedding vector
     */
    EmbeddingOutput encode(
            const std::string& text,
            bool normalize = true,
            EmbeddingProgressCallback progress_callback = nullptr
    );

    // ========================================================================
    // TOKENIZATION
    // ========================================================================

    /**
     * Tokenize text into tokens
     */
    std::vector<llama_token> tokenize(const std::string& text) const;

    /**
     * Get token count for text without full tokenization
     */
    int32_t estimate_token_count(const std::string& text) const;

private:
    /**
     * Apply pooling to token embeddings
     */
    std::vector<float> apply_pooling(
            const float* embeddings,
            int32_t n_tokens,
            int32_t n_embd,
            PoolingType pooling
    ) const;

    /**
     * L2 normalize a vector
     */
    void normalize_vector(std::vector<float>& vec) const;
};

// Global embedding state instance
extern EmbeddingState g_embedding_state;