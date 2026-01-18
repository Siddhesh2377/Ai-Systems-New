#include "embedding_state.h"
#include "../utils/logger.h"
#include <cmath>
#include <algorithm>
#include <chrono>

// Global instance
EmbeddingState g_embedding_state;

void EmbeddingState::release() {
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }

    n_embd = 0;
    ctx_size = 512;
    batch_size = 512;
    pooling_type = PoolingType::MEAN;

    LOG_INFO("EmbeddingState released");
}

int32_t EmbeddingState::get_embedding_dimension() const {
    if (!model) return 0;
    return llama_model_n_embd(model);
}

PoolingType EmbeddingState::detect_pooling_type() const {
    if (!model) return PoolingType::MEAN;

    // Check model metadata for pooling type
    char pooling_buf[32] = {0};
    int32_t len = llama_model_meta_val_str(model, "pooling.type", pooling_buf, sizeof(pooling_buf));

    if (len > 0) {
        std::string pooling_str(pooling_buf);
        if (pooling_str == "mean") return PoolingType::MEAN;
        if (pooling_str == "cls") return PoolingType::CLS;
        if (pooling_str == "last") return PoolingType::LAST;
        if (pooling_str == "max") return PoolingType::MAX;
    }

    // Default to mean pooling for most embedding models
    return PoolingType::MEAN;
}

std::vector<llama_token> EmbeddingState::tokenize(const std::string& text) const {
    if (!model || text.empty()) return {};

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};

    // Get max tokens
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t max_tokens = std::min(ctx_size, 8192);

    // Allocate buffer for tokens
    std::vector<llama_token> tokens(max_tokens);

    // Tokenize
    int32_t n_tokens = llama_tokenize(
            vocab,
            text.c_str(),
            text.length(),
            tokens.data(),
            max_tokens,
            true,   // add_special (add BOS/EOS)
            false   // parse_special
    );

    if (n_tokens < 0) {
        LOG_ERROR("Tokenization failed or buffer too small");
        return {};
    }

    tokens.resize(n_tokens);
    return tokens;
}

int32_t EmbeddingState::estimate_token_count(const std::string& text) const {
    // Rough estimate: ~4 characters per token for English
    return static_cast<int32_t>(text.length() / 4);
}

std::vector<float> EmbeddingState::apply_pooling(
        const float* embeddings,
        int32_t n_tokens,
        int32_t n_embd,
        PoolingType pooling
) const {
    std::vector<float> result(n_embd, 0.0f);

    if (!embeddings || n_tokens <= 0 || n_embd <= 0) {
        return result;
    }

    switch (pooling) {
        case PoolingType::MEAN: {
            // Average across all tokens
            for (int32_t i = 0; i < n_tokens; ++i) {
                for (int32_t j = 0; j < n_embd; ++j) {
                    result[j] += embeddings[i * n_embd + j];
                }
            }
            for (int32_t j = 0; j < n_embd; ++j) {
                result[j] /= static_cast<float>(n_tokens);
            }
            break;
        }

        case PoolingType::CLS: {
            // Use first token (CLS token)
            for (int32_t j = 0; j < n_embd; ++j) {
                result[j] = embeddings[j];
            }
            break;
        }

        case PoolingType::LAST: {
            // Use last token
            int32_t last_offset = (n_tokens - 1) * n_embd;
            for (int32_t j = 0; j < n_embd; ++j) {
                result[j] = embeddings[last_offset + j];
            }
            break;
        }

        case PoolingType::MAX: {
            // Max pooling across tokens
            for (int32_t j = 0; j < n_embd; ++j) {
                float max_val = embeddings[j];
                for (int32_t i = 1; i < n_tokens; ++i) {
                    max_val = std::max(max_val, embeddings[i * n_embd + j]);
                }
                result[j] = max_val;
            }
            break;
        }

        case PoolingType::NONE:
        default: {
            // Return all embeddings (not pooled)
            // For this case, result size should be n_tokens * n_embd
            result.resize(n_tokens * n_embd);
            std::copy(embeddings, embeddings + (n_tokens * n_embd), result.begin());
            break;
        }
    }

    return result;
}

void EmbeddingState::normalize_vector(std::vector<float>& vec) const {
    if (vec.empty()) return;

    // Calculate L2 norm
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    // Normalize (avoid division by zero)
    if (norm > 1e-8f) {
        for (float& val : vec) {
            val /= norm;
        }
    }
}

EmbeddingOutput EmbeddingState::encode(
        const std::string& text,
        bool normalize,
        EmbeddingProgressCallback progress_callback
) {
    EmbeddingOutput output;
    auto start_time = std::chrono::steady_clock::now();

    if (!is_ready()) {
        LOG_ERROR("EmbeddingState not ready");
        return output;
    }

    // Tokenize input
    std::vector<llama_token> tokens = tokenize(text);
    if (tokens.empty()) {
        LOG_ERROR("Tokenization failed");
        return output;
    }

    output.num_tokens = static_cast<int32_t>(tokens.size());
    LOG_INFO("Encoding %d tokens", output.num_tokens);

    // Report initial progress
    if (progress_callback) {
        progress_callback(0.0f, 0, output.num_tokens);
    }

    // Create batch for encoding
    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    // Process tokens in batches
    int32_t n_processed = 0;
    for (size_t i = 0; i < tokens.size(); i += batch_size) {
        int32_t batch_end = std::min(static_cast<int32_t>(i + batch_size),
                                     static_cast<int32_t>(tokens.size()));
        int32_t batch_len = batch_end - static_cast<int32_t>(i);

        // Fill batch
        batch.n_tokens = batch_len;
        for (int32_t j = 0; j < batch_len; ++j) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = static_cast<int32_t>(i + j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = false;  // We need embeddings, not logits
        }

        // Decode batch
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERROR("llama_decode failed at batch %zu", i / batch_size);
            llama_batch_free(batch);
            return output;
        }

        n_processed += batch_len;

        // Report progress
        if (progress_callback) {
            float progress = static_cast<float>(n_processed) / static_cast<float>(tokens.size());
            progress_callback(progress, n_processed, output.num_tokens);
        }
    }

    // Get embeddings from context
    // For embedding models, we need to get the embeddings directly
    const float* embd = llama_get_embeddings(ctx);
    if (!embd) {
        LOG_ERROR("No embeddings available - model may not be in embeddings mode");
        llama_batch_free(batch);
        return output;
    }

    // Apply pooling to get final embedding vector
    output.dimension = n_embd;
    output.pooling = pooling_type;
    output.embeddings = apply_pooling(embd, output.num_tokens, n_embd, pooling_type);

    // Normalize if requested
    if (normalize && pooling_type != PoolingType::NONE) {
        normalize_vector(output.embeddings);
    }

    // Calculate time taken
    auto end_time = std::chrono::steady_clock::now();
    output.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

    // Report completion
    if (progress_callback) {
        progress_callback(1.0f, output.num_tokens, output.num_tokens);
    }

    llama_batch_free(batch);

    LOG_INFO("Encoding completed: %d dimensions, %lld ms", output.dimension, output.time_ms);
    return output;
}