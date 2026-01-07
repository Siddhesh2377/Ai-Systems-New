/**
 * Model State Manager Implementation
 *
 * Updated for llama.cpp b7400+ (January 2026):
 * - Uses llama_memory_* API instead of deprecated llama_kv_cache_*
 * - Optimized detokenization for immediate streaming
 * - Improved sampler chain construction
 */

#include "model_state.h"
#include "../utils/logger.h"

#include <cstring>
#include <algorithm>
#include <jni.h>

// Global model state instance
ModelState g_state;

// ============================================================================
// SAMPLER CONSTRUCTION
// Updated sampler chain API
// ============================================================================

void ModelState::rebuild_sampler(
        int topK,
        float topP,
        float temp,
        float minP,
        int mirostat,
        float mirostatTau,
        float mirostatEta,
        int seed)
{
    // Free existing sampler
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        LOG_ERROR("Failed to get vocab for sampler rebuild");
        return;
    }

    // Initialize sampler chain
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* chain = llama_sampler_chain_init(sparams);

    // Add grammar sampler first if tools are enabled
    if (tools_enabled && grammar_sampler) {
        llama_sampler_chain_add(chain, grammar_sampler);
    }

    // Mirostat sampling branch
    if (mirostat > 0) {
        auto* mirostatSampler = llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab),
                seed,
                mirostatTau,
                mirostatEta,
                100 // m window
        );
        llama_sampler_chain_add(chain, mirostatSampler);
    }
        // Standard sampling branch
        // Standard sampling branch
    else {
        // 1. TEMPERATURE FIRST - must scale logits before filtering
        if (temp > 0.0f && std::abs(temp - 1.0f) > 1e-3f) {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        }

        // 2. FILTERING - order: top-k, top-p, min-p
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK));

        if (topP < 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1));
        }

        if (minP > 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_min_p(minP, 1));
        }

        // 3. DISTRIBUTION SAMPLING LAST - picks final token
        if (temp > 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
        } else {
            // Greedy sampling when temp=0
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        }
    }

    sampler = chain;
    llama_sampler_reset(sampler);

    LOG_INFO("Sampler rebuilt: topK=%d, topP=%.2f, temp=%.2f, minP=%.2f, "
             "mirostat=%d, tau=%.2f, eta=%.2f, seed=%d",
             topK, topP, temp, minP,
             mirostat, mirostatTau, mirostatEta, seed);
}

// ============================================================================
// TOKENIZATION
// ============================================================================

std::vector<llama_token> ModelState::tokenize(const std::string& text) const {
    if (!model) return {};

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};

    // Initial guess for token count (roughly 1 token per 4 chars)
    int32_t guess = static_cast<int32_t>(text.size() / 3 + 16);
    std::vector<llama_token> toks(static_cast<size_t>(guess));

    int32_t n = llama_tokenize(
            vocab,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            toks.data(),
            static_cast<int32_t>(toks.size()),
            true,  // add_bos
            true   // special tokens
    );

    // If buffer was too small, resize and retry
    if (n < 0) {
        toks.resize(static_cast<size_t>(-n));
        n = llama_tokenize(
                vocab,
                text.c_str(),
                static_cast<int32_t>(text.size()),
                toks.data(),
                static_cast<int32_t>(toks.size()),
                true,
                true
        );
    }

    if (n < 0) {
        LOG_ERROR("ModelState::tokenize: tokenization failed");
        return {};
    }

    toks.resize(static_cast<size_t>(n));
    return toks;
}

std::string ModelState::detokenize_single(llama_token t) const {
    if (!model) return {};

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) return {};

    // Most tokens are small, use stack buffer
    char buffer[256];
    int n = llama_token_to_piece(
            vocab, t,
            buffer, sizeof(buffer) - 1,
            0,     // lstrip
            false  // special
    );

    if (n >= 0 && n < static_cast<int>(sizeof(buffer))) {
        return std::string(buffer, static_cast<size_t>(n));
    }

    // Token is larger than buffer (rare) - allocate exact size
    if (n < 0) {
        std::string out(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(
                vocab, t,
                out.data(), static_cast<int>(out.size()),
                0, false
        );
        if (n > 0) {
            out.resize(static_cast<size_t>(n));
            return out;
        }
    }

    LOG_ERROR("Failed to detokenize token %d", t);
    return {};
}

// Legacy buffered detokenization
std::string ModelState::detokenize_buffered(llama_token t) {
    std::string piece = detokenize_single(t);
    if (piece.empty()) return {};

    // Add to carry buffer
    utf8_carry_buffer += piece;

    // Extract complete UTF-8 characters
    std::string complete_chars;
    size_t i = 0;

    while (i < utf8_carry_buffer.size()) {
        unsigned char c = static_cast<unsigned char>(utf8_carry_buffer[i]);
        size_t char_len = 0;

        // Determine UTF-8 character length
        if ((c & 0x80) == 0x00) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        } else {
            // Invalid start byte - skip
            LOG_WARN("Invalid UTF-8 start byte: 0x%02X", c);
            ++i;
            continue;
        }

        // Check if we have complete character
        if (i + char_len > utf8_carry_buffer.size()) {
            break;
        }

        // Validate continuation bytes
        bool valid = true;
        for (size_t j = 1; j < char_len; ++j) {
            unsigned char cont = static_cast<unsigned char>(utf8_carry_buffer[i + j]);
            if ((cont & 0xC0) != 0x80) {
                valid = false;
                break;
            }
        }

        if (valid) {
            complete_chars.append(utf8_carry_buffer.substr(i, char_len));
            i += char_len;
        } else {
            ++i;
        }
    }

    // Keep incomplete bytes in buffer
    utf8_carry_buffer = utf8_carry_buffer.substr(i);

    return complete_chars;
}

std::string ModelState::flush_utf8_buffer() {
    std::string remaining = utf8_carry_buffer;
    utf8_carry_buffer.clear();

    if (!remaining.empty()) {
        LOG_WARN("Flushing incomplete UTF-8 sequence: %zu bytes", remaining.size());
    }

    return remaining;
}

llama_token ModelState::space_token() const {
    if (!model) return 0;

    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_token out[4];
    int n = llama_tokenize(vocab, " ", 1, out, 4, true, true);
    return (n > 0) ? out[0] : 0;
}

// ============================================================================
// RESOURCE MANAGEMENT
// ============================================================================

void ModelState::release() {
    if (grammar_sampler) {
        llama_sampler_free(grammar_sampler);
        grammar_sampler = nullptr;
    }
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }

    utf8_carry_buffer.clear();
    llama_backend_free();

    LOG_INFO("ModelState: all resources released");
}

void ModelState::prepare_for_generation() {
    if (!ctx) return;

    // Clear KV cache - requires 2 arguments!
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_memory_clear(mem, true);
    }

    if (sampler) {
        llama_sampler_reset(sampler);
    }

    utf8_carry_buffer.clear();

    LOG_INFO("prepare_for_generation: KV cache cleared, sampler reset");
}

// ============================================================================
// INFERENCE
// ============================================================================

bool ModelState::decode_prompt(const std::vector<llama_token>& toks) const {
    if (!ctx || toks.empty()) return true;

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    int32_t pos = 0;
    size_t idx = 0;

    while (idx < toks.size()) {
        int32_t take = std::min<int32_t>(
                batch_size,
                static_cast<int32_t>(toks.size() - idx)
        );

        batch.n_tokens = take;
        for (int i = 0; i < take; ++i) {
            batch.token[i] = toks[idx + i];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == take - 1); // Only last token needs logits
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERROR("ModelState::decode_prompt: llama_decode failed");
            llama_batch_free(batch);
            return false;
        }

        pos += take;
        idx += static_cast<size_t>(take);
    }

    llama_batch_free(batch);
    return true;
}

void ModelState::warmup_context() const {
    llama_token space = space_token();
    if (space == 0) return;

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = space;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;

    llama_decode(ctx, batch);
    llama_batch_free(batch);
}

// ============================================================================
// STATE PERSISTENCE
// ============================================================================

jlong ModelState::get_state_size() const {
    if (!ctx) return 0;
    return static_cast<jlong>(llama_state_get_size(ctx));
}

void* ModelState::get_state_data(void* buffer, size_t size) const {
    if (!ctx) return nullptr;
    size_t written = llama_state_get_data(
            ctx,
            static_cast<uint8_t*>(buffer),
            size
    );
    return (written > 0) ? buffer : nullptr;
}

bool ModelState::load_state_data(const void* data, size_t size) const {
    if (!ctx) return false;
    size_t n = llama_state_set_data(
            ctx,
            static_cast<const uint8_t*>(data),
            size
    );
    return n == size;
}