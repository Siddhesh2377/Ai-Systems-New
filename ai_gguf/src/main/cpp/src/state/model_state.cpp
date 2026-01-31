/**
 * Model State Manager Implementation
 *
 * Updated for llama.cpp b7400+ (January 2026):
 * - Uses llama_memory_* API instead of deprecated llama_kv_cache_*
 * - Optimized detokenization for immediate streaming
 * - Improved sampler chain construction
 * - Grammar caching for tool calls (avoids rebuilds)
 * - Memory metrics tracking
 */

#include "model_state.h"
#include "../utils/logger.h"
#include "../chat/chat_template.h"

#include <cstring>
#include <cctype>
#include <algorithm>
#include <jni.h>

#if defined(__ANDROID__)
#include <sys/sysinfo.h>
#endif

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
    // Cache params for multi-turn rebuilds
    cached_sampler_params = {topK, topP, temp, minP, mirostat, mirostatTau, mirostatEta, seed};

    // Free existing sampler chain (this frees all samplers added to the chain,
    // but NOT our master grammar_sampler since we clone it before adding)
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

    // Add a CLONE of grammar sampler first if tools are enabled.
    // The master grammar_sampler is owned by ModelState for reuse across turns.
    // The chain takes ownership of the clone and frees it when the chain is freed.
    if (tools_enabled && grammar_sampler) {
        llama_sampler* grammar_clone = llama_sampler_clone(grammar_sampler);
        if (grammar_clone) {
            llama_sampler_chain_add(chain, grammar_clone);
        } else {
            LOG_WARN("Failed to clone grammar sampler, proceeding without grammar");
        }
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

void ModelState::rebuild_sampler_cached() {
    const auto& p = cached_sampler_params;
    rebuild_sampler(p.topK, p.topP, p.temp, p.minP,
                    p.mirostat, p.mirostatTau, p.mirostatEta, p.seed);
}

void ModelState::reset_grammar_sampler() {
    if (grammar_sampler) {
        llama_sampler_reset(grammar_sampler);
    }
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
    stop_strings.clear();
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

// ============================================================================
// GRAMMAR MANAGEMENT (Optimized for low-end devices)
// ============================================================================

void ModelState::update_grammar_if_needed() {
    if (!tools_enabled || tools_json.empty()) {
        // No tools - clean up any existing grammar
        if (grammar_sampler) {
            llama_sampler_free(grammar_sampler);
            grammar_sampler = nullptr;
        }
        grammar_needs_rebuild = false;
        cached_tools_json.clear();
        return;
    }

    // Check if we can reuse cached grammar (including "no grammar" cached state)
    if (!grammar_needs_rebuild && tools_json == cached_tools_json) {
        if (grammar_sampler) {
            LOG_INFO("Reusing cached grammar sampler");
        }
        return;
    }

    LOG_INFO("Building new grammar sampler (mode=%s, typed=%s)",
             grammar_mode == GrammarMode::STRICT ? "strict" : "lazy",
             use_typed_grammar ? "yes" : "no");

    // Free existing grammar sampler
    if (grammar_sampler) {
        llama_sampler_free(grammar_sampler);
        grammar_sampler = nullptr;
    }

    // Build both grammar strings upfront
    std::string typed_grammar;
    if (use_typed_grammar) {
        typed_grammar = chat::build_tool_grammar_typed(tools_json);
    }
    std::string generic_grammar = chat::build_tool_grammar(tools_json);

    if (typed_grammar.empty() && generic_grammar.empty()) {
        LOG_WARN("Failed to build any tool grammar string - continuing without grammar");
        cached_tools_json = tools_json;
        grammar_needs_rebuild = false;
        return;
    }

    // Log grammar strings for debugging
    if (!typed_grammar.empty()) {
        LOG_INFO("Typed grammar length: %zu chars", typed_grammar.size());
    }
    if (!generic_grammar.empty()) {
        LOG_INFO("Generic grammar length: %zu chars", generic_grammar.size());
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        LOG_ERROR("Failed to get vocab for grammar");
        cached_tools_json = tools_json;
        grammar_needs_rebuild = false;
        return;
    }

    // Helper: try to init sampler with given grammar in the preferred mode
    auto try_init_preferred = [&](const std::string& g) -> llama_sampler* {
        if (g.empty()) return nullptr;
        if (grammar_mode == GrammarMode::LAZY) {
            const char* tp[] = { "\\{" };
            return llama_sampler_init_grammar_lazy_patterns(
                vocab, g.c_str(), "root", tp, 1, nullptr, 0);
        } else {
            return llama_sampler_init_grammar(vocab, g.c_str(), "root");
        }
    };

    // Helper: try to init sampler with given grammar in the alternate mode
    auto try_init_alt = [&](const std::string& g) -> llama_sampler* {
        if (g.empty()) return nullptr;
        if (grammar_mode == GrammarMode::LAZY) {
            // Preferred is lazy, alt is strict
            return llama_sampler_init_grammar(vocab, g.c_str(), "root");
        } else {
            // Preferred is strict, alt is lazy
            const char* tp[] = { "\\{" };
            return llama_sampler_init_grammar_lazy_patterns(
                vocab, g.c_str(), "root", tp, 1, nullptr, 0);
        }
    };

    // Attempt 1: typed grammar + preferred mode
    if (!typed_grammar.empty()) {
        grammar_sampler = try_init_preferred(typed_grammar);
        if (grammar_sampler) {
            LOG_INFO("Grammar sampler created: typed + %s mode",
                     grammar_mode == GrammarMode::STRICT ? "strict" : "lazy");
        }
    }

    // Attempt 2: generic grammar + preferred mode
    if (!grammar_sampler && !generic_grammar.empty()) {
        LOG_INFO("Trying generic grammar with preferred mode...");
        grammar_sampler = try_init_preferred(generic_grammar);
        if (grammar_sampler) {
            LOG_INFO("Grammar sampler created: generic + %s mode",
                     grammar_mode == GrammarMode::STRICT ? "strict" : "lazy");
        }
    }

    // Attempt 3: typed grammar + alternate mode
    if (!grammar_sampler && !typed_grammar.empty()) {
        LOG_INFO("Trying typed grammar with alternate mode...");
        grammar_sampler = try_init_alt(typed_grammar);
        if (grammar_sampler) {
            LOG_INFO("Grammar sampler created: typed + %s mode",
                     grammar_mode == GrammarMode::STRICT ? "lazy" : "strict");
        }
    }

    // Attempt 4: generic grammar + alternate mode
    if (!grammar_sampler && !generic_grammar.empty()) {
        LOG_INFO("Trying generic grammar with alternate mode...");
        grammar_sampler = try_init_alt(generic_grammar);
        if (grammar_sampler) {
            LOG_INFO("Grammar sampler created: generic + %s mode",
                     grammar_mode == GrammarMode::STRICT ? "lazy" : "strict");
        }
    }

    // Cache state regardless of success - avoid retrying every generation
    cached_tools_json = tools_json;
    grammar_needs_rebuild = false;

    if (grammar_sampler) {
        LOG_INFO("Grammar sampler cached successfully");
    } else {
        // IMPORTANT: Do NOT set tools_enabled = false.
        // Grammar is optional - the model still sees the tool preamble in its
        // prompt, and ToolCallState detects tool calls in the output stream.
        // Grammar only makes tool calls more reliable by constraining output.
        LOG_WARN("All grammar init attempts failed - tool calling continues WITHOUT grammar constraints");
        LOG_WARN("Model will generate freely; tool calls detected via ToolCallState");
    }
}

// ============================================================================
// FALLBACK CHAT TEMPLATE
// ============================================================================

void ModelState::apply_fallback_chat_template() {
    if (!model) return;

    // Skip if a custom template is already set
    if (!chat_template_override.empty()) {
        LOG_INFO("Custom chat template already set, skipping fallback");
        return;
    }

    // Skip if the model already has a built-in template
    const char* builtin = llama_model_chat_template(model, nullptr);
    if (builtin && *builtin) {
        LOG_INFO("Model has built-in chat template, skipping fallback");
        return;
    }

    // No template — detect architecture and apply the right one
    static char arch_buf[128] = {0};
    int32_t len = llama_model_meta_val_str(model, "general.architecture",
                                           arch_buf, sizeof(arch_buf));
    std::string arch = (len > 0) ? std::string(arch_buf) : "";

    // Convert to lowercase for comparison
    for (auto& c : arch) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (arch.find("gemma") != std::string::npos) {
        // Gemma / Gemma2 template
        chat_template_override =
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n"
            "{% elif message['role'] == 'user' %}"
            "<start_of_turn>user\n"
            "{{ message['content'] }}<end_of_turn>\n"
            "<start_of_turn>model\n"
            "{% elif message['role'] == 'assistant' or message['role'] == 'model' %}"
            "{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";
        LOG_INFO("Applied fallback Gemma chat template for architecture: %s", arch.c_str());
    }
    else if (arch.find("llama") != std::string::npos ||
             arch.find("mistral") != std::string::npos ||
             arch.find("mixtral") != std::string::npos) {
        // Llama 3 / Mistral — ChatML-style
        chat_template_override =
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        LOG_INFO("Applied fallback ChatML template for architecture: %s", arch.c_str());
    }
    else if (arch.find("phi") != std::string::npos) {
        // Phi template
        chat_template_override =
            "{% for message in messages %}"
            "<|{{ message['role'] }}|>\n"
            "{{ message['content'] }}<|end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>\n{% endif %}";
        LOG_INFO("Applied fallback Phi template for architecture: %s", arch.c_str());
    }
    else if (arch.find("qwen") != std::string::npos) {
        // Qwen — ChatML
        chat_template_override =
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        LOG_INFO("Applied fallback ChatML template for architecture: %s", arch.c_str());
    }
    else {
        // Generic ChatML fallback — works reasonably with most models
        chat_template_override =
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n"
            "{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        LOG_INFO("Applied generic ChatML fallback template for unknown architecture: %s",
                 arch.empty() ? "(none)" : arch.c_str());
    }
}

// ============================================================================
// STOP STRING DETECTION
// ============================================================================

void ModelState::detect_stop_strings() {
    stop_strings.clear();

    if (!model) return;

    // Use custom template if set, otherwise use model's built-in template
    const char* tmpl = chat_template_override.empty()
                       ? llama_model_chat_template(model, nullptr)
                       : chat_template_override.c_str();

    bool matched_template = false;

    if (tmpl && *tmpl) {
        std::string t(tmpl);

        // Gemma: <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        if (t.find("<start_of_turn>") != std::string::npos) {
            stop_strings.push_back("<end_of_turn>");
            stop_strings.push_back("<start_of_turn>");
            matched_template = true;
        }
        // ChatML: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        else if (t.find("<|im_start|>") != std::string::npos) {
            stop_strings.push_back("<|im_end|>");
            stop_strings.push_back("<|im_start|>");
            matched_template = true;
        }
        // Llama 3: <|start_header_id|>user<|end_header_id|>\n...<|eot_id|>
        else if (t.find("<|start_header_id|>") != std::string::npos) {
            stop_strings.push_back("<|eot_id|>");
            stop_strings.push_back("<|start_header_id|>");
            matched_template = true;
        }
        // Phi: <|user|>\n...<|end|>\n<|assistant|>\n
        else if (t.find("<|assistant|>") != std::string::npos) {
            stop_strings.push_back("<|end|>");
            stop_strings.push_back("<|user|>");
            matched_template = true;
        }
        // Mistral/Mixtral: [INST]...[/INST]
        else if (t.find("[INST]") != std::string::npos) {
            stop_strings.push_back("</s>");
            stop_strings.push_back("[INST]");
            matched_template = true;
        }
        // Command-R
        else if (t.find("<|END_OF_TURN_TOKEN|>") != std::string::npos) {
            stop_strings.push_back("<|END_OF_TURN_TOKEN|>");
            stop_strings.push_back("<|START_OF_TURN_TOKEN|>");
            matched_template = true;
        }
    }

    // ====================================================================
    // FALLBACK STOP STRINGS
    // If no chat template or unrecognized template, the code uses a plain
    // "User: ... Assistant: ..." format. Small models will generate past
    // their turn and produce fake "User:" lines. Always add these as a
    // safety net — they catch the conversation-loop problem regardless
    // of template type.
    // ====================================================================
    stop_strings.push_back("\nUser:");
    stop_strings.push_back("\nHuman:");
    stop_strings.push_back("\n### User");
    stop_strings.push_back("\n<|user|>");

    if (matched_template) {
        LOG_INFO("Detected %zu stop strings (template + fallback):", stop_strings.size());
    } else {
        LOG_INFO("No chat template — using %zu fallback stop strings:", stop_strings.size());
    }
    for (const auto& s : stop_strings) {
        LOG_INFO("  stop: \"%s\"", s.c_str());
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

void ModelState::update_memory_metrics() {
    if (!model || !ctx) {
        memory_metrics = MemoryMetrics{};
        return;
    }

    // Estimate model size (this is approximate)
    // llama.cpp doesn't expose exact memory usage, so we estimate
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    int32_t n_embd = llama_model_n_embd(model);
    int32_t n_layer = llama_model_n_layer(model);

    // Rough estimate: vocab embedding + layers
    // This is a simplified calculation
    memory_metrics.model_size_bytes = static_cast<size_t>(n_vocab) * n_embd * sizeof(float);

    // Context memory estimate: KV cache
    // KV cache size = 2 * n_layer * ctx_size * n_embd * sizeof(float16)
    memory_metrics.context_size_bytes = estimate_context_memory(ctx_size, n_embd, n_layer);

#if defined(__ANDROID__)
    // Get system memory info
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        size_t total_mem = si.totalram * si.mem_unit;
        size_t used_mem = memory_metrics.model_size_bytes + memory_metrics.context_size_bytes;
        memory_metrics.memory_usage_percent = (total_mem > 0)
            ? (static_cast<float>(used_mem) / static_cast<float>(total_mem)) * 100.0f
            : 0.0f;
    }
#endif

    // Track peak
    size_t current_total = memory_metrics.model_size_bytes + memory_metrics.context_size_bytes;
    if (current_total > memory_metrics.peak_memory_bytes) {
        memory_metrics.peak_memory_bytes = current_total;
    }

    LOG_INFO("Memory metrics updated: model=%zu MB, ctx=%zu MB, peak=%zu MB",
             memory_metrics.model_size_bytes / (1024 * 1024),
             memory_metrics.context_size_bytes / (1024 * 1024),
             memory_metrics.peak_memory_bytes / (1024 * 1024));
}

size_t ModelState::estimate_context_memory(int32_t ctx_size, int32_t n_embd, int32_t n_layer) {
    // KV cache: 2 (K and V) * layers * context * embedding * sizeof(float16)
    // Plus some overhead for attention weights
    const size_t kv_cache = 2 * static_cast<size_t>(n_layer) *
                            static_cast<size_t>(ctx_size) *
                            static_cast<size_t>(n_embd) * sizeof(uint16_t);

    // Add ~10% overhead for internal buffers
    return static_cast<size_t>(kv_cache * 1.1);
}