// Tool-Neuron JNI Bridge — llama.cpp JNI interface for GGUFNativeLib

#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <sys/syscall.h>
#include <android/log.h>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"

#include "tool-manager.h"
#include "thread-engine.h"
#include "rag-engine.h"
#include "rag_ingest/rag_ingest.h"
#include "text_digest/text_digest.h"
#include "error_tracker.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#define TAG "ToolNeuron-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using json = nlohmann::ordered_json;

// ── llama.cpp log callback → Android logcat ──

static void llama_android_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (text == nullptr || text[0] == '\0') return;
    // Strip trailing newline
    size_t len = strlen(text);
    char buf[2048];
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, text, len);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) len--;
    buf[len] = '\0';
    if (len == 0) return;

    switch (level) {
        case GGML_LOG_LEVEL_ERROR: LOGE("%s", buf); break;
        case GGML_LOG_LEVEL_WARN:  LOGW("%s", buf); break;
        case GGML_LOG_LEVEL_DEBUG:
        case GGML_LOG_LEVEL_CONT:  break;
        default:                   LOGI("%s", buf); break;
    }
}

static std::once_flag g_backend_init_flag;

static void ensure_backend_init() {
    std::call_once(g_backend_init_flag, [] {
        llama_log_set(llama_android_log_callback, nullptr);
        llama_backend_init();
        LOGI("llama backend initialized, log callback set");
    });
}

// Cached JNI method IDs (resolved once per callback class, reused across generation calls)
// Avoids repeated GetObjectClass + GetMethodID (~5-30µs each) on every generation invocation
static jclass    g_cb_class       = nullptr; // global ref to last-seen callback class
static jmethodID g_onToken        = nullptr;
static jmethodID g_onToolCall     = nullptr;
static jmethodID g_onDone         = nullptr;
static jmethodID g_onError        = nullptr;
static jmethodID g_onMetrics      = nullptr;
static jmethodID g_onProgress     = nullptr; // nullable — added later, default no-op in Kotlin
static jmethodID g_onTokenBytes   = nullptr; // nullable — zero-copy byte[] fast path
static jmethodID g_onVlmStageMetrics = nullptr; // nullable — per-stage VLM timings

// Cached EmbeddingCallback method IDs
static jclass    g_embed_cb_class  = nullptr;
static jmethodID g_embed_onComplete = nullptr;
static jmethodID g_embed_onError   = nullptr;

// resolve and cache StreamCallback method IDs from the callback object's class
// returns true if all required methods are found
static bool ensure_callback_methods(JNIEnv * env, jobject callback) {
    jclass cls = env->GetObjectClass(callback);
    if (g_cb_class && env->IsSameObject(cls, g_cb_class)) {
        env->DeleteLocalRef(cls);
        return true;
    }
    // new class — resolve all method IDs
    if (g_cb_class) env->DeleteGlobalRef(g_cb_class);
    g_cb_class = (jclass)env->NewGlobalRef(cls);
    g_onToken    = env->GetMethodID(cls, "onToken",    "(Ljava/lang/String;)V");
    g_onToolCall = env->GetMethodID(cls, "onToolCall", "(Ljava/lang/String;Ljava/lang/String;)V");
    g_onDone     = env->GetMethodID(cls, "onDone",     "()V");
    g_onError    = env->GetMethodID(cls, "onError",    "(Ljava/lang/String;)V");
    g_onMetrics  = env->GetMethodID(cls, "onMetrics",  "(FFFIIFFFF)V");
    // onProgress is optional — don't fail if not found
    g_onProgress = env->GetMethodID(cls, "onProgress", "(F)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    // onTokenBytes is optional zero-copy fast path — don't fail if not found
    g_onTokenBytes = env->GetMethodID(cls, "onTokenBytes", "([BI)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    // onVlmStageMetrics is optional — VLM-only per-stage timings
    g_onVlmStageMetrics = env->GetMethodID(cls, "onVlmStageMetrics", "(FFI)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(cls);
    return g_onToken && g_onDone && g_onError;
}

// resolve and cache EmbeddingCallback method IDs
static bool ensure_embed_callback_methods(JNIEnv * env, jobject callback) {
    jclass cls = env->GetObjectClass(callback);
    if (g_embed_cb_class && env->IsSameObject(cls, g_embed_cb_class)) {
        env->DeleteLocalRef(cls);
        return true;
    }
    if (g_embed_cb_class) env->DeleteGlobalRef(g_embed_cb_class);
    g_embed_cb_class = (jclass)env->NewGlobalRef(cls);
    g_embed_onComplete = env->GetMethodID(cls, "onComplete", "(Lcom/dark/gguf_lib/models/EmbeddingResult;)V");
    g_embed_onError    = env->GetMethodID(cls, "onError",    "(Ljava/lang/String;)V");
    env->DeleteLocalRef(cls);
    return g_embed_onComplete && g_embed_onError;
}

// Global engine state (singleton — one model at a time, matches AAR behavior)

static struct {
    llama_model   * model   = nullptr;
    llama_context * ctx     = nullptr;
    common_sampler * sampler = nullptr;

    common_chat_templates_ptr chat_templates;

    // Sampling params (set via nativeSetSampling, updated via nativeUpdateSamplerParams)
    common_params_sampling sampling_params;

    // Config
    std::string system_prompt;
    std::string chat_template_override;

    // Tool calling
    std::string tools_json;
    int grammar_mode = 1; // 0=STRICT, 1=LAZY
    bool typed_grammar = true;

    // Engine subsystems
    tool_manager_t     * tool_mgr  = nullptr;

    // Thread mode (0=power_saving, 1=balanced, 2=performance)
    int thread_mode = 1;

    // Control vectors
    std::vector<llama_adapter_lora *> lora_adapters;

    // Generation state
    std::atomic<bool> cancel_flag{false};
    std::mutex gen_mutex;

    // Conversation tokens for state save/load
    std::vector<llama_token> session_tokens;

    // Context position tracking
    int n_past = 0;

    // Cross-turn prompt prefix cache (multi-turn context reuse)
    std::vector<llama_token> prev_prompt_tokens;

    // System prompt token count (protected region during context shifts)
    int n_system_tokens = 0;

    // Persona logit biases (set via nativeSetLogitBias, preserved across uncensored toggle)
    std::vector<llama_logit_bias> persona_biases;

    // Cached refusal token IDs (scanned once per model load, reused across setUncensored calls)
    std::vector<int32_t> cached_refusal_ids;
    bool refusal_ids_scanned = false;

    // Disk-backed prompt cache directory (set via nativeSetPromptCacheDir)
    std::string prompt_cache_dir;

    // Thinking mode (set via nativeSetThinkingEnabled)
    bool thinking_enabled = true;

    // StreamingLLM-style KV eviction policy
    int kv_n_sink   = 4;   // attention sink tokens, never evicted
    int kv_n_window = 0;   // 0 = disabled; >0 = max recency window
    bool kv_evict_at_full = false;

} g_state;

static void kv_evict_streaming();

// Memory introspection helpers ----------------------------------------------

static float read_proc_status_mb(const char * key) {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0.f;
    char line[256];
    size_t key_len = strlen(key);
    float kb = 0.f;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, key_len) == 0 && line[key_len] == ':') {
            long v = 0;
            if (sscanf(line + key_len + 1, " %ld", &v) == 1) kb = (float)v;
            break;
        }
    }
    fclose(f);
    return kb / 1024.f;
}

static float read_mem_total_mb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0.f;
    char line[256];
    float kb = 0.f;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemTotal:", 9) == 0) {
            long v = 0;
            if (sscanf(line + 9, " %ld", &v) == 1) kb = (float)v;
            break;
        }
    }
    fclose(f);
    return kb / 1024.f;
}

static void compute_memory_metrics(float & model_mb, float & ctx_mb, float & peak_mb, float & mem_pct) {
    model_mb = g_state.model ? (float)llama_model_size(g_state.model) / (1024.f * 1024.f) : 0.f;
    ctx_mb   = g_state.ctx   ? (float)llama_state_get_size(g_state.ctx) / (1024.f * 1024.f) : 0.f;
    peak_mb  = read_proc_status_mb("VmPeak");
    float total_mb = read_mem_total_mb();
    mem_pct  = (total_mb > 0.f && peak_mb > 0.f) ? (peak_mb / total_mb) * 100.f : 0.f;
}

// Helper: GGML type string to enum

static ggml_type cache_type_from_string(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    return GGML_TYPE_Q8_0; // default
}

// Helper: Apply thread config from thread-engine to llama context.
// mode: 0=power_saving, 1=balanced, 2=performance
static void apply_thread_mode(int mode) {
    tn_thread_config cfg = tn_thread_config_for_mode((tn_thread_mode)mode);
    g_state.thread_mode = mode;

    if (g_state.ctx) {
        llama_set_n_threads(g_state.ctx,
            cfg.n_threads_generation,
            cfg.n_threads_batch);
    }

    if (cfg.pin_to_perf_cores && cfg.n_perf_core_ids > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int i = 0; i < cfg.n_perf_core_ids; i++) {
            CPU_SET(cfg.perf_core_ids[i], &set);
        }
        if (sched_setaffinity(0, sizeof(set), &set) == 0) {
            LOGI("Pinned to %d performance cores (mode=%d)", cfg.n_perf_core_ids, mode);
        } else {
            LOGW("sched_setaffinity failed: %s", strerror(errno));
        }
    }
}

// Helper: Rebuild sampler from current params.
// force=false skips rebuild if only simple params changed (preserves repetition penalty history).
// force=true always rebuilds (needed when grammar, logit bias, or structural params change).
static bool g_sampler_needs_rebuild = true;

static void rebuild_sampler(bool force = true) {
    if (!force && !g_sampler_needs_rebuild && g_state.sampler) {
        // sampler exists and no structural changes — skip rebuild to preserve state
        // (repetition penalty ring buffer, mirostat mu, etc.)
        return;
    }
    if (g_state.sampler) {
        common_sampler_free(g_state.sampler);
        g_state.sampler = nullptr;
    }
    if (g_state.model) {
        g_state.sampler = common_sampler_init(g_state.model, g_state.sampling_params);
    }
    g_sampler_needs_rebuild = false;
}

// Mark sampler as needing rebuild (called when structural params change)
static void mark_sampler_dirty() {
    g_sampler_needs_rebuild = true;
}

// Helper: Build chat messages from JSON

static std::vector<common_chat_msg> parse_messages_json(const std::string & messages_json) {
    std::vector<common_chat_msg> msgs;
    try {
        auto j = json::parse(messages_json);
        if (j.is_array()) {
            for (auto & msg : j) {
                common_chat_msg cm;
                cm.role = msg.value("role", "user");
                cm.content = msg.value("content", "");

                // Remap non-standard roles to "assistant"
                // The Kotlin app sends persona names (e.g. "Luna", "Nova") as the role
                // for assistant messages. Chat templates only understand:
                // "system", "user", "assistant", "tool"
                if (cm.role != "system" && cm.role != "user" &&
                    cm.role != "assistant" && cm.role != "tool") {
                    LOGI("Remapping role '%s' -> 'assistant'", cm.role.c_str());
                    cm.role = "assistant";
                }

                msgs.push_back(cm);
            }
        }
    } catch (const std::exception & e) {
        LOGE("Failed to parse messages JSON: %s", e.what());
    }
    return msgs;
}

// Common EOS strings (safety net, like ChatterUI's commonStopStrings)
// These catch model turn boundaries across all template formats

static const std::vector<std::string> COMMON_STOP_STRINGS = {
    "</s>",
    "<|end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|EOT|>",
    "<|END_OF_TURN_TOKEN|>",
    "<|end_of_turn|>",
    "<|endoftext|>",
    "<end_of_turn>",
    "<eos>",
};

// Helper: Apply chat template to build prompt + stop sequences

struct chat_template_result {
    std::string prompt;
    std::vector<std::string> stops;
    common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    // Grammar constraints for tool calling
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string> preserved_tokens;
};

static chat_template_result apply_chat_template(const std::vector<common_chat_msg> & messages, bool add_generation_prompt = true) {
    chat_template_result out;

    if (!g_state.chat_templates) {
        // Fallback: simple concatenation
        std::string prompt;
        for (auto & msg : messages) {
            if (msg.role == "system") {
                prompt += msg.content + "\n";
            } else if (msg.role == "user") {
                prompt += "User: " + msg.content + "\n";
            } else if (msg.role == "assistant") {
                prompt += "Assistant: " + msg.content + "\n";
            } else if (msg.role == "tool") {
                prompt += "Tool result: " + msg.content + "\n";
            }
        }
        if (add_generation_prompt) {
            prompt += "Assistant:";
        }
        out.prompt = prompt;
        out.stops = {"\nUser:", "\nuser:", "\n\nUser:"};
        // Add common EOS strings as safety net
        out.stops.insert(out.stops.end(), COMMON_STOP_STRINGS.begin(), COMMON_STOP_STRINGS.end());
        return out;
    }

    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;
    inputs.enable_thinking = g_state.thinking_enabled;

    // Add tools if configured
    if (!g_state.tools_json.empty()) {
        try {
            auto tools_j = json::parse(g_state.tools_json);
            inputs.tools = common_chat_tools_parse_oaicompat(tools_j);
            if (g_state.grammar_mode == 0) {
                inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            }
        } catch (...) {
            LOGW("Failed to parse tools JSON for template");
        }
    }

    auto result = common_chat_templates_apply(g_state.chat_templates.get(), inputs);
    out.prompt = result.prompt;
    out.format = result.format;
    out.grammar = result.grammar;
    out.grammar_lazy = result.grammar_lazy;
    out.grammar_triggers = result.grammar_triggers;
    out.preserved_tokens = result.preserved_tokens;

    // If the template returned CONTENT_ONLY even though tools are configured,
    // the model's template doesn't natively support tools.
    // Inject our ToolManager's tool description as a fallback via prompt engineering.
    if (out.format == COMMON_CHAT_FORMAT_CONTENT_ONLY &&
        !g_state.tools_json.empty() && g_state.tool_mgr) {
        char * tool_prompt = tool_manager_get_prompt(g_state.tool_mgr);
        if (tool_prompt && tool_prompt[0]) {
            // Rebuild prompt with tool descriptions injected into the system message
            std::vector<common_chat_msg> augmented = messages;
            bool found_system = false;
            for (auto & m : augmented) {
                if (m.role == "system") {
                    m.content += "\n\n" + std::string(tool_prompt);
                    found_system = true;
                    break;
                }
            }
            if (!found_system) {
                augmented.insert(augmented.begin(), {"system", std::string(tool_prompt)});
            }

            // Re-apply template with augmented messages (no tools this time, we handle it via prompt)
            common_chat_templates_inputs aug_inputs;
            aug_inputs.messages = augmented;
            aug_inputs.add_generation_prompt = add_generation_prompt;
            aug_inputs.use_jinja = true;
            auto aug_result = common_chat_templates_apply(g_state.chat_templates.get(), aug_inputs);
            out.prompt = aug_result.prompt;
            LOGI("ToolManager prompt injected (model template doesn't support tools natively)");
        }
        tool_manager_free_string(tool_prompt);
    }

    // Collect stop sequences from template
    out.stops = result.additional_stops;

    // Always add common EOS strings as safety net (like ChatterUI)
    out.stops.insert(out.stops.end(), COMMON_STOP_STRINGS.begin(), COMMON_STOP_STRINGS.end());

    LOGI("Template applied: format=%d, %zu stop sequences", (int)out.format, out.stops.size());
    return out;
}

// Antiprompt detector (two-phase: full match + partial match buffering)
// Modeled after ChatterUI / llama.rn's findStoppingStrings approach

enum stop_type { STOP_FULL, STOP_PARTIAL };

struct antiprompt_state {
    std::vector<std::string> stops;
    std::string stopping_word;
    bool stopped = false;

    void set_stops(const std::vector<std::string> & s) {
        stops = s;
        stopping_word.clear();
        stopped = false;
    }

    // Check for a full stop string in the tail of the text.
    // Only searches within the region that could contain the stop string
    // (last token_size + max_stop_len chars).
    size_t find_stop(const std::string & text, size_t last_token_size, stop_type type) {
        size_t stop_pos = std::string::npos;

        for (auto & word : stops) {
            if (word.empty()) continue;
            size_t pos;

            if (type == STOP_FULL) {
                // Search only in the tail region that could contain the stop string
                size_t window = word.size() + last_token_size;
                size_t from = text.size() > window ? text.size() - window : 0;
                pos = text.find(word, from);
            } else {
                // Check if the end of text is a prefix of this stop string
                pos = find_partial(word, text);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_FULL) {
                    stopping_word = word;
                    stopped = true;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

private:
    // Check if the end of text is a partial match (prefix) of a stop string
    size_t find_partial(const std::string & word, const std::string & text) {
        if (text.empty() || word.empty()) return std::string::npos;

        // Check if text ends with any prefix of word (length 1..word.size()-1)
        size_t max_check = std::min(word.size() - 1, text.size());
        for (size_t len = max_check; len >= 1; len--) {
            if (text.compare(text.size() - len, len, word, 0, len) == 0) {
                return text.size() - len;
            }
        }
        return std::string::npos;
    }
};

// Helper: UTF-8 sanitizer with ASCII fast-path + batched JNI token sender.

static std::string g_utf8_buffer; // persistent buffer for incomplete UTF-8 bytes

// Fast check: returns true if all bytes are printable ASCII (0x01..0x7F).
// Most English LLM tokens pass this, skipping the expensive full validation.
static inline bool is_all_ascii(const char * data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if ((unsigned char)data[i] >= 0x80 || data[i] == 0x00) return false;
    }
    return true;
}

// Sanitize a string to contain only valid, complete UTF-8 sequences.
// Invalid bytes and overlong encodings are dropped. Incomplete trailing
// sequences are moved to g_utf8_buffer for the next call.
static std::string sanitize_utf8(const std::string & input) {
    // Fast path: pure ASCII needs no validation
    if (g_utf8_buffer.empty() && is_all_ascii(input.data(), input.size())) {
        return input;
    }

    std::string out;
    out.reserve(input.size());
    size_t i = 0;
    size_t len = input.size();

    while (i < len) {
        unsigned char c = (unsigned char)input[i];

        if (c == 0x00) { i++; continue; }

        if (c < 0x80) {
            out += (char)c;
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= len) {
                g_utf8_buffer.assign(input, i, len - i);
                return out;
            }
            unsigned char c1 = (unsigned char)input[i + 1];
            if ((c1 & 0xC0) != 0x80 || c < 0xC2) { i++; continue; }
            out.append(input, i, 2);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= len) {
                g_utf8_buffer.assign(input, i, len - i);
                return out;
            }
            unsigned char c1 = (unsigned char)input[i + 1];
            unsigned char c2 = (unsigned char)input[i + 2];
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) { i++; continue; }
            uint32_t cp = ((c & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
            if (cp < 0x0800 || (cp >= 0xD800 && cp <= 0xDFFF)) { i++; continue; }
            out.append(input, i, 3);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= len) {
                g_utf8_buffer.assign(input, i, len - i);
                return out;
            }
            unsigned char c1 = (unsigned char)input[i + 1];
            unsigned char c2 = (unsigned char)input[i + 2];
            unsigned char c3 = (unsigned char)input[i + 3];
            if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) { i++; continue; }
            uint32_t cp = ((c & 0x07) << 18) | ((c1 & 0x3F) << 12) |
                          ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            if (cp < 0x10000 || cp > 0x10FFFF) { i++; continue; }
            out.append(input, i, 4);
            i += 4;
        } else {
            i++;
        }
    }
    return out;
}

// Wrapper: create a JNI string that's guaranteed safe (one-shot, no buffering).
static jstring safe_new_string_utf(JNIEnv * env, const char * text) {
    if (!text || !text[0]) return env->NewStringUTF("");
    // Fast path: if pure ASCII, skip sanitize entirely
    size_t len = strlen(text);
    if (is_all_ascii(text, len)) {
        jstring result = env->NewStringUTF(text);
        if (!result) { env->ExceptionClear(); return env->NewStringUTF("?"); }
        return result;
    }
    std::string saved = std::move(g_utf8_buffer);
    g_utf8_buffer.clear();
    std::string clean = sanitize_utf8(text);
    g_utf8_buffer.clear(); // no buffering for one-shot
    if (!saved.empty()) g_utf8_buffer = std::move(saved);
    if (clean.empty()) return env->NewStringUTF("");
    jstring result = env->NewStringUTF(clean.c_str());
    if (!result) { env->ExceptionClear(); return env->NewStringUTF("?"); }
    return result;
}

// Batched token sender: accumulates text and only crosses JNI boundary
// when the buffer reaches a threshold or on explicit flush.
// This dramatically reduces per-token JNI overhead.
// Batch threshold before flushing to JNI/AIDL. Larger = fewer Binder transactions.
// 64 for direct in-process JNI, 256+ for AIDL service (Binder IPC ~20-50µs per call).
static size_t g_token_batch_threshold = 256;

// Pre-allocated byte array for zero-copy token delivery.
// Reused across all flushes — avoids per-flush jstring alloc/free overhead.
static jbyteArray g_token_byte_buf = nullptr;
static int        g_token_byte_cap = 0;

// ensure the pre-allocated byte buffer is large enough
static void ensure_token_byte_buf(JNIEnv * env, int needed) {
    if (g_token_byte_buf && g_token_byte_cap >= needed) return;
    if (g_token_byte_buf) env->DeleteGlobalRef(g_token_byte_buf);
    int cap = std::max(needed, 4096);
    jbyteArray local = env->NewByteArray(cap);
    g_token_byte_buf = (jbyteArray)env->NewGlobalRef(local);
    env->DeleteLocalRef(local);
    g_token_byte_cap = cap;
}

struct token_batcher {
    std::string buf;
    JNIEnv * env;
    jobject callback;
    jmethodID onToken;

    token_batcher(JNIEnv * e, jobject cb, jmethodID m)
        : env(e), callback(cb), onToken(m) { buf.reserve(256); }

    // Add text to the batch. Flushes to JNI if threshold reached.
    bool add(const char * text, size_t len) {
        buf.append(text, len);
        if (buf.size() >= g_token_batch_threshold) {
            return flush();
        }
        return true;
    }

    bool add(const std::string & text) { return add(text.data(), text.size()); }

    // Flush buffered text to JNI callback. Returns false if JNI exception.
    bool flush() {
        if (buf.empty()) return true;

        // Prepend any leftover UTF-8 bytes from previous flush
        std::string combined;
        if (!g_utf8_buffer.empty()) {
            combined = std::move(g_utf8_buffer);
            g_utf8_buffer.clear();
            combined += buf;
        } else {
            combined = std::move(buf);
        }
        buf.clear();

        std::string clean = sanitize_utf8(combined);
        if (clean.empty()) return true;

        // fast path: reuse pre-allocated jbyteArray (no alloc per flush)
        if (g_onTokenBytes) {
            int len = (int)clean.size();
            ensure_token_byte_buf(env, len);
            env->SetByteArrayRegion(g_token_byte_buf, 0, len, (const jbyte *)clean.data());
            env->CallVoidMethod(callback, g_onTokenBytes, g_token_byte_buf, (jint)len);
            return !env->ExceptionCheck();
        }

        // fallback: allocate jstring per flush
        jstring jtoken = env->NewStringUTF(clean.c_str());
        if (!jtoken) {
            env->ExceptionClear();
            g_utf8_buffer.clear();
            return false;
        }
        env->CallVoidMethod(callback, onToken, jtoken);
        env->DeleteLocalRef(jtoken);
        return true;
    }
};

// Helper: Tokenize a string

static std::vector<llama_token> tokenize_string(const std::string & text, bool add_special = true) {
    if (!g_state.model) return {};
    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    int n_tokens = text.size() + 256;
    std::vector<llama_token> tokens(n_tokens);
    int n = llama_tokenize(vocab, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), add_special, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), add_special, true);
    }
    tokens.resize(std::max(0, n));
    return tokens;
}

// Helper: Decode batch of tokens

// Reusable batch for prompt evaluation (avoids repeated alloc/free)
static llama_batch g_prompt_batch = {};
static int g_prompt_batch_cap = 0;

// Reusable single-token batch for generation loop (avoids per-token alloc/free)
static llama_batch g_single_batch = {};
static bool g_single_batch_init = false;

static llama_batch & get_single_batch() {
    if (!g_single_batch_init) {
        g_single_batch = llama_batch_init(1, 0, 1);
        g_single_batch_init = true;
    }
    return g_single_batch;
}

// progress_fn: optional callback invoked after each batch chunk with progress ratio (0.0-1.0).
// Used to report prompt evaluation progress to the Kotlin side during long prompts.
typedef void (*eval_progress_fn)(float progress, void * user_data);

static bool eval_tokens(const std::vector<llama_token> & tokens, int & n_past,
                         eval_progress_fn progress = nullptr, void * progress_data = nullptr) {
    if (tokens.empty()) return true;

    const int n_batch = llama_n_batch(g_state.ctx);

    // grow the reusable batch if needed
    if (g_prompt_batch_cap < n_batch) {
        if (g_prompt_batch_cap > 0) llama_batch_free(g_prompt_batch);
        g_prompt_batch = llama_batch_init(n_batch, 0, 1);
        g_prompt_batch_cap = n_batch;
    }

    int total = (int)tokens.size();
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int n_eval = std::min((int)(tokens.size() - i), n_batch);

        common_batch_clear(g_prompt_batch);
        for (int j = 0; j < n_eval; j++) {
            common_batch_add(g_prompt_batch, tokens[i + j], n_past + j, {0}, false);
        }
        g_prompt_batch.logits[g_prompt_batch.n_tokens - 1] = true;

        if (llama_decode(g_state.ctx, g_prompt_batch) != 0) {
            LOGE("Failed to decode batch at position %d", n_past);
            return false;
        }

        n_past += n_eval;

        // report progress to callback (fires once per batch chunk, not per token)
        if (progress) {
            float pct = (float)(i + n_eval) / (float)total;
            progress(pct, progress_data);
        }

        // allow cancellation during long prompt evaluation
        if (g_state.cancel_flag.load()) {
            LOGI("Prompt evaluation cancelled at %d/%d tokens", n_past, total);
            return false;
        }
    }

    return true;
}

// progress callback adapter: calls JNI onProgress from eval_tokens
struct jni_progress_ctx {
    JNIEnv * env;
    jobject callback;
};

static void jni_eval_progress(float progress, void * user_data) {
    auto * ctx = (jni_progress_ctx *)user_data;
    if (g_onProgress && ctx->env && ctx->callback) {
        ctx->env->CallVoidMethod(ctx->callback, g_onProgress, progress);
    }
}

// returns the number of matching leading tokens between two sequences
static int find_common_prefix(
    const std::vector<llama_token> & a,
    const std::vector<llama_token> & b) {
    int n = std::min((int)a.size(), (int)b.size());
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return i;
    }
    return n;
}

// check if prompt tokens fit in context window
// returns 0=ok, -1=prompt exceeds n_ctx (fatal)
static int check_prompt_fits(int n_prompt_tokens, int max_gen_tokens) {
    if (!g_state.ctx) return -1;
    int n_ctx = (int)llama_n_ctx(g_state.ctx);
    if (n_prompt_tokens >= n_ctx) {
        LOGE("Prompt (%d tokens) exceeds context window (%d)", n_prompt_tokens, n_ctx);
        return -1;
    }
    if (n_prompt_tokens + max_gen_tokens > n_ctx) {
        LOGW("Prompt (%d) + max_tokens (%d) exceeds n_ctx (%d), may need context shift",
             n_prompt_tokens, max_gen_tokens, n_ctx);
    }
    return 0;
}

// shift context when n_past approaches n_ctx
// removes older half of non-system tokens, shifts positions down
// returns true if shift succeeded
static bool try_context_shift() {
    if (!g_state.ctx) return false;

    // Prefer StreamingLLM eviction when policy is active
    if (g_state.kv_n_window > 0) {
        kv_evict_streaming();
        return g_state.n_past < (int)llama_n_ctx(g_state.ctx) - 1;
    }

    llama_memory_t mem = llama_get_memory(g_state.ctx);
    if (!mem || !llama_memory_can_shift(mem)) {
        LOGW("Context shift not supported by memory backend");
        return false;
    }

    int n_keep = std::max(g_state.n_system_tokens, 4);
    int n_discard = (g_state.n_past - n_keep) / 2;
    if (n_discard <= 0) return false;

    LOGI("Context shift: n_past=%d n_keep=%d n_discard=%d", g_state.n_past, n_keep, n_discard);

    llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
    llama_memory_seq_add(mem, 0, n_keep + n_discard, g_state.n_past, -n_discard);

    g_state.n_past -= n_discard;
    g_state.prev_prompt_tokens.clear();

    LOGI("Context shift done: new n_past=%d", g_state.n_past);
    return true;
}

// StreamingLLM-style eviction: keep [0, n_sink) + tail [n_past-n_window, n_past)
static void kv_evict_streaming() {
    if (!g_state.ctx || g_state.kv_n_window <= 0) return;
    int n_past   = g_state.n_past;
    int n_sink   = g_state.kv_n_sink;
    int n_window = g_state.kv_n_window;

    if (n_past <= n_sink + n_window) return; // nothing to evict

    llama_memory_t mem = llama_get_memory(g_state.ctx);
    if (!mem) return;

    int evict_end = n_past - n_window;
    llama_memory_seq_rm(mem, 0, n_sink, evict_end);
    // shift tail positions so they're contiguous after sinks
    llama_memory_seq_add(mem, 0, evict_end, n_past, -(evict_end - n_sink));
    g_state.n_past = n_sink + n_window;
    g_state.prev_prompt_tokens.clear();
    LOGI("KV evict: n_past %d -> %d (sink=%d window=%d)", n_past, g_state.n_past, n_sink, n_window);
}

// get current context usage as a ratio (0.0 to 1.0)
static float get_context_usage() {
    if (!g_state.ctx) return 0.0f;
    int n_ctx = (int)llama_n_ctx(g_state.ctx);
    if (n_ctx <= 0) return 0.0f;
    return (float)g_state.n_past / (float)n_ctx;
}


// Generate a hash of a string for prompt cache filenames
static std::string hash_string(const std::string & s) {
    uint64_t h = 14695981039346656037ULL;
    for (char c : s) {
        h ^= (uint64_t)(unsigned char)c;
        h *= 1099511628211ULL;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}

// Try to restore system prompt KV cache from disk.
// Returns true if cache was loaded (TTFT for turn 1 is near-zero).
static bool try_restore_prompt_cache(const std::string & system_prompt,
                                      const std::vector<llama_token> & sys_tokens) {
    if (g_state.prompt_cache_dir.empty() || system_prompt.empty()) return false;
    std::string cache_path = g_state.prompt_cache_dir + "/prompt_" + hash_string(system_prompt) + ".cache";

    FILE * f = fopen(cache_path.c_str(), "r");
    if (!f) return false;
    fclose(f);

    size_t n_token_count = 0;
    std::vector<llama_token> cache_tokens(llama_n_ctx(g_state.ctx));
    bool ok = llama_state_load_file(g_state.ctx, cache_path.c_str(),
                                     cache_tokens.data(), cache_tokens.size(), &n_token_count);
    if (ok && (int)n_token_count > 0) {
        g_state.n_past = (int)n_token_count;
        g_state.prev_prompt_tokens.assign(cache_tokens.begin(), cache_tokens.begin() + n_token_count);
        LOGI("Prompt cache restored: %zu tokens from %s", n_token_count, cache_path.c_str());
        return true;
    }
    LOGW("Prompt cache file exists but failed to load: %s", cache_path.c_str());
    return false;
}

// Save system prompt KV cache to disk for future warm restarts.
static void save_prompt_cache(const std::string & system_prompt,
                               const std::vector<llama_token> & tokens, int n_tokens) {
    if (g_state.prompt_cache_dir.empty() || system_prompt.empty()) return;
    std::string cache_path = g_state.prompt_cache_dir + "/prompt_" + hash_string(system_prompt) + ".cache";
    bool ok = llama_state_save_file(g_state.ctx, cache_path.c_str(), tokens.data(), n_tokens);
    if (ok) {
        LOGI("Prompt cache saved: %d tokens to %s", n_tokens, cache_path.c_str());
    } else {
        LOGW("Failed to save prompt cache to %s", cache_path.c_str());
    }
}

// JNI: nativeLoadModel

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModel(
        JNIEnv * env, jobject,
        jstring jpath, jint nCtx, jint nThreads,
        jboolean flashAttn, jstring jCacheTypeK, jstring jCacheTypeV) {

    ensure_backend_init();

    {
        const char * peek = env->GetStringUTFChars(jpath, nullptr);
        char detail[512];
        snprintf(detail, sizeof(detail),
            "path=%s ctx=%d threads=%d flash=%d",
            peek ? peek : "<null>", (int)nCtx, (int)nThreads, (int)flashAttn);
        tn_error_set_op("loadModel", detail);
        if (peek) env->ReleaseStringUTFChars(jpath, peek);
    }

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    // Clean up any existing model
    if (g_state.sampler) { common_sampler_free(g_state.sampler); g_state.sampler = nullptr; }
    if (g_state.ctx) { llama_free(g_state.ctx); g_state.ctx = nullptr; }
    if (g_state.model) { llama_model_free(g_state.model); g_state.model = nullptr; }
    g_state.chat_templates.reset();
    g_state.n_past = 0;
    g_state.session_tokens.clear();
    g_state.prev_prompt_tokens.clear();
    g_state.n_system_tokens = 0;

    // Copy JNI strings into std::string so they survive past the JNI release.
    // Without this, the context-create error path below dereferenced freed memory.
    std::string path_s, cacheK_s, cacheV_s;
    {
        const char * path  = env->GetStringUTFChars(jpath,        nullptr);
        const char * ck    = env->GetStringUTFChars(jCacheTypeK,  nullptr);
        const char * cv    = env->GetStringUTFChars(jCacheTypeV,  nullptr);
        if (path) path_s   = path;
        if (ck)   cacheK_s = ck;
        if (cv)   cacheV_s = cv;
        if (path) env->ReleaseStringUTFChars(jpath,       path);
        if (ck)   env->ReleaseStringUTFChars(jCacheTypeK, ck);
        if (cv)   env->ReleaseStringUTFChars(jCacheTypeV, cv);
    }

    LOGI("Loading model: %s (ctx=%d threads=%d flash=%d)",
         path_s.c_str(), nCtx, nThreads, flashAttn);

    // Model params
    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;

    g_state.model = llama_model_load_from_file(path_s.c_str(), mparams);

    if (!g_state.model) {
        LOGE("Failed to load model");
        tn_error_set_last(TN_ERR_MODEL_LOAD, "ModelLoad",
            "llama_model_load_from_file returned null. Likely causes: corrupt or non-GGUF file, unsupported architecture, or out of memory.");
        return JNI_FALSE;
    }

    // Context params — split thread counts for decode (memory-bound) vs batch (compute-bound)
    auto cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 4096;

    {
        tn_thread_config cfg = tn_thread_config_for_mode((tn_thread_mode)g_state.thread_mode);
        if (nThreads > 0) {
            cparams.n_threads = nThreads;
            cparams.n_threads_batch = nThreads;
        } else {
            cparams.n_threads = cfg.n_threads_generation;
            cparams.n_threads_batch = cfg.n_threads_batch;
        }
        cparams.n_batch = cfg.n_batch;
    }

    if (flashAttn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    cparams.type_k = cache_type_from_string(cacheK_s);
    cparams.type_v = cache_type_from_string(cacheV_s);

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOGE("Failed to create context");
        char msg[256];
        snprintf(msg, sizeof(msg),
            "llama_init_from_model failed (n_ctx=%d, type_k=%s, type_v=%s). "
            "Likely out of memory — try reducing Context Size or KV cache type.",
            (int)cparams.n_ctx,
            cacheK_s.empty() ? "?" : cacheK_s.c_str(),
            cacheV_s.empty() ? "?" : cacheV_s.c_str());
        tn_error_set_last(TN_ERR_OOM, "ContextAlloc", msg);
        llama_model_free(g_state.model);
        g_state.model = nullptr;
        return JNI_FALSE;
    }

    // pin to performance cores via thread-engine
    apply_thread_mode(g_state.thread_mode);

    // initialize chat templates from model
    g_state.chat_templates = common_chat_templates_init(
        g_state.model,
        g_state.chat_template_override);

    // initialize default sampler
    rebuild_sampler();

    // warm-up pass: decode a single token to fault-in hot weight pages.
    // without this, the first real query has high TTFT from page faults.
    {
        const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
        llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            llama_batch & sb = get_single_batch();
            common_batch_clear(sb);
            common_batch_add(sb, bos, 0, {0}, true);
            llama_decode(g_state.ctx, sb);
            llama_memory_clear(llama_get_memory(g_state.ctx), true);
            LOGI("Warm-up pass complete (model pages faulted in)");
        }
    }

    LOGI("Model loaded (ctx=%d threads_gen=%d threads_batch=%d mode=%d)",
         (int)llama_n_ctx(g_state.ctx), cparams.n_threads, cparams.n_threads_batch, g_state.thread_mode);

    return JNI_TRUE;
}

// JNI: nativeLoadModelFromFd

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModelFromFd(
        JNIEnv * env, jobject thiz,
        jint fd, jint nCtx, jint nThreads,
        jboolean flashAttn, jstring jCacheTypeK, jstring jCacheTypeV) {

    if (fd < 0) {
        LOGE("Invalid file descriptor: %d", fd);
        return JNI_FALSE;
    }

    // Duplicate the fd so we own it — the Kotlin-side ParcelFileDescriptor
    // may be closed/GC'd while we're still loading.  dup() gives us an
    // independent copy that survives until we're done.
    int owned_fd = dup(fd);
    if (owned_fd < 0) {
        LOGE("dup() failed for fd %d: %s", fd, strerror(errno));
        return JNI_FALSE;
    }

    // Validate the fd is seekable (required for mmap-based GGUF loading).
    // SAF fds from pipe-based providers aren't seekable and will fail mmap.
    off_t pos = lseek(owned_fd, 0, SEEK_CUR);
    if (pos == (off_t)-1) {
        LOGE("fd %d is not seekable (SAF pipe provider?): %s", fd, strerror(errno));
        close(owned_fd);
        return JNI_FALSE;
    }

    // /proc/self/fd/<n> gives llama.cpp a path it can fopen()
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", owned_fd);

    jstring jpath = env->NewStringUTF(path);
    jboolean result = Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModel(
        env, thiz, jpath, nCtx, nThreads, flashAttn, jCacheTypeK, jCacheTypeV);
    env->DeleteLocalRef(jpath);

    close(owned_fd);
    // Do NOT close the caller's fd here — Kotlin owns it via ParcelFileDescriptor
    // and will close it when GC'd. dup() above gave us our own copy.
    return result;
}

// JNI: nativeSetSampling

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetSampling(
        JNIEnv *, jobject,
        jfloat temperature, jint topK, jfloat topP, jfloat minP,
        jint mirostat, jfloat mirostatTau, jfloat mirostatEta, jint seed) {

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    g_state.sampling_params.temp = temperature;
    g_state.sampling_params.top_k = topK;
    g_state.sampling_params.top_p = topP;
    g_state.sampling_params.min_p = minP;
    g_state.sampling_params.mirostat = mirostat;
    g_state.sampling_params.mirostat_tau = mirostatTau;
    g_state.sampling_params.mirostat_eta = mirostatEta;
    g_state.sampling_params.seed = (seed < 0) ? LLAMA_DEFAULT_SEED : (uint32_t)seed;

    // simple param changes require rebuild because common_sampler doesn't support in-place updates
    mark_sampler_dirty();
    rebuild_sampler();

    LOGI("Sampling set: temp=%.2f top_k=%d top_p=%.2f min_p=%.2f mirostat=%d seed=%d",
         temperature, topK, topP, minP, mirostat, seed);
}

// JNI: nativeSetSystemPrompt

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetSystemPrompt(
        JNIEnv * env, jobject, jstring jprompt) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    g_state.system_prompt = prompt;
    env->ReleaseStringUTFChars(jprompt, prompt);
    LOGI("System prompt set (%zu chars)", g_state.system_prompt.size());
}

// JNI: nativeSetChatTemplate

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetChatTemplate(
        JNIEnv * env, jobject, jstring jtemplate) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * tmpl = env->GetStringUTFChars(jtemplate, nullptr);
    g_state.chat_template_override = tmpl;
    env->ReleaseStringUTFChars(jtemplate, tmpl);

    {
        char detail[256];
        snprintf(detail, sizeof(detail), "len=%zu", g_state.chat_template_override.size());
        tn_error_set_op("setChatTemplate", detail);
    }

    if (g_state.model) {
        try {
            g_state.chat_templates = common_chat_templates_init(
                g_state.model, g_state.chat_template_override);
        } catch (const std::exception & e) {
            tn_error_set_last(TN_ERR_TEMPLATE, "ChatTemplate",
                std::string("Invalid chat template: ").append(e.what()).c_str());
            g_state.chat_template_override.clear();
            g_state.chat_templates = common_chat_templates_init(g_state.model, "");
        }
    }

    LOGI("Chat template override set (%zu chars)", g_state.chat_template_override.size());
}

// JNI: nativeGenerateStream (single-turn)

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGenerateStream(
        JNIEnv * env, jobject, jstring jprompt, jint maxTokens, jobject callback) {

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    if (!g_state.model || !g_state.ctx) {
        LOGE("Model not loaded");
        return JNI_FALSE;
    }

    g_state.cancel_flag = false;
    g_utf8_buffer.clear();

    const char * prompt_cstr = env->GetStringUTFChars(jprompt, nullptr);
    std::string user_prompt(prompt_cstr);
    env->ReleaseStringUTFChars(jprompt, prompt_cstr);

    // resolve and cache callback method IDs (fast path if already cached)
    if (!ensure_callback_methods(env, callback)) {
        LOGE("Failed to find callback methods");
        return JNI_FALSE;
    }

    // build prompt using chat template
    std::vector<common_chat_msg> messages;
    if (!g_state.system_prompt.empty()) {
        messages.push_back({"system", g_state.system_prompt});
    }
    messages.push_back({"user", user_prompt});

    chat_template_result tmpl_result;
    try {
        tmpl_result = apply_chat_template(messages, true);
    } catch (const std::exception & e) {
        std::string err = std::string("Chat template error: ") + e.what();
        LOGE("%s", err.c_str());
        jstring jerr = env->NewStringUTF(err.c_str());
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    } catch (...) {
        LOGE("Unknown chat template error");
        jstring jerr = env->NewStringUTF("Unknown chat template error");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    auto tokens = tokenize_string(tmpl_result.prompt, true);

    if (tokens.empty()) {
        jstring jerr = env->NewStringUTF("Failed to tokenize prompt");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // set up antiprompt detector with template stops
    antiprompt_state antiprompt;
    antiprompt.set_stops(tmpl_result.stops);

    // check prompt fits in context window
    if (check_prompt_fits((int)tokens.size(), maxTokens) == -1) {
        jstring jerr = env->NewStringUTF("Prompt exceeds context window");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // single-turn prefix caching: reuse system prompt KV cache across calls.
    // when the same system prompt is used repeatedly, only the user message changes,
    // so we skip re-evaluating the shared prefix.
    llama_memory_t mem = llama_get_memory(g_state.ctx);
    int n_common = find_common_prefix(g_state.prev_prompt_tokens, tokens);

    if (n_common > 0 && n_common <= g_state.n_past) {
        bool removed = llama_memory_seq_rm(mem, 0, n_common, -1);
        if (!removed) {
            llama_memory_clear(mem, true);
            n_common = 0;
        }
        g_state.n_past = n_common;
        LOGI("Single-turn prefix reuse: %d/%d tokens cached", n_common, (int)tokens.size());
    } else {
        llama_memory_clear(mem, true);
        g_state.n_past = 0;
        n_common = 0;
    }

    // Apply grammar constraints for tool calling if available
    bool grammar_applied = false;
    common_params_sampling saved_params;
    if (!tmpl_result.grammar.empty() && !g_state.tools_json.empty()) {
        saved_params = g_state.sampling_params; // save for restore after generation
        g_state.sampling_params.grammar = tmpl_result.grammar;
        g_state.sampling_params.grammar_lazy =
            (g_state.grammar_mode == 0) ? tmpl_result.grammar_lazy : true;
        g_state.sampling_params.grammar_triggers = tmpl_result.grammar_triggers;
        // Resolve preserved_tokens strings to token IDs
        for (auto & tok_str : tmpl_result.preserved_tokens) {
            auto ids = tokenize_string(tok_str, false);
            for (auto id : ids) {
                g_state.sampling_params.preserved_tokens.insert(id);
            }
        }
        grammar_applied = true;
        LOGI("Grammar constraints applied for tool calling (lazy=%d, %zu triggers)",
             g_state.sampling_params.grammar_lazy, tmpl_result.grammar_triggers.size());
    }

    // reset sampler (with or without grammar)
    rebuild_sampler();

    auto t_start = std::chrono::high_resolution_clock::now();

    // evaluate only the new tokens beyond the cached prefix
    std::vector<llama_token> new_tokens(tokens.begin() + g_state.n_past, tokens.end());
    int prompt_tokens = (int)new_tokens.size();

    // set up progress reporting for long prompt evaluation
    jni_progress_ctx progress_ctx = { env, callback };
    if (!new_tokens.empty() && !eval_tokens(new_tokens, g_state.n_past,
                                             jni_eval_progress, &progress_ctx)) {
        if (grammar_applied) {
            g_state.sampling_params = saved_params;
            rebuild_sampler();
        }
        jstring jerr = env->NewStringUTF("Failed to evaluate prompt");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // save prompt tokens for next call's prefix comparison
    g_state.prev_prompt_tokens = tokens;

    auto t_prompt_done = std::chrono::high_resolution_clock::now();

    // generate tokens with two-phase antiprompt detection + batched JNI callbacks
    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    int n_generated = 0;
    std::string generated_text;
    generated_text.reserve(maxTokens * 4);
    size_t sent_count = 0;

    token_batcher batcher(env, callback, g_onToken);

    while (n_generated < maxTokens && !g_state.cancel_flag.load()) {
        if (!g_state.sampler) break;

        llama_token id = common_sampler_sample(g_state.sampler, g_state.ctx, -1);
        common_sampler_accept(g_state.sampler, id, true);

        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        // Detokenize
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) {
            buf[n] = '\0';
            generated_text.append(buf, n);

            // Two-phase antiprompt detection — use indices, not substr copies
            // Phase 1: Check for FULL stop string match in unsent region
            size_t unsent_start = std::min(sent_count, generated_text.size());
            size_t unsent_len = generated_text.size() - unsent_start;

            // Build a string_view-like reference (C++17 std::string_view not available in all NDKs)
            // We pass the unsent portion directly to find_stop
            std::string unsent(generated_text.data() + unsent_start, unsent_len);

            size_t stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_FULL);
            if (stop_pos != std::string::npos) {
                // Trim at stop and flush remaining unsent text before stop
                generated_text.resize(unsent_start + stop_pos);
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                }
                batcher.flush();
                LOGI("Antiprompt hit: '%s'", antiprompt.stopping_word.c_str());
                break;
            }

            // Phase 2: Check for PARTIAL stop string match - buffer if partial
            stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_PARTIAL);
            if (stop_pos == std::string::npos) {
                // No partial match — safe to send everything unsent
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                    sent_count = generated_text.size();
                }
            }
            // else: partial match found, hold back unsent text

            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                break;
            }
        }

        // shift context if we're about to overflow
        if (g_state.n_past >= (int)llama_n_ctx(g_state.ctx) - 1) {
            if (!try_context_shift()) {
                LOGW("Context full, stopping generation");
                break;
            }
        }

        // evaluate single token using reusable batch
        llama_batch & sb = get_single_batch();
        common_batch_clear(sb);
        common_batch_add(sb, id, g_state.n_past, {0}, true);
        if (llama_decode(g_state.ctx, sb) != 0) break;
        g_state.n_past++;
        n_generated++;


    }

    // flush any remaining buffered text
    if (sent_count < generated_text.size()) {
        batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
    }
    batcher.flush();
    if (!g_utf8_buffer.empty()) {
        batcher.buf = std::move(g_utf8_buffer);
        g_utf8_buffer.clear();
        batcher.flush();
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    // restore sampling params if grammar was applied
    if (grammar_applied) {
        g_state.sampling_params = saved_params;
        rebuild_sampler();
    }

    // check for tool calls in output (two strategies: template parser + our ToolManager)
    if (g_onToolCall && !g_state.tools_json.empty()) {
        bool found_tool_call = false;

        // Strategy 1: llama.cpp template-aware parser (works with models that follow their template)
        if (g_state.chat_templates) {
            try {
                common_chat_parser_params parser_params;
                parser_params.format = tmpl_result.format;

                auto parsed = common_chat_parse(generated_text, false, parser_params);
                for (auto & tc : parsed.tool_calls) {
                    // Wrap in the format Kotlin expects
                    json wrapped;
                    wrapped["name"] = tc.name;
                    try {
                        wrapped["arguments"] = json::parse(tc.arguments);
                    } catch (...) {
                        wrapped["arguments"] = tc.arguments;
                    }
                    std::string wrapped_str = wrapped.dump();

                    jstring jname = safe_new_string_utf(env, tc.name.c_str());
                    jstring jargs = safe_new_string_utf(env, wrapped_str.c_str());
                    env->CallVoidMethod(callback, g_onToolCall, jname, jargs);
                    env->DeleteLocalRef(jname);
                    env->DeleteLocalRef(jargs);
                    found_tool_call = true;
                    LOGI("Template parsed tool call: %s args=%s", tc.name.c_str(), wrapped_str.c_str());
                }
            } catch (const std::exception & e) {
                LOGW("Template tool call parsing failed: %s", e.what());
            }
        }

        // strategy 2: our ToolManager fallback (JSON + XML + function-call)
        if (!found_tool_call && g_state.tool_mgr) {
            auto result = tool_manager_parse_output(g_state.tool_mgr, generated_text.c_str());
            if (result.is_valid) {
                json wrapped;
                wrapped["name"] = result.tool_name;
                try {
                    wrapped["arguments"] = json::parse(result.arguments_json);
                } catch (...) {
                    wrapped["arguments"] = result.arguments_json;
                }
                std::string wrapped_str = wrapped.dump();

                jstring jname = safe_new_string_utf(env, result.tool_name);
                jstring jargs = safe_new_string_utf(env, wrapped_str.c_str());
                env->CallVoidMethod(callback, g_onToolCall, jname, jargs);
                env->DeleteLocalRef(jname);
                env->DeleteLocalRef(jargs);
                tool_manager_free_string((char *)result.tool_name);
                tool_manager_free_string((char *)result.arguments_json);
                LOGI("ToolManager fallback parsed tool call: %s", wrapped_str.c_str());
            }
        }
    }

    // metrics
    float prompt_ms = std::chrono::duration<float, std::milli>(t_prompt_done - t_start).count();
    float gen_ms = std::chrono::duration<float, std::milli>(t_end - t_prompt_done).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    float tps = gen_ms > 0 ? (n_generated / (gen_ms / 1000.0f)) : 0;
    float ttft_ms = prompt_ms;
    float model_mb = 0, ctx_mb = 0, peak_mb = 0, mem_pct = 0;
    compute_memory_metrics(model_mb, ctx_mb, peak_mb, mem_pct);

    if (g_onMetrics) {
        env->CallVoidMethod(callback, g_onMetrics,
            tps, ttft_ms, total_ms,
            prompt_tokens, n_generated,
            model_mb, ctx_mb, peak_mb, mem_pct);
    }

    env->CallVoidMethod(callback, g_onDone);

    LOGI("Generation complete: %d tokens, %.1f t/s, %.1f ms total",
         n_generated, tps, total_ms);

    return JNI_TRUE;
}

// JNI: nativeGenerateStreamMultiTurn

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGenerateStreamMultiTurn(
        JNIEnv * env, jobject, jstring jmessagesJson, jint maxTokens, jobject callback) {

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    if (!g_state.model || !g_state.ctx) {
        LOGE("Model not loaded");
        return JNI_FALSE;
    }

    g_state.cancel_flag = false;
    g_utf8_buffer.clear();

    const char * msgs_cstr = env->GetStringUTFChars(jmessagesJson, nullptr);
    std::string messages_json(msgs_cstr);
    env->ReleaseStringUTFChars(jmessagesJson, msgs_cstr);

    // resolve and cache callback method IDs
    if (!ensure_callback_methods(env, callback)) {
        LOGE("Failed to find callback methods");
        return JNI_FALSE;
    }

    auto messages = parse_messages_json(messages_json);

    if (!g_state.system_prompt.empty()) {
        if (messages.empty() || messages[0].role != "system") {
            messages.insert(messages.begin(), {"system", g_state.system_prompt});
        }
    }

    chat_template_result tmpl_result;
    try {
        tmpl_result = apply_chat_template(messages, true);
    } catch (const std::exception & e) {
        std::string err = std::string("Chat template error: ") + e.what();
        LOGE("%s", err.c_str());
        jstring jerr = env->NewStringUTF(err.c_str());
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    } catch (...) {
        LOGE("Unknown chat template error");
        jstring jerr = env->NewStringUTF("Unknown chat template error");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    auto tokens = tokenize_string(tmpl_result.prompt, true);

    if (tokens.empty()) {
        jstring jerr = env->NewStringUTF("Failed to tokenize prompt");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    antiprompt_state antiprompt;
    antiprompt.set_stops(tmpl_result.stops);

    LOGI("Multi-turn prompt length: %d tokens, %zu stop seqs", (int)tokens.size(), tmpl_result.stops.size());

    // check if prompt fits in context window
    if (check_prompt_fits((int)tokens.size(), maxTokens) == -1) {
        jstring jerr = env->NewStringUTF("Prompt exceeds context window");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // context reuse: find common prefix with previous prompt
    llama_memory_t mem = llama_get_memory(g_state.ctx);
    int n_common = find_common_prefix(g_state.prev_prompt_tokens, tokens);

    if (n_common > 0 && n_common <= g_state.n_past) {
        // remove stale tokens after the common prefix
        bool removed = llama_memory_seq_rm(mem, 0, n_common, -1);
        if (!removed) {
            LOGW("Partial seq_rm failed, falling back to full clear");
            llama_memory_clear(mem, true);
            n_common = 0;
        }
        g_state.n_past = n_common;
        LOGI("Context reuse: %d/%d tokens cached, %d new tokens to eval",
             n_common, (int)tokens.size(), (int)tokens.size() - n_common);
    } else {
        // no usable prefix — try disk-backed prompt cache before full re-eval
        llama_memory_clear(mem, true);
        g_state.n_past = 0;
        n_common = 0;

        if (!g_state.system_prompt.empty()) {
            auto sys_tokens = tokenize_string(g_state.system_prompt, true);
            if (try_restore_prompt_cache(g_state.system_prompt, sys_tokens)) {
                n_common = find_common_prefix(g_state.prev_prompt_tokens, tokens);
                if (n_common > 0 && n_common <= g_state.n_past) {
                    llama_memory_seq_rm(mem, 0, n_common, -1);
                    g_state.n_past = n_common;
                    LOGI("Disk cache hit: reusing %d/%d tokens", n_common, (int)tokens.size());
                } else {
                    llama_memory_clear(mem, true);
                    g_state.n_past = 0;
                    n_common = 0;
                }
            }
        }
        if (n_common == 0) {
            LOGI("No context reuse, full re-eval of %d tokens", (int)tokens.size());
        }
    }

    // track system prompt token count on first full evaluation
    if (g_state.n_past == 0 && !messages.empty() && messages[0].role == "system") {
        try {
            auto sys_msgs = std::vector<common_chat_msg>{messages[0]};
            auto sys_tmpl = apply_chat_template(sys_msgs, false);
            auto sys_tokens = tokenize_string(sys_tmpl.prompt, true);
            g_state.n_system_tokens = (int)sys_tokens.size();
            LOGI("System prompt: %d tokens (protected during shifts)", g_state.n_system_tokens);
        } catch (const std::exception & e) {
            // Some chat templates (e.g. Qwen 3.5) require user messages —
            // fall back to tokenizing raw system content
            LOGW("Template failed for system-only count (%s), using raw tokenization", e.what());
            auto sys_tokens = tokenize_string(messages[0].content, false);
            g_state.n_system_tokens = (int)sys_tokens.size() + 4; // +4 for template overhead
            LOGI("System prompt: ~%d tokens (estimated, protected during shifts)", g_state.n_system_tokens);
        } catch (...) {
            LOGW("Template failed for system-only count, using raw tokenization");
            auto sys_tokens = tokenize_string(messages[0].content, false);
            g_state.n_system_tokens = (int)sys_tokens.size() + 4;
            LOGI("System prompt: ~%d tokens (estimated, protected during shifts)", g_state.n_system_tokens);
        }
    }

    // apply grammar constraints for tool calling if available
    bool grammar_applied = false;
    common_params_sampling saved_params;
    if (!tmpl_result.grammar.empty() && !g_state.tools_json.empty()) {
        saved_params = g_state.sampling_params;
        g_state.sampling_params.grammar = tmpl_result.grammar;
        g_state.sampling_params.grammar_lazy =
            (g_state.grammar_mode == 0) ? tmpl_result.grammar_lazy : true;
        g_state.sampling_params.grammar_triggers = tmpl_result.grammar_triggers;
        for (auto & tok_str : tmpl_result.preserved_tokens) {
            auto ids = tokenize_string(tok_str, false);
            for (auto id : ids) {
                g_state.sampling_params.preserved_tokens.insert(id);
            }
        }
        grammar_applied = true;
        LOGI("Grammar constraints applied for tool calling (lazy=%d)", g_state.sampling_params.grammar_lazy);
    }

    rebuild_sampler();

    auto t_start = std::chrono::high_resolution_clock::now();

    // only evaluate tokens beyond the cached prefix
    std::vector<llama_token> new_tokens(tokens.begin() + g_state.n_past, tokens.end());
    int prompt_tokens = (int)new_tokens.size();

    // progress reporting for long prompt evaluation
    jni_progress_ctx mt_progress = { env, callback };
    if (!new_tokens.empty() && !eval_tokens(new_tokens, g_state.n_past,
                                             jni_eval_progress, &mt_progress)) {
        if (grammar_applied) {
            g_state.sampling_params = saved_params;
            rebuild_sampler();
        }
        jstring jerr = env->NewStringUTF("Failed to evaluate prompt");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // try saving system prompt cache for future warm restarts
    if (g_state.n_past > 0 && n_common == 0 && !g_state.system_prompt.empty()) {
        save_prompt_cache(g_state.system_prompt, tokens, g_state.n_past);
    }

    g_state.prev_prompt_tokens = tokens;

    auto t_prompt_done = std::chrono::high_resolution_clock::now();

    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    int n_generated = 0;
    std::string generated_text;
    generated_text.reserve(maxTokens * 4);
    size_t sent_count = 0;



    token_batcher batcher(env, callback, g_onToken);

    while (n_generated < maxTokens && !g_state.cancel_flag.load()) {
        if (!g_state.sampler) break;

        llama_token id = common_sampler_sample(g_state.sampler, g_state.ctx, -1);
        common_sampler_accept(g_state.sampler, id, true);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) {
            buf[n] = '\0';
            generated_text.append(buf, n);

            size_t unsent_start = std::min(sent_count, generated_text.size());
            size_t unsent_len = generated_text.size() - unsent_start;
            std::string unsent(generated_text.data() + unsent_start, unsent_len);

            size_t stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_FULL);
            if (stop_pos != std::string::npos) {
                generated_text.resize(unsent_start + stop_pos);
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                }
                batcher.flush();
                break;
            }

            stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_PARTIAL);
            if (stop_pos == std::string::npos) {
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                    sent_count = generated_text.size();
                }
            }

            if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
        }

        if (g_state.n_past >= (int)llama_n_ctx(g_state.ctx) - 1) {
            if (!try_context_shift()) break;
        }

        llama_batch & sb = get_single_batch();
        common_batch_clear(sb);
        common_batch_add(sb, id, g_state.n_past, {0}, true);
        if (llama_decode(g_state.ctx, sb) != 0) break;
        g_state.n_past++;
        n_generated++;


    }

    if (sent_count < generated_text.size()) {
        batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
    }
    batcher.flush();
    if (!g_utf8_buffer.empty()) {
        batcher.buf = std::move(g_utf8_buffer);
        g_utf8_buffer.clear();
        batcher.flush();
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    if (grammar_applied) {
        g_state.sampling_params = saved_params;
        rebuild_sampler();
    }

    // check for tool calls (two strategies: template parser + our ToolManager)
    if (g_onToolCall && !g_state.tools_json.empty()) {
        bool found_tool_call = false;

        if (g_state.chat_templates) {
            try {
                common_chat_parser_params parser_params;
                parser_params.format = tmpl_result.format;
                auto parsed = common_chat_parse(generated_text, false, parser_params);
                for (auto & tc : parsed.tool_calls) {
                    json wrapped;
                    wrapped["name"] = tc.name;
                    try { wrapped["arguments"] = json::parse(tc.arguments); }
                    catch (...) { wrapped["arguments"] = tc.arguments; }
                    std::string wrapped_str = wrapped.dump();
                    jstring jname = safe_new_string_utf(env, tc.name.c_str());
                    jstring jargs = safe_new_string_utf(env, wrapped_str.c_str());
                    env->CallVoidMethod(callback, g_onToolCall, jname, jargs);
                    env->DeleteLocalRef(jname);
                    env->DeleteLocalRef(jargs);
                    found_tool_call = true;
                }
            } catch (const std::exception & e) {
                LOGW("Template tool call parsing failed: %s", e.what());
            }
        }

        if (!found_tool_call && g_state.tool_mgr) {
            auto result = tool_manager_parse_output(g_state.tool_mgr, generated_text.c_str());
            if (result.is_valid) {
                json wrapped;
                wrapped["name"] = result.tool_name;
                try { wrapped["arguments"] = json::parse(result.arguments_json); }
                catch (...) { wrapped["arguments"] = result.arguments_json; }
                std::string wrapped_str = wrapped.dump();
                jstring jname = safe_new_string_utf(env, result.tool_name);
                jstring jargs = safe_new_string_utf(env, wrapped_str.c_str());
                env->CallVoidMethod(callback, g_onToolCall, jname, jargs);
                env->DeleteLocalRef(jname);
                env->DeleteLocalRef(jargs);
                tool_manager_free_string((char *)result.tool_name);
                tool_manager_free_string((char *)result.arguments_json);
            }
        }
    }

    float prompt_ms = std::chrono::duration<float, std::milli>(t_prompt_done - t_start).count();
    float gen_ms = std::chrono::duration<float, std::milli>(t_end - t_prompt_done).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    float tps = gen_ms > 0 ? (n_generated / (gen_ms / 1000.0f)) : 0;
    float ttft_ms = prompt_ms;
    float model_mb = 0, ctx_mb = 0, peak_mb = 0, mem_pct = 0;
    compute_memory_metrics(model_mb, ctx_mb, peak_mb, mem_pct);

    if (g_onMetrics) {
        env->CallVoidMethod(callback, g_onMetrics,
            tps, ttft_ms, total_ms,
            prompt_tokens, n_generated,
            model_mb, ctx_mb, peak_mb, mem_pct);
    }

    env->CallVoidMethod(callback, g_onDone);

    LOGI("Multi-turn generation complete: %d tokens, %.1f t/s", n_generated, tps);

    return JNI_TRUE;
}

// JNI: nativeStopGeneration

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeStopGeneration(JNIEnv *, jobject) {
    g_state.cancel_flag = true;
    LOGI("Generation stop requested");
}

// JNI: nativeRelease

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    if (g_state.sampler) {
        common_sampler_free(g_state.sampler);
        g_state.sampler = nullptr;
    }
    if (g_state.ctx) {
        llama_free(g_state.ctx);
        g_state.ctx = nullptr;
    }
    if (g_state.model) {
        llama_model_free(g_state.model);
        g_state.model = nullptr;
    }
    g_state.chat_templates.reset();
    g_state.n_past = 0;
    g_state.session_tokens.clear();
    g_state.prev_prompt_tokens.clear();
    g_state.n_system_tokens = 0;
    g_state.system_prompt.clear();
    g_state.chat_template_override.clear();
    g_state.tools_json.clear();

    // clean up persona and optimization state
    g_state.persona_biases.clear();
    g_state.lora_adapters.clear();
    g_state.cached_refusal_ids.clear();
    g_state.refusal_ids_scanned = false;

    // Free reusable batches
    if (g_prompt_batch_cap > 0) {
        llama_batch_free(g_prompt_batch);
        g_prompt_batch = {};
        g_prompt_batch_cap = 0;
    }
    if (g_single_batch_init) {
        llama_batch_free(g_single_batch);
        g_single_batch = {};
        g_single_batch_init = false;
    }

    // Clean up engine subsystems
    if (g_state.tool_mgr) {
        tool_manager_free(g_state.tool_mgr);
        g_state.tool_mgr = nullptr;
    }
    LOGI("Model released");
}

// JNI: nativeGetModelInfo

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetModelInfo(JNIEnv * env, jobject) {
    if (!g_state.model) return nullptr;

    try {
        json info;

        // Get model description
        char desc[256] = {0};
        llama_model_desc(g_state.model, desc, sizeof(desc));
        info["description"] = desc;

        // Context size
        if (g_state.ctx) {
            info["n_ctx"] = (int)llama_n_ctx(g_state.ctx);
        }

        // Model size
        info["n_params"] = (int64_t)llama_model_n_params(g_state.model);
        info["model_size"] = (int64_t)llama_model_size(g_state.model);

        // Get metadata via llama_model_meta_val_str
        char buf[256];
        if (llama_model_meta_val_str(g_state.model, "general.name", buf, sizeof(buf)) > 0) {
            info["name"] = buf;
        }
        if (llama_model_meta_val_str(g_state.model, "general.architecture", buf, sizeof(buf)) > 0) {
            info["architecture"] = buf;
        }
        if (llama_model_meta_val_str(g_state.model, "general.file_type", buf, sizeof(buf)) > 0) {
            info["file_type"] = buf;
        }

        // Vocab info
        const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
        if (vocab) {
            info["n_vocab"] = llama_vocab_n_tokens(vocab);
        }

        std::string json_str = info.dump();
        return safe_new_string_utf(env, json_str.c_str());
    } catch (const std::exception & e) {
        LOGE("Failed to get model info: %s", e.what());
        return nullptr;
    }
}

// JNI: nativeIsToolCallingSupported

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeIsToolCallingSupported(JNIEnv *, jobject) {
    if (!g_state.model) return JNI_FALSE;

    // Check if model has a chat template (indicates tool calling support)
    if (g_state.chat_templates) {
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// JNI: nativeSetToolsJson

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetToolsJson(
        JNIEnv * env, jobject, jstring jtoolsJson) {
    const char * json_cstr = env->GetStringUTFChars(jtoolsJson, nullptr);
    g_state.tools_json = json_cstr;
    env->ReleaseStringUTFChars(jtoolsJson, json_cstr);

    // Also register tools with our ToolManager for fallback multi-format parsing
    if (g_state.tools_json.empty()) {
        LOGI("Tools JSON set (empty)");
        return;
    }

    if (!g_state.tool_mgr) {
        g_state.tool_mgr = tool_manager_create();
    }
    tool_manager_clear(g_state.tool_mgr);

    try {
        auto tools_j = json::parse(g_state.tools_json);
        if (tools_j.is_array()) {
            for (auto & t : tools_j) {
                std::string name = t.value("name", "");
                std::string desc;

                // Handle OpenAI-style {"type":"function","function":{...}} format
                if (t.contains("function") && t["function"].is_object()) {
                    auto & func = t["function"];
                    name = func.value("name", name);
                    desc = func.value("description", "");
                } else {
                    desc = t.value("description", "");
                }

                if (name.empty()) continue;

                // Build param defs from JSON schema
                std::vector<tool_param_def> params;
                std::vector<std::string> param_names; // keep strings alive
                std::vector<std::string> param_descs;

                json props;
                std::vector<std::string> required_params;

                if (t.contains("function") && t["function"].contains("parameters")) {
                    auto & schema = t["function"]["parameters"];
                    if (schema.contains("properties")) props = schema["properties"];
                    if (schema.contains("required") && schema["required"].is_array()) {
                        for (auto & r : schema["required"]) required_params.push_back(r.get<std::string>());
                    }
                } else if (t.contains("parameters")) {
                    auto & schema = t["parameters"];
                    if (schema.contains("properties")) props = schema["properties"];
                    if (schema.contains("required") && schema["required"].is_array()) {
                        for (auto & r : schema["required"]) required_params.push_back(r.get<std::string>());
                    }
                }

                for (auto & [pname, pval] : props.items()) {
                    param_names.push_back(pname);
                    param_descs.push_back(pval.value("description", ""));

                    tool_param_type ptype = TOOL_PARAM_STRING;
                    std::string type_str = pval.value("type", "string");
                    if (type_str == "number" || type_str == "integer") ptype = TOOL_PARAM_NUMBER;
                    else if (type_str == "boolean") ptype = TOOL_PARAM_BOOLEAN;
                    else if (type_str == "array") ptype = TOOL_PARAM_ARRAY;
                    else if (type_str == "object") ptype = TOOL_PARAM_OBJECT;

                    bool is_required = false;
                    for (auto & r : required_params) {
                        if (r == pname) { is_required = true; break; }
                    }

                    params.push_back({
                        param_names.back().c_str(),
                        param_descs.back().c_str(),
                        ptype,
                        is_required
                    });
                }

                tool_def td;
                td.name = name.c_str();
                td.description = desc.c_str();
                td.params = params.empty() ? nullptr : params.data();
                td.n_params = (int32_t)params.size();
                tool_manager_register(g_state.tool_mgr, &td);
            }
        }
    } catch (const std::exception & e) {
        LOGW("Failed to register tools with ToolManager: %s", e.what());
    }

    LOGI("Tools JSON set (%zu chars), ToolManager registered", g_state.tools_json.size());
}

// JNI: nativeSetGrammarMode

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetGrammarMode(JNIEnv *, jobject, jint mode) {
    g_state.grammar_mode = mode;
    LOGI("Grammar mode set to %d", mode);
}

// JNI: nativeSetTypedGrammar

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetTypedGrammar(JNIEnv *, jobject, jboolean enabled) {
    g_state.typed_grammar = enabled;
    LOGI("Typed grammar %s", enabled ? "enabled" : "disabled");
}

// JNI: nativeUpdateSamplerParams

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeUpdateSamplerParams(
        JNIEnv * env, jobject, jstring jparamsJson) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * json_cstr = env->GetStringUTFChars(jparamsJson, nullptr);
    std::string json_str(json_cstr);
    env->ReleaseStringUTFChars(jparamsJson, json_cstr);

    {
        char detail[512];
        snprintf(detail, sizeof(detail), "json=%.480s", json_str.c_str());
        tn_error_set_op("updateSamplerParams", detail);
    }

    try {
        auto j = json::parse(json_str);

        if (j.contains("temperature"))     g_state.sampling_params.temp = j["temperature"].get<float>();
        if (j.contains("topK"))            g_state.sampling_params.top_k = j["topK"].get<int>();
        if (j.contains("top_k"))           g_state.sampling_params.top_k = j["top_k"].get<int>();
        if (j.contains("topP"))            g_state.sampling_params.top_p = j["topP"].get<float>();
        if (j.contains("top_p"))           g_state.sampling_params.top_p = j["top_p"].get<float>();
        if (j.contains("minP"))            g_state.sampling_params.min_p = j["minP"].get<float>();
        if (j.contains("min_p"))           g_state.sampling_params.min_p = j["min_p"].get<float>();
        if (j.contains("mirostat"))        g_state.sampling_params.mirostat = j["mirostat"].get<int>();
        if (j.contains("mirostatTau"))     g_state.sampling_params.mirostat_tau = j["mirostatTau"].get<float>();
        if (j.contains("mirostat_tau"))    g_state.sampling_params.mirostat_tau = j["mirostat_tau"].get<float>();
        if (j.contains("mirostatEta"))     g_state.sampling_params.mirostat_eta = j["mirostatEta"].get<float>();
        if (j.contains("mirostat_eta"))    g_state.sampling_params.mirostat_eta = j["mirostat_eta"].get<float>();
        if (j.contains("seed"))            g_state.sampling_params.seed = j["seed"].get<uint32_t>();
        if (j.contains("repeatPenalty"))   g_state.sampling_params.penalty_repeat = j["repeatPenalty"].get<float>();
        if (j.contains("repeat_penalty"))  g_state.sampling_params.penalty_repeat = j["repeat_penalty"].get<float>();
        if (j.contains("frequencyPenalty"))g_state.sampling_params.penalty_freq = j["frequencyPenalty"].get<float>();
        if (j.contains("frequency_penalty"))g_state.sampling_params.penalty_freq = j["frequency_penalty"].get<float>();
        if (j.contains("presencePenalty")) g_state.sampling_params.penalty_present = j["presencePenalty"].get<float>();
        if (j.contains("presence_penalty"))g_state.sampling_params.penalty_present = j["presence_penalty"].get<float>();
        if (j.contains("penaltyLastN"))    g_state.sampling_params.penalty_last_n = j["penaltyLastN"].get<int>();
        if (j.contains("penalty_last_n"))  g_state.sampling_params.penalty_last_n = j["penalty_last_n"].get<int>();

        // DRY sampler params
        if (j.contains("dryMultiplier"))   g_state.sampling_params.dry_multiplier = j["dryMultiplier"].get<float>();
        if (j.contains("dry_multiplier"))  g_state.sampling_params.dry_multiplier = j["dry_multiplier"].get<float>();
        if (j.contains("dryBase"))         g_state.sampling_params.dry_base = j["dryBase"].get<float>();
        if (j.contains("dry_base"))        g_state.sampling_params.dry_base = j["dry_base"].get<float>();
        if (j.contains("dryAllowedLength"))g_state.sampling_params.dry_allowed_length = j["dryAllowedLength"].get<int>();
        if (j.contains("dryPenaltyLastN")) g_state.sampling_params.dry_penalty_last_n = j["dryPenaltyLastN"].get<int>();

        // XTC sampler params
        if (j.contains("xtcProbability"))  g_state.sampling_params.xtc_probability = j["xtcProbability"].get<float>();
        if (j.contains("xtc_probability")) g_state.sampling_params.xtc_probability = j["xtc_probability"].get<float>();
        if (j.contains("xtcThreshold"))    g_state.sampling_params.xtc_threshold = j["xtcThreshold"].get<float>();
        if (j.contains("xtc_threshold"))   g_state.sampling_params.xtc_threshold = j["xtc_threshold"].get<float>();

        rebuild_sampler();
        LOGI("Sampler params updated");
        return JNI_TRUE;

    } catch (const std::exception & e) {
        LOGE("Failed to parse sampler params JSON: %s", e.what());
        tn_error_set_last(TN_ERR_INVALID_PARAM, "InvalidParam", e.what());
        return JNI_FALSE;
    }
}

// JNI: nativeSetLogitBias

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetLogitBias(
        JNIEnv * env, jobject, jstring jbiasJson) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * json_cstr = env->GetStringUTFChars(jbiasJson, nullptr);
    std::string json_str(json_cstr);
    env->ReleaseStringUTFChars(jbiasJson, json_cstr);

    try {
        auto j = json::parse(json_str);

        g_state.sampling_params.logit_bias.clear();

        if (j.is_object()) {
            for (auto & [key, val] : j.items()) {
                llama_logit_bias bias;
                // Key can be token ID or token string
                try {
                    bias.token = std::stoi(key);
                } catch (...) {
                    // Try to tokenize the string to get token ID
                    auto tokens = tokenize_string(key, false);
                    if (!tokens.empty()) {
                        bias.token = tokens[0];
                    } else {
                        continue;
                    }
                }
                bias.bias = val.get<float>();
                g_state.sampling_params.logit_bias.push_back(bias);
            }
        } else if (j.is_array()) {
            for (auto & item : j) {
                if (item.contains("token") && item.contains("bias")) {
                    llama_logit_bias bias;
                    auto token_val = item["token"];
                    if (token_val.is_number()) {
                        bias.token = token_val.get<int>();
                    } else if (token_val.is_string()) {
                        auto tokens = tokenize_string(token_val.get<std::string>(), false);
                        if (!tokens.empty()) {
                            bias.token = tokens[0];
                        } else {
                            continue;
                        }
                    }
                    bias.bias = item["bias"].get<float>();
                    g_state.sampling_params.logit_bias.push_back(bias);
                }
            }
        }

        // Save a copy as persona biases so setUncensored can merge without losing them
        g_state.persona_biases = g_state.sampling_params.logit_bias;

        // Refusal biases are tracked in cached_refusal_ids and re-applied by nativeSetUncensored

        rebuild_sampler();
        LOGI("Logit bias set: %zu persona + merged", g_state.persona_biases.size());

    } catch (const std::exception & e) {
        LOGE("Failed to parse logit bias JSON: %s", e.what());
    }
}

// JNI: nativeLoadControlVectors

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadControlVectors(
        JNIEnv * env, jobject, jstring jvectorsJson) {
    if (!g_state.model || !g_state.ctx) return JNI_FALSE;

    const char * json_cstr = env->GetStringUTFChars(jvectorsJson, nullptr);
    std::string json_str(json_cstr);
    env->ReleaseStringUTFChars(jvectorsJson, json_cstr);

    try {
        auto j = json::parse(json_str);

        // Parse control vector paths and scales
        std::vector<common_control_vector_load_info> cvs;
        if (j.is_array()) {
            for (auto & item : j) {
                common_control_vector_load_info cv;
                cv.fname = item.value("path", "");
                cv.strength = item.value("scale", 1.0f);
                if (!cv.fname.empty()) {
                    cvs.push_back(cv);
                }
            }
        } else if (j.is_object()) {
            common_control_vector_load_info cv;
            cv.fname = j.value("path", "");
            cv.strength = j.value("scale", 1.0f);
            if (!cv.fname.empty()) {
                cvs.push_back(cv);
            }
        }

        if (cvs.empty()) {
            LOGW("No valid control vectors found in JSON");
            return JNI_FALSE;
        }

        // Load control vectors
        auto cvec = common_control_vector_load(cvs);
        if (cvec.n_embd == -1) {
            LOGE("Failed to load control vectors");
            return JNI_FALSE;
        }

        int n_embd = llama_model_n_embd(g_state.model);
        if (cvec.n_embd != n_embd) {
            LOGE("Control vector dimension mismatch: %d vs %d", cvec.n_embd, n_embd);
            return JNI_FALSE;
        }

        int err = llama_set_adapter_cvec(g_state.ctx,
                                          cvec.data.data(),
                                          cvec.data.size(),
                                          cvec.n_embd,
                                          -1, -1);
        if (err) {
            LOGE("Failed to apply control vector");
            return JNI_FALSE;
        }

        LOGI("Control vectors loaded and applied");
        return JNI_TRUE;

    } catch (const std::exception & e) {
        LOGE("Failed to load control vectors: %s", e.what());
        return JNI_FALSE;
    }
}

// JNI: nativeClearControlVector

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeClearControlVector(JNIEnv *, jobject) {
    if (!g_state.ctx) return;

    // Pass nullptr to clear the control vector
    llama_set_adapter_cvec(g_state.ctx, nullptr, 0, 0, -1, -1);

    LOGI("Control vector cleared");
}

// JNI: nativeGetStateSize
extern "C" JNIEXPORT jlong JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetStateSize(JNIEnv *, jobject) {
    if (!g_state.ctx) return 0;
    return (jlong)llama_state_get_size(g_state.ctx);
}

// JNI: nativeGetContextUsage
extern "C" JNIEXPORT jfloat JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetContextUsage(JNIEnv *, jobject) {
    return (jfloat)get_context_usage();
}

// JNI: nativeStateSaveToFile

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeStateSaveToFile(
        JNIEnv * env, jobject, jstring jpath) {
    if (!g_state.ctx) return JNI_FALSE;

    const char * path = env->GetStringUTFChars(jpath, nullptr);
    bool ok = llama_state_save_file(g_state.ctx, path,
                                     g_state.session_tokens.data(),
                                     g_state.session_tokens.size());
    LOGI("State save to %s: %s", path, ok ? "success" : "failed");
    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

// JNI: nativeStateLoadFromFile

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeStateLoadFromFile(
        JNIEnv * env, jobject, jstring jpath) {
    if (!g_state.ctx) return JNI_FALSE;

    const char * path = env->GetStringUTFChars(jpath, nullptr);

    size_t n_token_count = 0;
    g_state.session_tokens.resize(llama_n_ctx(g_state.ctx));

    bool ok = llama_state_load_file(g_state.ctx, path,
                                     g_state.session_tokens.data(),
                                     g_state.session_tokens.size(),
                                     &n_token_count);

    if (ok) {
        g_state.session_tokens.resize(n_token_count);
        g_state.n_past = n_token_count;
        LOGI("State loaded from %s: %zu tokens", path, n_token_count);
    } else {
        g_state.session_tokens.clear();
        LOGE("Failed to load state from %s", path);
    }

    env->ReleaseStringUTFChars(jpath, path);
    return ok ? JNI_TRUE : JNI_FALSE;
}

// Embedding Engine (separate model instance for text embeddings)

static struct {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    std::mutex mutex;
} g_embed;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadEmbeddingModel(
        JNIEnv * env, jobject,
        jstring jpath, jint nThreads, jint nCtx) {

    std::lock_guard<std::mutex> lock(g_embed.mutex);

    if (g_embed.ctx) { llama_free(g_embed.ctx); g_embed.ctx = nullptr; }
    if (g_embed.model) { llama_model_free(g_embed.model); g_embed.model = nullptr; }

    const char * path = env->GetStringUTFChars(jpath, nullptr);

    auto mparams = llama_model_default_params();
    mparams.use_mmap = true;

    g_embed.model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_embed.model) {
        LOGE("Failed to load embedding model");
        return JNI_FALSE;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 512;
    cparams.n_threads = nThreads > 0 ? nThreads : tn_thread_config_for_mode((tn_thread_mode)g_state.thread_mode).n_threads_batch;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.n_batch = 512;
    cparams.embeddings = true;

    g_embed.ctx = llama_init_from_model(g_embed.model, cparams);
    if (!g_embed.ctx) {
        LOGE("Failed to create embedding context");
        llama_model_free(g_embed.model);
        g_embed.model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Embedding model loaded (ctx=%d)", nCtx);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeEncodeText(
        JNIEnv * env, jobject,
        jstring jtext, jboolean normalize, jobject callback) {

    std::lock_guard<std::mutex> lock(g_embed.mutex);

    // resolve and cache embedding callback method IDs
    ensure_embed_callback_methods(env, callback);

    if (!g_embed.model || !g_embed.ctx) {
        jstring jerr = env->NewStringUTF("Embedding model not loaded");
        env->CallVoidMethod(callback, g_embed_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    const char * text_cstr = env->GetStringUTFChars(jtext, nullptr);
    std::string text(text_cstr);
    env->ReleaseStringUTFChars(jtext, text_cstr);

    // Tokenize using embedding model's vocab
    const llama_vocab * vocab = llama_model_get_vocab(g_embed.model);
    int n_tokens_max = text.size() + 256;
    std::vector<llama_token> tokens(n_tokens_max);
    int n = llama_tokenize(vocab, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(),
                           tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(std::max(0, n));

    if (tokens.empty()) {
        jstring jerr = env->NewStringUTF("Failed to tokenize text");
        env->CallVoidMethod(callback, g_embed_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    llama_memory_clear(llama_get_memory(g_embed.ctx), true);

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {0}, i == tokens.size() - 1);
    }

    if (llama_decode(g_embed.ctx, batch) != 0) {
        llama_batch_free(batch);
        jstring jerr = env->NewStringUTF("Failed to decode for embeddings");
        env->CallVoidMethod(callback, g_embed_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    llama_batch_free(batch);

    int n_embd = llama_model_n_embd(g_embed.model);
    const float * embd = llama_get_embeddings_seq(g_embed.ctx, 0);
    if (!embd) {
        embd = llama_get_embeddings_ith(g_embed.ctx, tokens.size() - 1);
    }

    if (!embd) {
        jstring jerr = env->NewStringUTF("Failed to get embeddings");
        env->CallVoidMethod(callback, g_embed_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    std::vector<float> result(embd, embd + n_embd);
    if (normalize) {
        float norm = 0.0f;
        for (float v : result) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float & v : result) v /= norm;
        }
    }

    jclass resultClass = env->FindClass("com/dark/gguf_lib/models/EmbeddingResult");
    if (!resultClass) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        LOGE("EmbeddingResult class not found — likely R8 stripped or wrong classloader");
        tn_error_set_last(TN_ERR_UNKNOWN, "EncodeText",
            "EmbeddingResult class not found at runtime");
        return JNI_FALSE;
    }
    jmethodID resultCtor = env->GetMethodID(resultClass, "<init>", "([F)V");
    if (!resultCtor) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(resultClass);
        LOGE("EmbeddingResult constructor signature mismatch");
        return JNI_FALSE;
    }
    jfloatArray jembd = env->NewFloatArray(n_embd);
    env->SetFloatArrayRegion(jembd, 0, n_embd, result.data());
    jobject resultObj = env->NewObject(resultClass, resultCtor, jembd);

    env->CallVoidMethod(callback, g_embed_onComplete, resultObj);

    env->DeleteLocalRef(jembd);
    env->DeleteLocalRef(resultObj);
    env->DeleteLocalRef(resultClass);

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeReleaseEmbeddingModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_embed.mutex);

    if (g_embed.ctx) { llama_free(g_embed.ctx); g_embed.ctx = nullptr; }
    if (g_embed.model) { llama_model_free(g_embed.model); g_embed.model = nullptr; }

    LOGI("Embedding model released");
}

// Character / Personality JNI bindings — implemented directly via sampler params

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetPersonality(
        JNIEnv * env, jobject, jstring jparamsJson) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * json_cstr = env->GetStringUTFChars(jparamsJson, nullptr);
    std::string json_str(json_cstr);
    env->ReleaseStringUTFChars(jparamsJson, json_cstr);

    try {
        auto j = json::parse(json_str);
        // Map personality traits directly to sampler params — no separate character engine needed
        g_state.sampling_params.temp          = j.value("temperature", 0.7f);
        g_state.sampling_params.top_p         = j.value("topP", j.value("top_p", 0.9f));
        g_state.sampling_params.penalty_repeat = j.value("repetitionPenalty", j.value("repetition_penalty", 1.1f));
        // creativity → min_p (higher creativity = lower min_p filter = more token diversity)
        float creativity = j.value("creativity", 0.5f);
        g_state.sampling_params.min_p = 0.1f - creativity * 0.08f; // 0.5 → 0.06, 1.0 → 0.02
        rebuild_sampler();
        LOGI("Personality applied: temp=%.2f top_p=%.2f",
             g_state.sampling_params.temp, g_state.sampling_params.top_p);
    } catch (const std::exception & e) {
        LOGE("Failed to set personality: %s", e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetMood(JNIEnv *, jobject, jint mood) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    // 0=NEUTRAL 1=HAPPY 2=SAD 3=EXCITED 4=CALM 5=ANGRY 6=CURIOUS 7=CREATIVE 8=FOCUSED 9=CUSTOM
    static const float mood_temp[]    = { 0.7f, 0.8f, 0.6f, 0.9f, 0.5f, 0.85f, 0.75f, 0.9f, 0.6f, 0.7f };
    static const float mood_penalty[] = { 1.1f, 1.05f,1.15f,1.0f, 1.2f, 1.0f,  1.1f,  1.0f, 1.15f,1.1f };
    int m = (int)mood;
    if (m < 0 || m > 9) m = 0;
    g_state.sampling_params.temp = mood_temp[m];
    g_state.sampling_params.penalty_repeat = mood_penalty[m];
    rebuild_sampler();
    LOGI("Mood set: %d (temp=%.2f)", m, mood_temp[m]);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetCharacterContext(JNIEnv * env, jobject) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"temperature\":%.2f,\"top_p\":%.2f,\"penalty\":%.2f}",
             g_state.sampling_params.temp,
             g_state.sampling_params.top_p,
             g_state.sampling_params.penalty_repeat);
    return safe_new_string_utf(env, buf);
}

static bool g_uncensored = false;

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetUncensored(JNIEnv *, jobject, jboolean enabled) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    g_uncensored = (bool)enabled;

    if (enabled && g_state.model) {
        if (!g_state.refusal_ids_scanned) {
            const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
            g_state.cached_refusal_ids.clear();

            static const char * refusal_patterns[] = {
                "cannot", "can't", "refus", "unable", "won't",
                "Sorry", "sorry", "apologize", "instead",
                "suggest", "recommend", "encourage",
                "clarify", "elaborate", "rephrase",
                "inappropriate", "unethical", "illegal",
                "dangerous", "harmful", "unsafe", "immoral",
                "guidelines", "disclaimer", "advisable",
                "irresponsible", "unacceptable", "condemn",
                "consequences", "authorities",
                "wellbeing", "well-being", "welfare",
                "concerned", "distress",
                nullptr
            };

            int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (int32_t id = 0; id < n_vocab; id++) {
                char buf[256] = {};
                int len = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, true);
                if (len <= 0) continue;
                buf[len] = '\0';
                for (int pi = 0; refusal_patterns[pi]; pi++) {
                    if (strstr(buf, refusal_patterns[pi])) {
                        g_state.cached_refusal_ids.push_back(id);
                        break;
                    }
                }
            }
            g_state.refusal_ids_scanned = true;
            LOGI("Refusal scan: %zu tokens cached", g_state.cached_refusal_ids.size());
        }

        g_state.sampling_params.logit_bias = g_state.persona_biases;
        for (int32_t id : g_state.cached_refusal_ids) {
            llama_logit_bias lb;
            lb.token = id;
            lb.bias = -100.0f;
            g_state.sampling_params.logit_bias.push_back(lb);
        }
        rebuild_sampler();
        LOGI("Uncensored ON: suppressing %zu refusal tokens", g_state.cached_refusal_ids.size());
    } else if (!enabled) {
        g_state.sampling_params.logit_bias = g_state.persona_biases;
        rebuild_sampler();
        LOGI("Uncensored OFF");
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetUncensored(JNIEnv *, jobject) {
    return g_uncensored ? JNI_TRUE : JNI_FALSE;
}

// JNI: nativeSupportsThinking — detect from chat template

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSupportsThinking(JNIEnv *, jobject) {
    if (!g_state.chat_templates) return JNI_FALSE;
    return common_chat_templates_support_enable_thinking(g_state.chat_templates.get())
           ? JNI_TRUE : JNI_FALSE;
}

// JNI: nativeSetThinkingEnabled — enable/disable thinking in chat template

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetThinkingEnabled(JNIEnv *, jobject, jboolean enabled) {
    g_state.thinking_enabled = (enabled == JNI_TRUE);
    LOGI("Thinking %s", g_state.thinking_enabled ? "enabled" : "disabled");
}

// JNI: nativeSetThreadMode — switch thread mode at runtime (0=power_saving 1=balanced 2=performance)
extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetThreadMode(JNIEnv *, jobject, jint mode) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    if (mode < 0 || mode > 2) mode = 1;
    apply_thread_mode(mode);
    LOGI("Thread mode set: %d", mode);
}

// JNI: nativeSetTokenBatchSize — tune Binder IPC batch size for AIDL service use
// Larger = fewer IPC calls, higher latency to first visible token. Default=256.

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetTokenBatchSize(JNIEnv *, jobject, jint bytes) {
    if (bytes >= 1) g_token_batch_threshold = (size_t)bytes;
}

// JNI: nativeSetPromptCacheDir — set directory for disk-backed prompt cache

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetPromptCacheDir(
        JNIEnv * env, jobject, jstring jpath) {
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    g_state.prompt_cache_dir = path;
    env->ReleaseStringUTFChars(jpath, path);
    LOGI("Prompt cache dir set: %s", g_state.prompt_cache_dir.c_str());
}

// JNI: nativeWarmUp — run a warm-up decode pass to fault-in model weight pages

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeWarmUp(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    if (!g_state.model || !g_state.ctx) return JNI_FALSE;

    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) return JNI_FALSE;

    llama_batch & sb = get_single_batch();
    common_batch_clear(sb);
    common_batch_add(sb, bos, 0, {0}, true);
    int rc = llama_decode(g_state.ctx, sb);
    llama_memory_clear(llama_get_memory(g_state.ctx), true);
    g_state.n_past = 0;
    g_state.prev_prompt_tokens.clear();
    LOGI("Manual warm-up pass complete (rc=%d)", rc);
    return rc == 0 ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// RAG Engine JNI bindings (separate model instance for retrieval-augmented generation)
// ============================================================================

static struct {
    rag_engine_t * engine = nullptr;
    std::mutex     mutex;
} g_rag;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeCreateRagEngine(
        JNIEnv *, jobject,
        jint nThreads, jint chunkSize, jint chunkOverlap,
        jint nDims, jint topK, jint topN, jboolean lateChunking) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (g_rag.engine) {
        rag_engine_free(g_rag.engine);
        g_rag.engine = nullptr;
    }

    rag_engine_params params = rag_engine_default_params();
    if (nThreads > 0)    params.n_threads     = nThreads;
    if (chunkSize > 0)   params.chunk_size    = chunkSize;
    if (chunkOverlap >= 0) params.chunk_overlap = chunkOverlap;
    if (nDims > 0)       params.n_dims        = nDims;
    if (topK > 0)        params.top_k         = topK;
    if (topN > 0)        params.top_n         = topN;
    params.late_chunking = lateChunking;

    g_rag.engine = rag_engine_create(params);
    if (!g_rag.engine) {
        LOGE("Failed to create RAG engine");
        return JNI_FALSE;
    }

    LOGI("RAG engine created (chunks=%d, overlap=%d, dims=%d, late=%d)",
         params.chunk_size, params.chunk_overlap, params.n_dims, params.late_chunking);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadRagModel(
        JNIEnv * env, jobject, jstring jpath) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) {
        LOGE("RAG engine not created");
        return JNI_FALSE;
    }

    const char * path = env->GetStringUTFChars(jpath, nullptr);
    int32_t rc = rag_engine_load_model(g_rag.engine, path);
    env->ReleaseStringUTFChars(jpath, path);

    if (rc != 0) {
        LOGE("Failed to load RAG embedding model (rc=%d)", rc);
        return JNI_FALSE;
    }

    LOGI("RAG embedding model loaded");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadRagModelFromFd(
        JNIEnv *, jobject, jint fd) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) {
        LOGE("RAG engine not created");
        return JNI_FALSE;
    }

    int32_t rc = rag_engine_load_model_from_fd(g_rag.engine, fd);
    if (rc != 0) {
        LOGE("Failed to load RAG model from fd=%d (rc=%d)", fd, rc);
        return JNI_FALSE;
    }

    LOGI("RAG embedding model loaded from fd=%d", fd);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagIsLoaded(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);
    return (g_rag.engine && rag_engine_is_loaded(g_rag.engine)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagAddDocument(
        JNIEnv * env, jobject, jstring jtext, jstring jdocId) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine || !rag_engine_is_loaded(g_rag.engine)) {
        LOGE("RAG engine not ready for indexing");
        return -1;
    }

    const char * text = env->GetStringUTFChars(jtext, nullptr);
    const char * doc_id = env->GetStringUTFChars(jdocId, nullptr);

    int32_t n_chunks = rag_engine_add_document(g_rag.engine, text, doc_id);

    env->ReleaseStringUTFChars(jdocId, doc_id);
    env->ReleaseStringUTFChars(jtext, text);

    if (n_chunks < 0) {
        LOGE("Failed to add document to RAG index");
    } else {
        LOGI("RAG document added: %d chunks", n_chunks);
    }
    return n_chunks;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagRemoveDocument(
        JNIEnv * env, jobject, jstring jdocId) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) return -1;

    const char * doc_id = env->GetStringUTFChars(jdocId, nullptr);
    int32_t rc = rag_engine_remove_document(g_rag.engine, doc_id);
    env->ReleaseStringUTFChars(jdocId, doc_id);
    return rc;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagClear(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);
    if (g_rag.engine) rag_engine_clear(g_rag.engine);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagDocumentCount(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);
    return g_rag.engine ? rag_engine_document_count(g_rag.engine) : 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagChunkCount(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);
    return g_rag.engine ? rag_engine_chunk_count(g_rag.engine) : 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagQuery(
        JNIEnv * env, jobject, jstring jquery) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine || !rag_engine_is_loaded(g_rag.engine)) {
        return nullptr;
    }

    const char * query_cstr = env->GetStringUTFChars(jquery, nullptr);

    int32_t n_results = 0;
    rag_result * results = rag_engine_query(g_rag.engine, query_cstr, &n_results);
    env->ReleaseStringUTFChars(jquery, query_cstr);

    if (!results || n_results <= 0) {
        if (results) rag_engine_free_results(results, n_results);
        return env->NewStringUTF("[]");
    }

    // Build JSON array of results
    json arr = json::array();
    for (int32_t i = 0; i < n_results; i++) {
        arr.push_back({
            {"text",        results[i].text ? results[i].text : ""},
            {"doc_id",      results[i].doc_id ? results[i].doc_id : ""},
            {"chunk_index", results[i].chunk_index},
            {"score",       results[i].score}
        });
    }
    rag_engine_free_results(results, n_results);

    std::string json_str = arr.dump();
    return env->NewStringUTF(json_str.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagBuildPrompt(
        JNIEnv * env, jobject, jstring jquery, jstring juserPrompt) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine || !rag_engine_is_loaded(g_rag.engine)) {
        return nullptr;
    }

    const char * query = env->GetStringUTFChars(jquery, nullptr);
    const char * user_prompt = env->GetStringUTFChars(juserPrompt, nullptr);

    char * prompt = rag_engine_build_prompt(g_rag.engine, query, user_prompt);

    env->ReleaseStringUTFChars(juserPrompt, user_prompt);
    env->ReleaseStringUTFChars(jquery, query);

    if (!prompt) return nullptr;

    jstring result = env->NewStringUTF(prompt);
    rag_engine_free_string(prompt);
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagInfo(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) return nullptr;

    char * info = rag_engine_info_json(g_rag.engine);
    if (!info) return nullptr;

    jstring result = env->NewStringUTF(info);
    rag_engine_free_string(info);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeReleaseRagEngine(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (g_rag.engine) {
        rag_engine_free(g_rag.engine);
        g_rag.engine = nullptr;
    }
    LOGI("RAG engine released");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagIngestBytes(
        JNIEnv * env, jobject,
        jbyteArray jbytes, jstring jmime, jstring jname, jstring jdocId) {

    if (!jbytes) return -3;

    jsize len = env->GetArrayLength(jbytes);
    if (len <= 0) return -3;

    jbyte * raw = env->GetByteArrayElements(jbytes, nullptr);
    if (!raw) return -4;

    const char * mime = jmime ? env->GetStringUTFChars(jmime, nullptr) : nullptr;
    const char * name = jname ? env->GetStringUTFChars(jname, nullptr) : nullptr;
    const char * doc_id = env->GetStringUTFChars(jdocId, nullptr);

    char * text = nullptr;
    int rc = rag_ingest_extract(
        reinterpret_cast<const uint8_t *>(raw), (size_t) len,
        mime, name, &text);

    env->ReleaseByteArrayElements(jbytes, raw, JNI_ABORT);
    if (mime) env->ReleaseStringUTFChars(jmime, mime);
    if (name) env->ReleaseStringUTFChars(jname, name);

    if (rc != 0 || !text) {
        env->ReleaseStringUTFChars(jdocId, doc_id);
        LOGW("Ingest parse failed rc=%d", rc);
        return rc < 0 ? rc : -2;
    }

    int32_t n_chunks = -1;
    {
        std::lock_guard<std::mutex> lock(g_rag.mutex);
        if (g_rag.engine && rag_engine_is_loaded(g_rag.engine)) {
            n_chunks = rag_engine_add_document(g_rag.engine, text, doc_id);
        } else {
            LOGE("Ingest: RAG engine not ready");
            n_chunks = -6;
        }
    }

    rag_ingest_free_string(text);
    env->ReleaseStringUTFChars(jdocId, doc_id);

    if (n_chunks < 0) LOGE("Ingest indexing failed: %d", n_chunks);
    else              LOGI("Ingest indexed: %d chunks", n_chunks);

    return n_chunks;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagDetectKind(
        JNIEnv * env, jobject,
        jbyteArray jbytes, jstring jmime, jstring jname) {

    const uint8_t * ptr = nullptr;
    jsize len = 0;
    jbyte * raw = nullptr;
    if (jbytes) {
        len = env->GetArrayLength(jbytes);
        if (len > 0) {
            raw = env->GetByteArrayElements(jbytes, nullptr);
            ptr = reinterpret_cast<const uint8_t *>(raw);
        }
    }
    const char * mime = jmime ? env->GetStringUTFChars(jmime, nullptr) : nullptr;
    const char * name = jname ? env->GetStringUTFChars(jname, nullptr) : nullptr;

    int kind = (int) rag_ingest_detect_kind(ptr, (size_t) len, mime, name);

    if (raw) env->ReleaseByteArrayElements(jbytes, raw, JNI_ABORT);
    if (mime) env->ReleaseStringUTFChars(jmime, mime);
    if (name) env->ReleaseStringUTFChars(jname, name);

    return kind;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagQueryFiltered(
        JNIEnv * env, jobject, jstring jquery, jstring jdocIdPrefix) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine || !rag_engine_is_loaded(g_rag.engine)) {
        return nullptr;
    }

    const char * query_cstr = env->GetStringUTFChars(jquery, nullptr);
    const char * prefix = jdocIdPrefix ? env->GetStringUTFChars(jdocIdPrefix, nullptr) : nullptr;

    int32_t n_results = 0;
    rag_result * results = rag_engine_query_filtered(
        g_rag.engine, query_cstr, prefix, &n_results);

    env->ReleaseStringUTFChars(jquery, query_cstr);
    if (prefix) env->ReleaseStringUTFChars(jdocIdPrefix, prefix);

    if (!results || n_results <= 0) {
        if (results) rag_engine_free_results(results, n_results);
        return env->NewStringUTF("[]");
    }

    json arr = json::array();
    for (int32_t i = 0; i < n_results; i++) {
        arr.push_back({
            {"text",        results[i].text ? results[i].text : ""},
            {"doc_id",      results[i].doc_id ? results[i].doc_id : ""},
            {"chunk_index", results[i].chunk_index},
            {"score",       results[i].score}
        });
    }
    rag_engine_free_results(results, n_results);

    std::string json_str = arr.dump();
    return env->NewStringUTF(json_str.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagExtractText(
        JNIEnv * env, jobject,
        jbyteArray jbytes, jstring jmime, jstring jname) {

    if (!jbytes) return nullptr;

    jsize len = env->GetArrayLength(jbytes);
    if (len <= 0) return nullptr;

    jbyte * raw = env->GetByteArrayElements(jbytes, nullptr);
    if (!raw) return nullptr;

    const char * mime = jmime ? env->GetStringUTFChars(jmime, nullptr) : nullptr;
    const char * name = jname ? env->GetStringUTFChars(jname, nullptr) : nullptr;

    char * text = rag_engine_extract_text(
        reinterpret_cast<const uint8_t *>(raw), (int32_t) len, mime, name);

    env->ReleaseByteArrayElements(jbytes, raw, JNI_ABORT);
    if (mime) env->ReleaseStringUTFChars(jmime, mime);
    if (name) env->ReleaseStringUTFChars(jname, name);

    if (!text) return nullptr;

    jstring out = env->NewStringUTF(text);
    rag_engine_free_string(text);
    return out;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagExportIndex(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) return nullptr;

    int32_t size = 0;
    uint8_t * buf = rag_engine_export_index(g_rag.engine, &size);
    if (!buf || size <= 0) {
        if (buf) rag_engine_free_buffer(buf);
        return nullptr;
    }

    jbyteArray arr = env->NewByteArray(size);
    if (!arr) {
        rag_engine_free_buffer(buf);
        return nullptr;
    }
    env->SetByteArrayRegion(arr, 0, size, reinterpret_cast<const jbyte *>(buf));
    rag_engine_free_buffer(buf);
    LOGI("RAG index exported: %d bytes", size);
    return arr;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRagImportIndex(
        JNIEnv * env, jobject, jbyteArray jbuf) {

    std::lock_guard<std::mutex> lock(g_rag.mutex);

    if (!g_rag.engine) return -6;
    if (!jbuf) return -5;

    jsize len = env->GetArrayLength(jbuf);
    if (len <= 0) return -5;

    jbyte * raw = env->GetByteArrayElements(jbuf, nullptr);
    if (!raw) return -5;

    int32_t rc = rag_engine_import_index(
        g_rag.engine, reinterpret_cast<const uint8_t *>(raw), (int32_t) len);

    env->ReleaseByteArrayElements(jbuf, raw, JNI_ABORT);

    if (rc == 0) LOGI("RAG index imported: %d bytes", (int) len);
    else         LOGE("RAG index import failed rc=%d", rc);

    return rc;
}

// ════════════════════════════════════════════
//  AGENT ENGINE — Native orchestrator
//  Plan → Execute (tool calls) → Summarize
//  Uses the already-loaded g_state model
// ════════════════════════════════════════════

// Agent callback JNI method IDs (cached like StreamCallback)
static jclass    g_agent_cb_class       = nullptr;
static jmethodID g_agent_onPlan         = nullptr;
static jmethodID g_agent_onToolCall     = nullptr;
static jmethodID g_agent_onToolResult   = nullptr;
static jmethodID g_agent_onToken        = nullptr;
static jmethodID g_agent_onSummary      = nullptr;
static jmethodID g_agent_onComplete     = nullptr;
static jmethodID g_agent_onError        = nullptr;
static jmethodID g_agent_executeTool    = nullptr; // synchronous upcall

static bool ensure_agent_callback_methods(JNIEnv * env, jobject callback) {
    jclass cls = env->GetObjectClass(callback);
    if (g_agent_cb_class && env->IsSameObject(cls, g_agent_cb_class)) {
        env->DeleteLocalRef(cls);
        return true;
    }
    if (g_agent_cb_class) env->DeleteGlobalRef(g_agent_cb_class);
    g_agent_cb_class = (jclass)env->NewGlobalRef(cls);

    g_agent_onPlan       = env->GetMethodID(cls, "onPlan",       "(Ljava/lang/String;)V");
    g_agent_onToolCall   = env->GetMethodID(cls, "onToolCall",   "(ILjava/lang/String;Ljava/lang/String;)V");
    g_agent_onToolResult = env->GetMethodID(cls, "onToolResult", "(ILjava/lang/String;Ljava/lang/String;ZJ)V");
    g_agent_onToken      = env->GetMethodID(cls, "onToken",      "(Ljava/lang/String;Z)V");
    g_agent_onSummary    = env->GetMethodID(cls, "onSummary",    "(Ljava/lang/String;)V");
    g_agent_onComplete   = env->GetMethodID(cls, "onComplete",   "()V");
    g_agent_onError      = env->GetMethodID(cls, "onError",      "(Ljava/lang/String;)V");
    g_agent_executeTool  = env->GetMethodID(cls, "executeToolFromNative",
                                            "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");

    env->DeleteLocalRef(cls);
    return g_agent_onPlan && g_agent_onToolCall && g_agent_onToolResult &&
           g_agent_onToken && g_agent_onSummary && g_agent_onComplete &&
           g_agent_onError && g_agent_executeTool;
}

// Agent state (separate from g_state — agent orchestration only)
static struct {
    jobject  callback_ref = nullptr;  // global ref to Kotlin AgentCallback
    std::string tool_schemas_json;
    std::atomic<bool> cancel_flag{false};
    std::mutex mutex;
    bool initialized = false;
} g_agent;

// ── Agent helper: generate text and return as string ──
// Uses the existing g_state model/context. Caller must hold g_state.gen_mutex.
// If stream_to_callback is true, also streams tokens to agent callback.

static std::string agent_generate_text(
        JNIEnv * env,
        const std::vector<common_chat_msg> & messages,
        int max_tokens,
        bool use_grammar,
        bool stream_to_callback,
        bool is_summary) {

    if (!g_state.model || !g_state.ctx) return "";

    chat_template_result tmpl_result;
    try {
        // Temporarily set grammar mode for tool calling phase
        int saved_grammar_mode = g_state.grammar_mode;
        if (use_grammar && !g_state.tools_json.empty()) {
            g_state.grammar_mode = 0; // STRICT
        } else {
            // Clear tools temporarily for free-form generation
            std::string saved_tools = g_state.tools_json;
            g_state.tools_json.clear();
            tmpl_result = apply_chat_template(messages, true);
            g_state.tools_json = saved_tools;
            g_state.grammar_mode = saved_grammar_mode;
            goto after_template;
        }
        tmpl_result = apply_chat_template(messages, true);
        g_state.grammar_mode = saved_grammar_mode;
    } catch (const std::exception & e) {
        LOGE("Agent template error: %s", e.what());
        return "";
    }

after_template:
    auto tokens = tokenize_string(tmpl_result.prompt, true);
    if (tokens.empty()) return "";

    // Check prompt fits
    if (check_prompt_fits((int)tokens.size(), max_tokens) == -1) {
        LOGW("Agent prompt exceeds context window");
        return "";
    }

    // Apply grammar if needed
    bool grammar_applied = false;
    common_params_sampling saved_params;
    if (use_grammar && !tmpl_result.grammar.empty() && !g_state.tools_json.empty()) {
        saved_params = g_state.sampling_params;
        g_state.sampling_params.grammar = tmpl_result.grammar;
        g_state.sampling_params.grammar_lazy =
            (g_state.grammar_mode == 0) ? tmpl_result.grammar_lazy : true;
        g_state.sampling_params.grammar_triggers = tmpl_result.grammar_triggers;
        for (auto & tok_str : tmpl_result.preserved_tokens) {
            auto ids = tokenize_string(tok_str, false);
            for (auto id : ids) {
                g_state.sampling_params.preserved_tokens.insert(id);
            }
        }
        grammar_applied = true;
    }

    // Clear KV cache for fresh generation (agent generates independent prompts)
    llama_memory_t mem = llama_get_memory(g_state.ctx);
    llama_memory_clear(mem, true);
    g_state.n_past = 0;

    rebuild_sampler();

    // Evaluate prompt
    if (!eval_tokens(tokens, g_state.n_past, nullptr, nullptr)) {
        if (grammar_applied) {
            g_state.sampling_params = saved_params;
            rebuild_sampler();
        }
        return "";
    }

    // Set up antiprompt
    antiprompt_state antiprompt;
    antiprompt.set_stops(tmpl_result.stops);

    // Generate
    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    std::string generated;
    generated.reserve(max_tokens * 4);
    int n_generated = 0;

    while (n_generated < max_tokens && !g_agent.cancel_flag.load() && !g_state.cancel_flag.load()) {
        if (!g_state.sampler) break;

        llama_token id = common_sampler_sample(g_state.sampler, g_state.ctx, -1);
        common_sampler_accept(g_state.sampler, id, true);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) {
            buf[n] = '\0';
            generated.append(buf, n);

            // Check antiprompt
            std::string tail(generated.end() - std::min((size_t)64, generated.size()), generated.end());
            size_t stop_pos = antiprompt.find_stop(tail, (size_t)n, STOP_FULL);
            if (stop_pos != std::string::npos) {
                // Trim at stop
                size_t actual_stop = generated.size() - tail.size() + stop_pos;
                generated.resize(actual_stop);
                break;
            }

            // Stream tokens to callback if requested
            if (stream_to_callback && g_agent.callback_ref) {
                jstring jtoken = safe_new_string_utf(env, buf);
                env->CallVoidMethod(g_agent.callback_ref, g_agent_onToken,
                                    jtoken, (jboolean)is_summary);
                env->DeleteLocalRef(jtoken);
            }
        }

        // Context shift if needed
        if (g_state.n_past >= (int)llama_n_ctx(g_state.ctx) - 1) {
            if (!try_context_shift()) break;
        }

        llama_batch & sb = get_single_batch();
        common_batch_clear(sb);
        common_batch_add(sb, id, g_state.n_past, {0}, true);
        if (llama_decode(g_state.ctx, sb) != 0) break;
        g_state.n_past++;
        n_generated++;
    }

    // Restore grammar params
    if (grammar_applied) {
        g_state.sampling_params = saved_params;
        rebuild_sampler();
    }

    LOGI("Agent generated %d tokens (grammar=%d, summary=%d)", n_generated, use_grammar, is_summary);
    return generated;
}

// ── Agent helper: parse tool call from generated text ──

struct agent_tool_call {
    std::string name;
    std::string args_json;
    bool valid = false;
};

static agent_tool_call agent_parse_tool_call(const std::string & text) {
    agent_tool_call result;

    // Strategy 1: llama.cpp template-aware parser
    if (g_state.chat_templates) {
        try {
            common_chat_parser_params params;
            // Use default format
            auto parsed = common_chat_parse(text, false, params);
            if (!parsed.tool_calls.empty()) {
                auto & tc = parsed.tool_calls[0];
                result.name = tc.name;
                result.args_json = tc.arguments;
                result.valid = true;
                return result;
            }
        } catch (...) {}
    }

    // Strategy 2: ToolManager fallback
    if (g_state.tool_mgr) {
        auto tm_result = tool_manager_parse_output(g_state.tool_mgr, text.c_str());
        if (tm_result.is_valid) {
            result.name = tm_result.tool_name;
            result.args_json = tm_result.arguments_json;
            result.valid = true;
            tool_manager_free_string((char *)tm_result.tool_name);
            tool_manager_free_string((char *)tm_result.arguments_json);
            return result;
        }
    }

    // Strategy 3: Raw JSON extraction
    try {
        auto j = json::parse(text);
        if (j.contains("name") && j.contains("arguments")) {
            result.name = j["name"].get<std::string>();
            result.args_json = j["arguments"].dump();
            result.valid = true;
            return result;
        }
    } catch (...) {}

    // Strategy 4: Find JSON in text
    auto pos = text.find("{\"name\"");
    if (pos == std::string::npos) pos = text.find("{\"tool_call");
    if (pos != std::string::npos) {
        int depth = 0;
        size_t end = pos;
        for (size_t i = pos; i < text.size(); i++) {
            if (text[i] == '{') depth++;
            else if (text[i] == '}') { depth--; if (depth == 0) { end = i + 1; break; } }
        }
        if (end > pos) {
            try {
                auto j = json::parse(text.substr(pos, end - pos));
                if (j.contains("name")) {
                    result.name = j["name"].get<std::string>();
                    if (j.contains("arguments")) {
                        result.args_json = j["arguments"].dump();
                    }
                    result.valid = true;
                    return result;
                }
            } catch (...) {}
        }
    }

    return result;
}

// ── JNI: nativeInitAgentSystem ──

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeInitAgentSystem(
        JNIEnv * env, jobject, jobject callback, jstring jtoolSchemasJson) {

    std::lock_guard<std::mutex> lock(g_agent.mutex);

    if (!g_state.model || !g_state.ctx) {
        LOGE("Cannot init agent: no model loaded");
        return JNI_FALSE;
    }

    if (!ensure_agent_callback_methods(env, callback)) {
        LOGE("Failed to resolve agent callback methods");
        return JNI_FALSE;
    }

    // Store global ref to callback
    if (g_agent.callback_ref) env->DeleteGlobalRef(g_agent.callback_ref);
    g_agent.callback_ref = env->NewGlobalRef(callback);

    // Store tool schemas
    if (jtoolSchemasJson) {
        const char * schemas = env->GetStringUTFChars(jtoolSchemasJson, nullptr);
        g_agent.tool_schemas_json = schemas;
        env->ReleaseStringUTFChars(jtoolSchemasJson, schemas);
    }

    g_agent.cancel_flag = false;
    g_agent.initialized = true;

    LOGI("Agent system initialized with %zu bytes of tool schemas",
         g_agent.tool_schemas_json.size());
    return JNI_TRUE;
}

// ── JNI: nativeRunAgentStep ──

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeRunAgentStep(
        JNIEnv * env, jobject,
        jstring juserMessage, jstring jsystemPrompt, jint maxRounds) {

    // Lock both agent and generation mutexes (agent owns generation during its step)
    std::lock_guard<std::mutex> agent_lock(g_agent.mutex);
    std::lock_guard<std::mutex> gen_lock(g_state.gen_mutex);

    if (!g_agent.initialized || !g_agent.callback_ref) {
        LOGE("Agent not initialized");
        return;
    }
    if (!g_state.model || !g_state.ctx) {
        jstring jerr = safe_new_string_utf(env, "No model loaded");
        env->CallVoidMethod(g_agent.callback_ref, g_agent_onError, jerr);
        env->DeleteLocalRef(jerr);
        return;
    }

    g_agent.cancel_flag = false;

    const char * user_cstr = env->GetStringUTFChars(juserMessage, nullptr);
    std::string user_message(user_cstr);
    env->ReleaseStringUTFChars(juserMessage, user_cstr);

    const char * sys_cstr = env->GetStringUTFChars(jsystemPrompt, nullptr);
    std::string system_prompt(sys_cstr);
    env->ReleaseStringUTFChars(jsystemPrompt, sys_cstr);

    // Ensure tools are configured in g_state for grammar-constrained generation
    std::string saved_tools = g_state.tools_json;
    if (!g_agent.tool_schemas_json.empty()) {
        g_state.tools_json = g_agent.tool_schemas_json;
    }

    // ── Phase 1: Plan ──
    LOGI("Agent phase: PLAN");
    {
        std::vector<common_chat_msg> plan_msgs;
        plan_msgs.push_back({"system", system_prompt + "\n\nCreate a brief 1-2 sentence plan for this request. Do NOT call any tools yet."});
        plan_msgs.push_back({"user", user_message});

        std::string plan = agent_generate_text(env, plan_msgs, 256, false, true, false);

        if (g_agent.cancel_flag.load()) {
            g_state.tools_json = saved_tools;
            return;
        }

        if (!plan.empty()) {
            jstring jplan = safe_new_string_utf(env, plan.c_str());
            env->CallVoidMethod(g_agent.callback_ref, g_agent_onPlan, jplan);
            env->DeleteLocalRef(jplan);
        }
    }

    // ── Phase 2: Execute tool calls ──
    LOGI("Agent phase: EXECUTE (max %d rounds)", maxRounds);
    struct tool_step {
        int round;
        std::string tool_name;
        std::string args_json;
        std::string result_json;
        bool success;
        long time_ms;
    };
    std::vector<tool_step> steps;

    for (int round = 0; round < maxRounds && !g_agent.cancel_flag.load(); round++) {
        // Build multi-turn context with previous tool call results
        std::vector<common_chat_msg> exec_msgs;
        exec_msgs.push_back({"system", system_prompt + "\n\nCall the next tool needed to accomplish the user's request. If no more tools are needed, respond with a plain text summary."});
        exec_msgs.push_back({"user", user_message});

        // Inject previous tool call / result pairs as alternating assistant/tool messages
        for (auto & s : steps) {
            json tc_json;
            tc_json["name"] = s.tool_name;
            try { tc_json["arguments"] = json::parse(s.args_json); }
            catch (...) { tc_json["arguments"] = s.args_json; }
            exec_msgs.push_back({"assistant", tc_json.dump()});
            exec_msgs.push_back({"tool", s.result_json});
        }

        std::string output = agent_generate_text(env, exec_msgs, 300, true, false, false);

        if (g_agent.cancel_flag.load()) break;
        if (output.empty()) break;

        // Parse tool call
        auto parsed = agent_parse_tool_call(output);
        if (!parsed.valid) {
            LOGI("Agent round %d: no tool call found, ending execution", round);
            break;
        }

        // Notify Kotlin of tool call
        jstring jname = safe_new_string_utf(env, parsed.name.c_str());
        jstring jargs = safe_new_string_utf(env, parsed.args_json.c_str());
        env->CallVoidMethod(g_agent.callback_ref, g_agent_onToolCall,
                            (jint)round, jname, jargs);
        env->DeleteLocalRef(jname);
        env->DeleteLocalRef(jargs);

        // Execute tool via synchronous upcall to Kotlin
        auto t_start = std::chrono::steady_clock::now();

        jstring jexec_name = safe_new_string_utf(env, parsed.name.c_str());
        jstring jexec_args = safe_new_string_utf(env, parsed.args_json.c_str());
        jstring jresult = (jstring)env->CallObjectMethod(
            g_agent.callback_ref, g_agent_executeTool, jexec_name, jexec_args);
        env->DeleteLocalRef(jexec_name);
        env->DeleteLocalRef(jexec_args);

        // JNI spec: any pending exception from the upcall must be cleared before
        // making further JNI calls — otherwise the next call hits UB.
        std::string result_str;
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            if (jresult) env->DeleteLocalRef(jresult);
            result_str = "{\"error\":\"tool executor threw\"}";
        } else if (jresult) {
            const char * res_cstr = env->GetStringUTFChars(jresult, nullptr);
            result_str = res_cstr ? res_cstr : "";
            if (res_cstr) env->ReleaseStringUTFChars(jresult, res_cstr);
            env->DeleteLocalRef(jresult);
        } else {
            result_str = "{\"error\":\"null result\"}";
        }

        auto t_end = std::chrono::steady_clock::now();
        long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        bool success = result_str.find("\"error\"") == std::string::npos;
        steps.push_back({round, parsed.name, parsed.args_json, result_str, success, elapsed});

        // Notify Kotlin of tool result
        jstring jresult_str = safe_new_string_utf(env, result_str.c_str());
        jstring jresult_name = safe_new_string_utf(env, parsed.name.c_str());
        env->CallVoidMethod(g_agent.callback_ref, g_agent_onToolResult,
                            (jint)round, jresult_name, jresult_str,
                            (jboolean)success, (jlong)elapsed);
        env->DeleteLocalRef(jresult_str);
        env->DeleteLocalRef(jresult_name);

        LOGI("Agent round %d: %s → %s (%ld ms)",
             round, parsed.name.c_str(), success ? "success" : "error", elapsed);
    }

    if (g_agent.cancel_flag.load()) {
        g_state.tools_json = saved_tools;
        return;
    }

    // ── Phase 3: Summarize ──
    LOGI("Agent phase: SUMMARIZE");
    {
        std::vector<common_chat_msg> sum_msgs;
        std::string steps_text;
        for (auto & s : steps) {
            steps_text += "- " + s.tool_name + "(" + s.args_json + ") → " +
                          s.result_json.substr(0, 500) + "\n";
        }

        sum_msgs.push_back({"system", "Summarize what was accomplished. Be concise and helpful."});
        sum_msgs.push_back({"user", user_message});
        sum_msgs.push_back({"assistant", "I executed the following steps:\n" + steps_text});
        sum_msgs.push_back({"user", "Please provide a clear, concise summary of what you did and the results."});

        std::string summary = agent_generate_text(env, sum_msgs, 512, false, true, true);

        if (!summary.empty() && !g_agent.cancel_flag.load()) {
            jstring jsummary = safe_new_string_utf(env, summary.c_str());
            env->CallVoidMethod(g_agent.callback_ref, g_agent_onSummary, jsummary);
            env->DeleteLocalRef(jsummary);
        }
    }

    // Restore tools
    g_state.tools_json = saved_tools;

    // Complete
    if (!g_agent.cancel_flag.load()) {
        env->CallVoidMethod(g_agent.callback_ref, g_agent_onComplete);
    }

    LOGI("Agent step complete: %zu tool calls", steps.size());
}

// ── JNI: nativeStopAgent ──

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeStopAgent(JNIEnv *, jobject) {
    g_agent.cancel_flag = true;
    g_state.cancel_flag = true; // also stop any in-progress generation
    LOGI("Agent stop requested");
}

// ── JNI: nativeReleaseAgentSystem ──

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeReleaseAgentSystem(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_agent.mutex);

    g_agent.cancel_flag = true;
    if (g_agent.callback_ref) {
        env->DeleteGlobalRef(g_agent.callback_ref);
        g_agent.callback_ref = nullptr;
    }
    g_agent.tool_schemas_json.clear();
    g_agent.initialized = false;

    LOGI("Agent system released");
}

// ════════════════════════════════════════════
//  VLM (Vision Language Model) JNI Bridge
// ════════════════════════════════════════════

static struct {
    mtmd_context * ctx = nullptr;
    std::mutex     mutex;
} g_vlm;

// ── JNI: nativeVlmLoadProjector ──
//
// imageMinTokens / imageMaxTokens let the caller cap the mmproj token budget.
// Pass -1 for either to use the model default. Lowering imageMaxTokens reduces
// the overview resolution but does NOT cap the per-tile count for LFM2-VL;
// the tile cap is a compile-time constant in clip.cpp.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjector(
        JNIEnv * env, jobject, jstring jpath, jint nThreads,
        jint imageMinTokens, jint imageMaxTokens) {

    std::lock_guard<std::mutex> lock(g_vlm.mutex);

    if (!g_state.model) {
        LOGE("VLM: text model must be loaded first");
        return JNI_FALSE;
    }

    // release previous projector if any
    if (g_vlm.ctx) {
        mtmd_free(g_vlm.ctx);
        g_vlm.ctx = nullptr;
    }

    const char * path = env->GetStringUTFChars(jpath, nullptr);

    auto params = mtmd_context_params_default();
    params.use_gpu       = false;  // CPU only on mobile
    params.n_threads     = nThreads > 0 ? nThreads : tn_thread_config_for_mode((tn_thread_mode)g_state.thread_mode).n_threads_batch;
    params.print_timings = false;
    params.warmup        = true;
    if (imageMinTokens > 0) params.image_min_tokens = imageMinTokens;
    if (imageMaxTokens > 0) params.image_max_tokens = imageMaxTokens;

    g_vlm.ctx = mtmd_init_from_file(path, g_state.model, params);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_vlm.ctx) {
        LOGE("VLM: failed to load projector");
        return JNI_FALSE;
    }

    LOGI("VLM: projector loaded (vision=%d, audio=%d, img_tokens=[%d..%d])",
         mtmd_support_vision(g_vlm.ctx), mtmd_support_audio(g_vlm.ctx),
         imageMinTokens, imageMaxTokens);
    return JNI_TRUE;
}

// ── JNI: nativeVlmLoadProjectorFromFd ──

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjectorFromFd(
        JNIEnv * env, jobject thiz, jint fd, jint nThreads,
        jint imageMinTokens, jint imageMaxTokens) {

    if (fd < 0) {
        LOGE("VLM: invalid file descriptor: %d", fd);
        return JNI_FALSE;
    }

    // Own the fd so /proc/self/fd/<n> stays valid across the load — Kotlin's
    // ParcelFileDescriptor may close the original mid-load.
    int owned_fd = dup(fd);
    if (owned_fd < 0) {
        LOGE("VLM: dup() failed for fd %d: %s", fd, strerror(errno));
        return JNI_FALSE;
    }

    // mmap-based loading requires a seekable fd; SAF pipe providers fail this.
    if (lseek(owned_fd, 0, SEEK_CUR) == (off_t)-1) {
        LOGE("VLM: fd %d is not seekable: %s", fd, strerror(errno));
        close(owned_fd);
        return JNI_FALSE;
    }

    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", owned_fd);
    jstring jpath = env->NewStringUTF(fd_path);
    jboolean result = Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjector(
        env, thiz, jpath, nThreads, imageMinTokens, imageMaxTokens);
    env->DeleteLocalRef(jpath);

    close(owned_fd);
    return result;
}

// ── JNI: nativeVlmRelease ──

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vlm.mutex);
    if (g_vlm.ctx) {
        mtmd_free(g_vlm.ctx);
        g_vlm.ctx = nullptr;
        LOGI("VLM: projector released");
    }
}

// ── JNI: nativeVlmIsLoaded ──

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmIsLoaded(JNIEnv *, jobject) {
    return g_vlm.ctx != nullptr ? JNI_TRUE : JNI_FALSE;
}

// ── JNI: nativeVlmGetInfo ──

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmGetInfo(JNIEnv * env, jobject) {
    if (!g_vlm.ctx) return env->NewStringUTF("{}");

    json info;
    info["supports_vision"] = mtmd_support_vision(g_vlm.ctx);
    info["supports_audio"]  = mtmd_support_audio(g_vlm.ctx);
    info["default_marker"]  = mtmd_default_marker();

    std::string s = info.dump();
    return env->NewStringUTF(s.c_str());
}

// ── JNI: nativeVlmGetDefaultMarker ──

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmGetDefaultMarker(JNIEnv * env, jobject) {
    return env->NewStringUTF(mtmd_default_marker());
}

// ── JNI: nativeVlmGenerateStream ──
//
// Generates a response from text + images. The prompt should contain
// image markers (from nativeVlmGetDefaultMarker) where images go.
//
// imageDataArray: array of byte[] — each is raw file bytes (JPEG/PNG)
// This clears the KV cache and starts fresh (VLM doesn't support multi-turn context reuse).

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmGenerateStream(
        JNIEnv * env, jobject,
        jstring jmessagesJson,
        jobjectArray imageDataArray,
        jint maxTokens,
        jobject callback) {

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    if (!g_state.model || !g_state.ctx) {
        LOGE("VLM: text model not loaded");
        return JNI_FALSE;
    }
    if (!g_vlm.ctx) {
        LOGE("VLM: projector not loaded");
        return JNI_FALSE;
    }

    g_state.cancel_flag = false;
    g_utf8_buffer.clear();

    if (!ensure_callback_methods(env, callback)) {
        LOGE("VLM: failed to find callback methods");
        return JNI_FALSE;
    }

    // Parse messages JSON and apply chat template
    const char * msgs_cstr = env->GetStringUTFChars(jmessagesJson, nullptr);
    std::string messages_json(msgs_cstr);
    env->ReleaseStringUTFChars(jmessagesJson, msgs_cstr);

    auto messages = parse_messages_json(messages_json);
    if (!g_state.system_prompt.empty()) {
        if (messages.empty() || messages[0].role != "system") {
            messages.insert(messages.begin(), {"system", g_state.system_prompt});
        }
    }

    chat_template_result tmpl_result;
    try {
        tmpl_result = apply_chat_template(messages, true);
    } catch (const std::exception & e) {
        std::string err = std::string("VLM chat template error: ") + e.what();
        LOGE("%s", err.c_str());
        jstring jerr = env->NewStringUTF(err.c_str());
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // Collect image data from Java byte arrays
    int n_images = imageDataArray ? env->GetArrayLength(imageDataArray) : 0;

    struct image_buf {
        std::vector<unsigned char> data;
    };
    std::vector<image_buf> image_bufs(n_images);

    for (int i = 0; i < n_images; i++) {
        auto jbytes = (jbyteArray)env->GetObjectArrayElement(imageDataArray, i);
        int len = env->GetArrayLength(jbytes);
        image_bufs[i].data.resize(len);
        env->GetByteArrayRegion(jbytes, 0, len, (jbyte *)image_bufs[i].data.data());
        env->DeleteLocalRef(jbytes);
    }

    // Create mtmd bitmaps from image data
    std::vector<mtmd_bitmap *> bitmaps;
    for (int i = 0; i < n_images; i++) {
        mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_buf(
            g_vlm.ctx, image_bufs[i].data.data(), image_bufs[i].data.size());
        if (!bmp) {
            LOGE("VLM: failed to decode image %d", i);
            for (auto * b : bitmaps) mtmd_bitmap_free(b);
            jstring jerr = env->NewStringUTF("Failed to decode image");
            env->CallVoidMethod(callback, g_onError, jerr);
            env->DeleteLocalRef(jerr);
            return JNI_FALSE;
        }
        bitmaps.push_back(bmp);
    }

    // Build const pointer array for mtmd_tokenize
    std::vector<const mtmd_bitmap *> bitmap_ptrs(bitmaps.begin(), bitmaps.end());

    // Tokenize prompt + images into chunks
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text;
    input_text.text         = tmpl_result.prompt.c_str();
    input_text.add_special  = true;
    input_text.parse_special = true;

    int32_t tok_result = mtmd_tokenize(g_vlm.ctx, chunks,
        &input_text, bitmap_ptrs.data(), bitmap_ptrs.size());

    for (auto * b : bitmaps) mtmd_bitmap_free(b);

    if (tok_result != 0) {
        mtmd_input_chunks_free(chunks);
        LOGE("VLM: tokenization failed");
        jstring jerr = env->NewStringUTF("Failed to tokenize multimodal input");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    // Clear KV cache — VLM always starts fresh
    llama_memory_t mem = llama_get_memory(g_state.ctx);
    if (mem) llama_memory_clear(mem, true);
    g_state.n_past = 0;
    g_state.prev_prompt_tokens.clear();

    rebuild_sampler();

    auto t_start = std::chrono::high_resolution_clock::now();

    // Report progress for image encoding
    if (g_onProgress) {
        env->CallVoidMethod(callback, g_onProgress, 0.1f);
    }

    // Walk chunks manually so we can split vision-encode time from LLM
    // prompt-eval time on image embeddings, and stream progress between
    // chunks instead of a single blocking call.
    const int32_t vlm_n_batch = 512;  // mobile-friendly cap
    int64_t t_encode_us = 0;
    int64_t t_decode_us = 0;
    int32_t n_image_tokens = 0;
    llama_pos new_n_past = 0;
    int32_t eval_result = 0;

    const size_t n_chunks_total = mtmd_input_chunks_size(chunks);
    for (size_t ci = 0; ci < n_chunks_total && eval_result == 0; ci++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
        const bool is_last = (ci == n_chunks_total - 1);
        const enum mtmd_input_chunk_type ctype = mtmd_input_chunk_get_type(chunk);

        if (ctype == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            const int64_t t0 = llama_time_us();
            eval_result = mtmd_helper_eval_chunk_single(
                g_vlm.ctx, g_state.ctx, chunk,
                new_n_past, 0, vlm_n_batch, is_last, &new_n_past);
            t_decode_us += llama_time_us() - t0;
        } else {
            // Vision / audio encoder forward
            const int64_t t_enc0 = llama_time_us();
            eval_result = mtmd_encode_chunk(g_vlm.ctx, chunk);
            t_encode_us += llama_time_us() - t_enc0;
            if (eval_result != 0) break;

            float * embd = mtmd_get_output_embd(g_vlm.ctx);
            const int32_t n_tok = (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
            n_image_tokens += n_tok;

            const int64_t t_dec0 = llama_time_us();
            eval_result = mtmd_helper_decode_image_chunk(
                g_vlm.ctx, g_state.ctx, chunk, embd,
                new_n_past, 0, vlm_n_batch, &new_n_past);
            t_decode_us += llama_time_us() - t_dec0;
        }

        if (g_onProgress && n_chunks_total > 1) {
            float p = 0.1f + 0.4f * ((float)(ci + 1) / (float)n_chunks_total);
            env->CallVoidMethod(callback, g_onProgress, p);
        }
    }

    mtmd_input_chunks_free(chunks);

    if (eval_result != 0) {
        LOGE("VLM: chunk evaluation failed (%d)", eval_result);
        jstring jerr = env->NewStringUTF("Failed to process multimodal input");
        env->CallVoidMethod(callback, g_onError, jerr);
        env->DeleteLocalRef(jerr);
        return JNI_FALSE;
    }

    g_state.n_past = new_n_past;
    int prompt_tokens = g_state.n_past;

    const float vlm_encode_ms = t_encode_us / 1000.0f;
    const float vlm_decode_ms = t_decode_us / 1000.0f;

    if (g_onVlmStageMetrics) {
        env->CallVoidMethod(callback, g_onVlmStageMetrics,
            vlm_encode_ms, vlm_decode_ms, (jint)n_image_tokens);
    }

    if (g_onProgress) {
        env->CallVoidMethod(callback, g_onProgress, 0.5f);
    }

    auto t_prompt_done = std::chrono::high_resolution_clock::now();

    LOGI("VLM: prompt processed %d tokens (image=%d, encode=%.0fms, decode=%.0fms), starting generation",
         prompt_tokens, n_image_tokens, vlm_encode_ms, vlm_decode_ms);

    // ── Autoregressive generation loop (reuses existing sampling infrastructure) ──

    const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
    int n_generated = 0;
    std::string generated_text;
    generated_text.reserve(maxTokens * 4);
    size_t sent_count = 0;

    antiprompt_state antiprompt;
    antiprompt.set_stops(tmpl_result.stops);

    token_batcher batcher(env, callback, g_onToken);

    while (n_generated < maxTokens && !g_state.cancel_flag.load()) {
        if (!g_state.sampler) break;

        llama_token id = common_sampler_sample(g_state.sampler, g_state.ctx, -1);
        common_sampler_accept(g_state.sampler, id, true);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf) - 1, 0, true);
        if (n > 0) {
            buf[n] = '\0';
            generated_text.append(buf, n);

            size_t unsent_start = std::min(sent_count, generated_text.size());
            size_t unsent_len = generated_text.size() - unsent_start;
            std::string unsent(generated_text.data() + unsent_start, unsent_len);

            size_t stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_FULL);
            if (stop_pos != std::string::npos) {
                generated_text.resize(unsent_start + stop_pos);
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                }
                batcher.flush();
                break;
            }

            stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_PARTIAL);
            if (stop_pos == std::string::npos) {
                if (sent_count < generated_text.size()) {
                    batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
                    sent_count = generated_text.size();
                }
            }

            if (env->ExceptionCheck()) { env->ExceptionClear(); break; }
        }

        if (g_state.n_past >= (int)llama_n_ctx(g_state.ctx) - 1) {
            if (!try_context_shift()) break;
        }

        llama_batch & sb = get_single_batch();
        common_batch_clear(sb);
        common_batch_add(sb, id, g_state.n_past, {0}, true);
        if (llama_decode(g_state.ctx, sb) != 0) break;
        g_state.n_past++;
        n_generated++;
    }

    // Flush remaining tokens
    if (sent_count < generated_text.size()) {
        batcher.add(generated_text.data() + sent_count, generated_text.size() - sent_count);
    }
    batcher.flush();
    if (!g_utf8_buffer.empty()) {
        batcher.buf = std::move(g_utf8_buffer);
        g_utf8_buffer.clear();
        batcher.flush();
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    float prompt_ms = std::chrono::duration<float, std::milli>(t_prompt_done - t_start).count();
    float gen_ms = std::chrono::duration<float, std::milli>(t_end - t_prompt_done).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    float tps = gen_ms > 0 ? (n_generated / (gen_ms / 1000.0f)) : 0;
    float model_mb = 0, ctx_mb = 0, peak_mb = 0, mem_pct = 0;
    compute_memory_metrics(model_mb, ctx_mb, peak_mb, mem_pct);

    if (g_onMetrics) {
        env->CallVoidMethod(callback, g_onMetrics,
            tps, prompt_ms, total_ms,
            prompt_tokens, n_generated,
            model_mb, ctx_mb, peak_mb, mem_pct);
    }

    env->CallVoidMethod(callback, g_onDone);

    LOGI("VLM: generation complete — %d tokens, %.1f t/s, prompt %.0fms", n_generated, tps, prompt_ms);
    return JNI_TRUE;
}

// ── KV Eviction Policy ────────────────────────────────────────────────────────

// nativeSetKvPolicy(nSink, nWindow, evictAtFull)
extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetKvPolicy(
        JNIEnv *, jobject, jint nSink, jint nWindow, jboolean evictAtFull) {
    g_state.kv_n_sink        = (int)nSink;
    g_state.kv_n_window      = (int)nWindow;
    g_state.kv_evict_at_full = (bool)evictAtFull;
    LOGI("KV policy: sink=%d window=%d evict_at_full=%d", (int)nSink, (int)nWindow, (int)evictAtFull);
}

// nativeEvictToBudget — apply StreamingLLM eviction immediately
extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeEvictToBudget(JNIEnv *, jobject) {
    kv_evict_streaming();
}



// ── Error Tracker JNI ──────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeErrorInit(JNIEnv *, jobject) {
    tn_error_init();
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeErrorSetCrashLogPath(
        JNIEnv * env, jobject, jstring jpath) {
    if (!jpath) return;
    const char * p = env->GetStringUTFChars(jpath, nullptr);
    tn_error_set_crash_log_path(p);
    env->ReleaseStringUTFChars(jpath, p);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeErrorGetLastJson(JNIEnv * env, jobject) {
    const char * j = tn_error_get_last_json();
    return env->NewStringUTF(j ? j : "{}");
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeErrorClear(JNIEnv *, jobject) {
    tn_error_clear_last();
    tn_error_clear_op();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeTextDigest(
        JNIEnv * env, jobject,
        jstring jtext, jstring jquery,
        jint jtargetTokens,
        jfloat jwQuery, jfloat jwCentrality, jfloat jwLead, jfloat jwEntity,
        jfloat jmmrLambda,
        jint jmaxSentences, jint jminSentenceChars, jint jmaxSentenceChars,
        jint jtextrankIters, jfloat jtextrankDamping) {

    if (!jtext) return nullptr;

    const char * tcs = env->GetStringUTFChars(jtext, nullptr);
    const char * qcs = jquery ? env->GetStringUTFChars(jquery, nullptr) : nullptr;
    std::string text_str = tcs ? tcs : "";
    std::string query_str = qcs ? qcs : "";
    if (tcs) env->ReleaseStringUTFChars(jtext, tcs);
    if (qcs) env->ReleaseStringUTFChars(jquery, qcs);

    text_digest::Options opts;
    if (jtargetTokens > 0) opts.target_tokens = jtargetTokens;
    if (jwQuery >= 0.f) opts.w_query = jwQuery;
    if (jwCentrality >= 0.f) opts.w_centrality = jwCentrality;
    if (jwLead >= 0.f) opts.w_lead = jwLead;
    if (jwEntity >= 0.f) opts.w_entity = jwEntity;
    if (jmmrLambda > 0.f) opts.mmr_lambda = jmmrLambda;
    if (jmaxSentences > 0) opts.max_sentences = jmaxSentences;
    if (jminSentenceChars > 0) opts.min_sentence_chars = jminSentenceChars;
    if (jmaxSentenceChars > 0) opts.max_sentence_chars = jmaxSentenceChars;
    if (jtextrankIters > 0) opts.textrank_iterations = jtextrankIters;
    if (jtextrankDamping > 0.f) opts.textrank_damping = jtextrankDamping;

    std::string out = text_digest::compress(text_str, query_str, opts);
    return env->NewStringUTF(out.c_str());
}
