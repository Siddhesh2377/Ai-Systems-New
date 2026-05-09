// JNI bridge for com.dark.gguf_lib. Wraps llama.cpp + the tool-neuron engine
// helpers (thread-engine, rag-engine, mtmd) and exposes a flat C-callable
// surface to Kotlin via GGUFNativeLib.

#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <android/log.h>

#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "chat.h"
#include "ggml-backend.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "thread-engine.h"
#include "rag-engine.h"
#include "vt_cache.h"
#include "vlm_kv_cache.h"
#include "rag_ingest/rag_ingest.h"
#include "text_digest/text_digest.h"
#include "error_tracker.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <nlohmann/json.hpp>

#define TAG "gguf_lib"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using json = nlohmann::ordered_json;

static void llama_android_log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (text == nullptr || text[0] == '\0') return;
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

        // ggml's static-link build only auto-registers CPU. With GGML_BACKEND_DL=OFF
        // (our config — single .so, no separate libggml-vulkan.so to dlopen), the
        // Vulkan backend stays silent unless we call its reg() manually here.
#ifdef GGML_USE_VULKAN
        if (auto * reg = ggml_backend_vk_reg()) {
            ggml_backend_register(reg);
            LOGI("Vulkan backend registered (devices=%zu)",
                 ggml_backend_reg_dev_count(reg));
        } else {
            LOGW("Vulkan backend reg returned null (libvulkan missing or device unsupported)");
        }
#endif
        LOGI("ggml backends registered: %zu, devices: %zu",
             ggml_backend_reg_count(), ggml_backend_dev_count());
    });
}

// Cached StreamCallback method IDs. JNI GetObjectClass + GetMethodID are
// re-done only when a different callback class shows up; the hot path is a
// single IsSameObject check per generate call.
static jclass    g_cb_class            = nullptr;
static jmethodID g_onToken             = nullptr;
static jmethodID g_onDone              = nullptr;
static jmethodID g_onError             = nullptr;
static jmethodID g_onMetrics           = nullptr;
static jmethodID g_onProgress          = nullptr;
static jmethodID g_onTokenBytes        = nullptr; // zero-copy byte[] fast path
static jmethodID g_onVlmStageMetrics   = nullptr;
static jmethodID g_onVlmCacheStatus    = nullptr; // VT cache hit/miss feedback
static jmethodID g_onVlmKvCacheStatus  = nullptr; // VLM-KV cache hit/miss feedback

static jclass    g_embed_cb_class   = nullptr;
static jmethodID g_embed_onComplete = nullptr;
static jmethodID g_embed_onError    = nullptr;

static bool ensure_callback_methods(JNIEnv * env, jobject callback) {
    jclass cls = env->GetObjectClass(callback);
    if (g_cb_class && env->IsSameObject(cls, g_cb_class)) {
        env->DeleteLocalRef(cls);
        return true;
    }
    if (g_cb_class) env->DeleteGlobalRef(g_cb_class);
    g_cb_class = (jclass)env->NewGlobalRef(cls);
    g_onToken    = env->GetMethodID(cls, "onToken",    "(Ljava/lang/String;)V");
    g_onDone     = env->GetMethodID(cls, "onDone",     "()V");
    g_onError    = env->GetMethodID(cls, "onError",    "(Ljava/lang/String;)V");
    g_onMetrics  = env->GetMethodID(cls, "onMetrics",  "(FFFIIFFFF)V");
    g_onProgress = env->GetMethodID(cls, "onProgress", "(F)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_onTokenBytes = env->GetMethodID(cls, "onTokenBytes", "([BI)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_onVlmStageMetrics = env->GetMethodID(cls, "onVlmStageMetrics", "(FFI)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_onVlmCacheStatus = env->GetMethodID(cls, "onVlmCacheStatus", "(ZII)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    g_onVlmKvCacheStatus = env->GetMethodID(cls, "onVlmKvCacheStatus", "(ZI)V");
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(cls);
    return g_onToken && g_onDone && g_onError;
}

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

// Singleton engine state. One model at a time. gen_mutex guards everything;
// the cancel_flag is the only field touched outside the mutex (atomic).
static struct {
    llama_model    * model   = nullptr;
    llama_context  * ctx     = nullptr;
    common_sampler * sampler = nullptr;

    common_chat_templates_ptr chat_templates;
    common_params_sampling    sampling_params;

    std::string system_prompt;
    std::string chat_template_override;

    // 0 = power_saving, 1 = balanced, 2 = performance — drives thread-engine
    int thread_mode = 1;

    std::atomic<bool> cancel_flag{false};
    std::mutex        gen_mutex;

    std::vector<llama_token> session_tokens;
    int                      n_past = 0;
    std::vector<llama_token> prev_prompt_tokens;
    int                      n_system_tokens = 0;

    std::string prompt_cache_dir;
    bool        thinking_enabled = true;

    // StreamingLLM-style eviction. nWindow=0 disables → context shift fallback.
    int  kv_n_sink        = 4;
    int  kv_n_window      = 0;
    bool kv_evict_at_full = false;

    // Defaults captured from the last load() — used when the user passes -1
    // for runtime knobs. Lets the caller revert to model-load defaults.
    bool use_mmap  = true;
    bool use_mlock = false;
} g_state;

static void kv_evict_streaming();

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

static ggml_type cache_type_from_string(const std::string & s) {
    if (s == "f32")  return GGML_TYPE_F32;
    if (s == "f16")  return GGML_TYPE_F16;
    if (s == "q8_0") return GGML_TYPE_Q8_0;
    if (s == "q4_0") return GGML_TYPE_Q4_0;
    if (s == "q4_1") return GGML_TYPE_Q4_1;
    if (s == "q5_0") return GGML_TYPE_Q5_0;
    if (s == "q5_1") return GGML_TYPE_Q5_1;
    return GGML_TYPE_Q8_0;
}

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
            LOGI("Pinned to %d perf cores (mode=%d)", cfg.n_perf_core_ids, mode);
        } else {
            LOGW("sched_setaffinity failed: %s", strerror(errno));
        }
    }
}

// rebuild_sampler(force=false) skips the rebuild if only simple knobs (temp,
// top_p) changed — preserving the sampler's running state (rep-penalty ring,
// mirostat mu). force=true is required when grammar / logit bias / preserved
// tokens change because common_sampler doesn't support in-place edits.
static bool g_sampler_needs_rebuild = true;

static void rebuild_sampler(bool force = true) {
    if (!force && !g_sampler_needs_rebuild && g_state.sampler) return;
    if (g_state.sampler) {
        common_sampler_free(g_state.sampler);
        g_state.sampler = nullptr;
    }
    if (g_state.model) {
        g_state.sampler = common_sampler_init(g_state.model, g_state.sampling_params);
    }
    g_sampler_needs_rebuild = false;
}

static void mark_sampler_dirty() {
    g_sampler_needs_rebuild = true;
}

// Kotlin sometimes sends persona names ("Luna", "Nova") as the message role
// for assistant turns. Chat templates only accept the four canonical roles,
// so anything else is remapped to "assistant".
static std::vector<common_chat_msg> parse_messages_json(const std::string & messages_json) {
    std::vector<common_chat_msg> msgs;
    try {
        auto j = json::parse(messages_json);
        if (j.is_array()) {
            for (auto & msg : j) {
                common_chat_msg cm;
                cm.role    = msg.value("role", "user");
                cm.content = msg.value("content", "");
                if (cm.role != "system" && cm.role != "user" &&
                    cm.role != "assistant") {
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

// Safety-net stop strings — catch model turn boundaries that aren't part of
// the chat template's additional_stops list.
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

struct chat_template_result {
    std::string              prompt;
    std::vector<std::string> stops;
};

static chat_template_result apply_chat_template(const std::vector<common_chat_msg> & messages,
                                                 bool add_generation_prompt = true) {
    chat_template_result out;

    if (!g_state.chat_templates) {
        std::string prompt;
        for (auto & msg : messages) {
            if (msg.role == "system")         prompt += msg.content + "\n";
            else if (msg.role == "user")      prompt += "User: " + msg.content + "\n";
            else if (msg.role == "assistant") prompt += "Assistant: " + msg.content + "\n";
        }
        if (add_generation_prompt) prompt += "Assistant:";
        out.prompt = prompt;
        out.stops  = {"\nUser:", "\nuser:", "\n\nUser:"};
        out.stops.insert(out.stops.end(), COMMON_STOP_STRINGS.begin(), COMMON_STOP_STRINGS.end());
        return out;
    }

    common_chat_templates_inputs inputs;
    inputs.messages              = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = g_state.thinking_enabled;

    auto result = common_chat_templates_apply(g_state.chat_templates.get(), inputs);
    out.prompt = result.prompt;
    out.stops  = result.additional_stops;
    out.stops.insert(out.stops.end(), COMMON_STOP_STRINGS.begin(), COMMON_STOP_STRINGS.end());
    return out;
}

// Two-phase antiprompt: STOP_FULL hits when a complete stop string lands in
// the tail; STOP_PARTIAL hits when the tail is a prefix of one — meaning we
// must hold the unsent tokens until the next decode resolves the ambiguity.
enum stop_type { STOP_FULL, STOP_PARTIAL };

struct antiprompt_state {
    std::vector<std::string> stops;
    std::string              stopping_word;
    bool                     stopped = false;

    void set_stops(const std::vector<std::string> & s) {
        stops = s;
        stopping_word.clear();
        stopped = false;
    }

    size_t find_stop(const std::string & text, size_t last_token_size, stop_type type) {
        size_t stop_pos = std::string::npos;
        for (auto & word : stops) {
            if (word.empty()) continue;
            size_t pos;
            if (type == STOP_FULL) {
                size_t window = word.size() + last_token_size;
                size_t from   = text.size() > window ? text.size() - window : 0;
                pos = text.find(word, from);
            } else {
                pos = find_partial(word, text);
            }
            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_FULL) {
                    stopping_word = word;
                    stopped       = true;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

private:
    static size_t find_partial(const std::string & word, const std::string & text) {
        if (text.empty() || word.empty()) return std::string::npos;
        size_t max_check = std::min(word.size() - 1, text.size());
        for (size_t len = max_check; len >= 1; len--) {
            if (text.compare(text.size() - len, len, word, 0, len) == 0) {
                return text.size() - len;
            }
        }
        return std::string::npos;
    }
};

// UTF-8 incomplete-sequence carry buffer. Tokens can split a multibyte glyph
// across two callbacks; we append the trailing bytes here and prepend on the
// next flush so Kotlin only sees valid UTF-8.
static std::string g_utf8_buffer;

static inline bool is_all_ascii(const char * data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if ((unsigned char)data[i] >= 0x80 || data[i] == 0x00) return false;
    }
    return true;
}

// Strip invalid bytes and overlong encodings; carry incomplete trailing
// multi-byte sequences to g_utf8_buffer for the next call.
static std::string sanitize_utf8(const std::string & input) {
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

// One-shot sanitization for non-streaming JNI strings. Saves & restores the
// streaming UTF-8 carry so a model-info call mid-generate doesn't corrupt it.
static jstring safe_new_string_utf(JNIEnv * env, const char * text) {
    if (!text || !text[0]) return env->NewStringUTF("");
    size_t len = strlen(text);
    if (is_all_ascii(text, len)) {
        jstring result = env->NewStringUTF(text);
        if (!result) { env->ExceptionClear(); return env->NewStringUTF("?"); }
        return result;
    }
    std::string saved = std::move(g_utf8_buffer);
    g_utf8_buffer.clear();
    std::string clean = sanitize_utf8(text);
    g_utf8_buffer.clear();
    if (!saved.empty()) g_utf8_buffer = std::move(saved);
    if (clean.empty()) return env->NewStringUTF("");
    jstring result = env->NewStringUTF(clean.c_str());
    if (!result) { env->ExceptionClear(); return env->NewStringUTF("?"); }
    return result;
}

// Token-batching threshold. Larger → fewer Binder/JNI calls but more latency
// before first visible token. 64 = direct in-process JNI, 256+ = AIDL service.
static size_t g_token_batch_threshold = 256;

// Reused jbyteArray for the zero-copy onTokenBytes path. Avoids allocating
// a fresh array every flush.
static jbyteArray g_token_byte_buf = nullptr;
static int        g_token_byte_cap = 0;

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
    JNIEnv *    env;
    jobject     callback;
    jmethodID   onToken;

    token_batcher(JNIEnv * e, jobject cb, jmethodID m)
        : env(e), callback(cb), onToken(m) { buf.reserve(256); }

    bool add(const char * text, size_t len) {
        buf.append(text, len);
        return buf.size() >= g_token_batch_threshold ? flush() : true;
    }
    bool add(const std::string & text) { return add(text.data(), text.size()); }

    bool flush() {
        if (buf.empty()) return true;

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

        if (g_onTokenBytes) {
            int len = (int)clean.size();
            ensure_token_byte_buf(env, len);
            env->SetByteArrayRegion(g_token_byte_buf, 0, len, (const jbyte *)clean.data());
            env->CallVoidMethod(callback, g_onTokenBytes, g_token_byte_buf, (jint)len);
            return !env->ExceptionCheck();
        }

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

// Reusable batches: one sized to n_batch for prompt eval, one of size 1 for
// the autoregressive generation loop. Avoids per-call alloc/free.
static llama_batch g_prompt_batch     = {};
static int         g_prompt_batch_cap = 0;
static llama_batch g_single_batch     = {};
static bool        g_single_batch_init = false;

static llama_batch & get_single_batch() {
    if (!g_single_batch_init) {
        g_single_batch      = llama_batch_init(1, 0, 1);
        g_single_batch_init = true;
    }
    return g_single_batch;
}

typedef void (*eval_progress_fn)(float progress, void * user_data);

static bool eval_tokens(const std::vector<llama_token> & tokens, int & n_past,
                         eval_progress_fn progress = nullptr, void * progress_data = nullptr) {
    if (tokens.empty()) return true;

    const int n_batch = llama_n_batch(g_state.ctx);

    if (g_prompt_batch_cap < n_batch) {
        if (g_prompt_batch_cap > 0) llama_batch_free(g_prompt_batch);
        g_prompt_batch     = llama_batch_init(n_batch, 0, 1);
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

        if (progress) progress((float)(i + n_eval) / (float)total, progress_data);

        if (g_state.cancel_flag.load()) {
            LOGI("Prompt evaluation cancelled at %d/%d tokens", n_past, total);
            return false;
        }
    }
    return true;
}

struct jni_progress_ctx {
    JNIEnv * env;
    jobject  callback;
};

static void jni_eval_progress(float progress, void * user_data) {
    auto * ctx = (jni_progress_ctx *)user_data;
    if (g_onProgress && ctx->env && ctx->callback) {
        ctx->env->CallVoidMethod(ctx->callback, g_onProgress, progress);
    }
}

static int find_common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    int n = std::min((int)a.size(), (int)b.size());
    for (int i = 0; i < n; i++) if (a[i] != b[i]) return i;
    return n;
}

// 0 = fits, -1 = prompt alone overflows the context window (fatal).
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

// Drops the older half of non-system tokens, shifting tail positions down.
// When the StreamingLLM policy is active, defers to kv_evict_streaming.
static bool try_context_shift() {
    if (!g_state.ctx) return false;

    if (g_state.kv_n_window > 0) {
        kv_evict_streaming();
        return g_state.n_past < (int)llama_n_ctx(g_state.ctx) - 1;
    }

    llama_memory_t mem = llama_get_memory(g_state.ctx);
    if (!mem || !llama_memory_can_shift(mem)) {
        LOGW("Context shift not supported by memory backend");
        return false;
    }

    int n_keep    = std::max(g_state.n_system_tokens, 4);
    int n_discard = (g_state.n_past - n_keep) / 2;
    if (n_discard <= 0) return false;

    llama_memory_seq_rm(mem, 0, n_keep, n_keep + n_discard);
    llama_memory_seq_add(mem, 0, n_keep + n_discard, g_state.n_past, -n_discard);

    g_state.n_past -= n_discard;
    g_state.prev_prompt_tokens.clear();
    LOGI("Context shift: n_past -> %d (kept %d sys, dropped %d)",
         g_state.n_past, n_keep, n_discard);
    return true;
}

// Keep [0, n_sink) and the tail [n_past-n_window, n_past); evict the middle.
static void kv_evict_streaming() {
    if (!g_state.ctx || g_state.kv_n_window <= 0) return;
    int n_past   = g_state.n_past;
    int n_sink   = g_state.kv_n_sink;
    int n_window = g_state.kv_n_window;
    if (n_past <= n_sink + n_window) return;

    llama_memory_t mem = llama_get_memory(g_state.ctx);
    if (!mem) return;

    int evict_end = n_past - n_window;
    llama_memory_seq_rm(mem, 0, n_sink, evict_end);
    llama_memory_seq_add(mem, 0, evict_end, n_past, -(evict_end - n_sink));
    g_state.n_past = n_sink + n_window;
    g_state.prev_prompt_tokens.clear();
    LOGI("KV evict: n_past %d -> %d (sink=%d window=%d)",
         n_past, g_state.n_past, n_sink, n_window);
}

static float get_context_usage() {
    if (!g_state.ctx) return 0.0f;
    int n_ctx = (int)llama_n_ctx(g_state.ctx);
    if (n_ctx <= 0) return 0.0f;
    return (float)g_state.n_past / (float)n_ctx;
}

// FNV-1a 64-bit — used purely as a stable filename for the disk-backed prompt
// cache; not security-sensitive.
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

static bool try_restore_prompt_cache(const std::string & system_prompt,
                                      const std::vector<llama_token> & /*sys_tokens*/) {
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
        LOGI("Prompt cache restored: %zu tokens", n_token_count);
        return true;
    }
    LOGW("Prompt cache file exists but failed to load: %s", cache_path.c_str());
    return false;
}

static void save_prompt_cache(const std::string & system_prompt,
                               const std::vector<llama_token> & tokens, int n_tokens) {
    if (g_state.prompt_cache_dir.empty() || system_prompt.empty()) return;
    std::string cache_path = g_state.prompt_cache_dir + "/prompt_" + hash_string(system_prompt) + ".cache";
    if (!llama_state_save_file(g_state.ctx, cache_path.c_str(), tokens.data(), n_tokens)) {
        LOGW("Failed to save prompt cache to %s", cache_path.c_str());
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModel(
        JNIEnv * env, jobject,
        jstring jpath, jint nCtx, jint nThreads, jint nBatch,
        jboolean flashAttn, jboolean useMmap, jboolean useMlock,
        jstring jCacheTypeK, jstring jCacheTypeV,
        jboolean opOffload) {

    ensure_backend_init();

    {
        const char * peek = env->GetStringUTFChars(jpath, nullptr);
        char detail[512];
        snprintf(detail, sizeof(detail),
            "path=%s ctx=%d threads=%d batch=%d flash=%d mmap=%d mlock=%d",
            peek ? peek : "<null>", (int)nCtx, (int)nThreads, (int)nBatch,
            (int)flashAttn, (int)useMmap, (int)useMlock);
        tn_error_set_op("loadModel", detail);
        if (peek) env->ReleaseStringUTFChars(jpath, peek);
    }

    std::lock_guard<std::mutex> lock(g_state.gen_mutex);

    if (g_state.sampler) { common_sampler_free(g_state.sampler); g_state.sampler = nullptr; }
    if (g_state.ctx)     { llama_free(g_state.ctx); g_state.ctx = nullptr; }
    if (g_state.model)   { llama_model_free(g_state.model); g_state.model = nullptr; }
    g_state.chat_templates.reset();
    g_state.n_past = 0;
    g_state.session_tokens.clear();
    g_state.prev_prompt_tokens.clear();
    g_state.n_system_tokens = 0;

    // Copy JNI strings before any early returns — the error-path snprintf below
    // reads cacheK_s/cacheV_s, which the original code freed too early.
    std::string path_s, cacheK_s, cacheV_s;
    {
        const char * path = env->GetStringUTFChars(jpath,       nullptr);
        const char * ck   = env->GetStringUTFChars(jCacheTypeK, nullptr);
        const char * cv   = env->GetStringUTFChars(jCacheTypeV, nullptr);
        if (path) path_s   = path;
        if (ck)   cacheK_s = ck;
        if (cv)   cacheV_s = cv;
        if (path) env->ReleaseStringUTFChars(jpath,       path);
        if (ck)   env->ReleaseStringUTFChars(jCacheTypeK, ck);
        if (cv)   env->ReleaseStringUTFChars(jCacheTypeV, cv);
    }

    LOGI("Loading model: %s (ctx=%d threads=%d batch=%d flash=%d mmap=%d mlock=%d)",
         path_s.c_str(), nCtx, nThreads, nBatch, flashAttn, useMmap, useMlock);

    auto mparams = llama_model_default_params();
    mparams.use_mmap  = (bool)useMmap;
    mparams.use_mlock = (bool)useMlock;
    g_state.use_mmap  = mparams.use_mmap;
    g_state.use_mlock = mparams.use_mlock;

    // Default n_gpu_layers is -1 (= all layers offloaded). On a UMA SoC like
    // Adreno 810 that's the wrong trade-off: weight-on-GPU forces every op
    // through Vulkan including single-token decode, which destroys tok/s.
    // We keep all layer weights on CPU and rely on op_offload (below) to
    // route only the *large* ops to GPU per-tensor.
    mparams.n_gpu_layers = 0;

    g_state.model = llama_model_load_from_file(path_s.c_str(), mparams);
    if (!g_state.model) {
        tn_error_set_last(TN_ERR_MODEL_LOAD, "ModelLoad",
            "llama_model_load_from_file returned null. Likely causes: corrupt or non-GGUF file, unsupported architecture, or out of memory.");
        return JNI_FALSE;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = nCtx > 0 ? nCtx : 4096;

    {
        tn_thread_config cfg = tn_thread_config_for_mode((tn_thread_mode)g_state.thread_mode);
        if (nThreads > 0) {
            cparams.n_threads       = nThreads;
            cparams.n_threads_batch = nThreads;
        } else {
            cparams.n_threads       = cfg.n_threads_generation;
            cparams.n_threads_batch = cfg.n_threads_batch;
        }
        cparams.n_batch = nBatch > 0 ? nBatch : cfg.n_batch;
    }

    if (flashAttn) cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k = cache_type_from_string(cacheK_s);
    cparams.type_v = cache_type_from_string(cacheV_s);

    // Per-op CPU/GPU routing. When true and a non-CPU backend (Vulkan, etc.)
    // is registered with ggml, large ops (batch ≥ 32 by default) get offloaded
    // to GPU while single-token decode stays on CPU. No layer weights are
    // moved — purely a compute-side hint. See VLM.md "Per-op routing" for the
    // trade-off discussion. Override the threshold via GGML_OP_OFFLOAD_MIN_BATCH.
    cparams.op_offload = (bool)opOffload;
    if (cparams.op_offload) {
        LOGI("op_offload: enabled — large ops will be routed to GPU when available");
    }

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
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

    apply_thread_mode(g_state.thread_mode);
    g_state.chat_templates = common_chat_templates_init(g_state.model, g_state.chat_template_override);
    rebuild_sampler();

    // Warm-up: single-token decode to fault-in hot weight pages so the first
    // real query doesn't pay the page-fault tax in TTFT.
    {
        const llama_vocab * vocab = llama_model_get_vocab(g_state.model);
        llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            llama_batch & sb = get_single_batch();
            common_batch_clear(sb);
            common_batch_add(sb, bos, 0, {0}, true);
            llama_decode(g_state.ctx, sb);
            llama_memory_clear(llama_get_memory(g_state.ctx), true);
        }
    }

    LOGI("Model loaded (ctx=%d threads_gen=%d threads_batch=%d batch=%d mode=%d)",
         (int)llama_n_ctx(g_state.ctx), cparams.n_threads, cparams.n_threads_batch,
         cparams.n_batch, g_state.thread_mode);
    return JNI_TRUE;
}

// dup() the caller's fd so /proc/self/fd/<n> stays valid for the entire load —
// the Kotlin ParcelFileDescriptor may be GC'd before llama.cpp finishes mmap.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModelFromFd(
        JNIEnv * env, jobject thiz,
        jint fd, jint nCtx, jint nThreads, jint nBatch,
        jboolean flashAttn, jboolean useMmap, jboolean useMlock,
        jstring jCacheTypeK, jstring jCacheTypeV,
        jboolean opOffload) {

    if (fd < 0) {
        LOGE("Invalid file descriptor: %d", fd);
        return JNI_FALSE;
    }

    int owned_fd = dup(fd);
    if (owned_fd < 0) {
        LOGE("dup() failed for fd %d: %s", fd, strerror(errno));
        return JNI_FALSE;
    }

    // SAF pipe-based providers return non-seekable fds — mmap-loading needs seek.
    if (lseek(owned_fd, 0, SEEK_CUR) == (off_t)-1) {
        LOGE("fd %d is not seekable (SAF pipe provider?): %s", fd, strerror(errno));
        close(owned_fd);
        return JNI_FALSE;
    }

    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", owned_fd);

    jstring jpath = env->NewStringUTF(path);
    jboolean result = Java_com_dark_gguf_1lib_GGUFNativeLib_nativeLoadModel(
        env, thiz, jpath, nCtx, nThreads, nBatch, flashAttn, useMmap, useMlock,
        jCacheTypeK, jCacheTypeV, opOffload);
    env->DeleteLocalRef(jpath);

    // Close our dup, not the caller's fd — Kotlin owns the original via PFD.
    close(owned_fd);
    return result;
}

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

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetSystemPrompt(
        JNIEnv * env, jobject, jstring jprompt) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    g_state.system_prompt = prompt;
    env->ReleaseStringUTFChars(jprompt, prompt);
    LOGI("System prompt set (%zu chars)", g_state.system_prompt.size());
}

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

    rebuild_sampler();

    auto t_start = std::chrono::high_resolution_clock::now();

    // evaluate only the new tokens beyond the cached prefix
    std::vector<llama_token> new_tokens(tokens.begin() + g_state.n_past, tokens.end());
    int prompt_tokens = (int)new_tokens.size();

    // set up progress reporting for long prompt evaluation
    jni_progress_ctx progress_ctx = { env, callback };
    if (!new_tokens.empty() && !eval_tokens(new_tokens, g_state.n_past,
                                             jni_eval_progress, &progress_ctx)) {
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

    rebuild_sampler();

    auto t_start = std::chrono::high_resolution_clock::now();

    // only evaluate tokens beyond the cached prefix
    std::vector<llama_token> new_tokens(tokens.begin() + g_state.n_past, tokens.end());
    int prompt_tokens = (int)new_tokens.size();

    // progress reporting for long prompt evaluation
    jni_progress_ctx mt_progress = { env, callback };
    if (!new_tokens.empty() && !eval_tokens(new_tokens, g_state.n_past,
                                             jni_eval_progress, &mt_progress)) {
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
            size_t unsent_len   = generated_text.size() - unsent_start;
            // Hot path: this runs on every token. The std::string ctor copies
            // unsent_len bytes; antiprompt.find_stop only reads within that
            // window. Per-token cost is negligible (typically <128 bytes).
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

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeStopGeneration(JNIEnv *, jobject) {
    g_state.cancel_flag = true;
    LOGI("Generation stop requested");
}

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
    LOGI("Model released");
}

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

        rebuild_sampler();
        LOGI("Logit bias set: %zu entries", g_state.sampling_params.logit_bias.size());

    } catch (const std::exception & e) {
        LOGE("Failed to parse logit bias JSON: %s", e.what());
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetStateSize(JNIEnv *, jobject) {
    if (!g_state.ctx) return 0;
    return (jlong)llama_state_get_size(g_state.ctx);
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeGetContextUsage(JNIEnv *, jobject) {
    return (jfloat)get_context_usage();
}

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

// Embedding engine: independent model + context. Runs in parallel with the
// chat engine — its own mutex, no shared state.
static struct {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    std::mutex      mutex;
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

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSupportsThinking(JNIEnv *, jobject) {
    if (!g_state.chat_templates) return JNI_FALSE;
    return common_chat_templates_support_enable_thinking(g_state.chat_templates.get())
           ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetThinkingEnabled(JNIEnv *, jobject, jboolean enabled) {
    g_state.thinking_enabled = (enabled == JNI_TRUE);
    LOGI("Thinking %s", g_state.thinking_enabled ? "enabled" : "disabled");
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetThreadMode(JNIEnv *, jobject, jint mode) {
    std::lock_guard<std::mutex> lock(g_state.gen_mutex);
    if (mode < 0 || mode > 2) mode = 1;
    apply_thread_mode(mode);
    LOGI("Thread mode set: %d", mode);
}

// Larger threshold = fewer IPC calls, higher latency to first visible token.
extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetTokenBatchSize(JNIEnv *, jobject, jint bytes) {
    if (bytes >= 1) g_token_batch_threshold = (size_t)bytes;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeSetPromptCacheDir(
        JNIEnv * env, jobject, jstring jpath) {
    const char * path = env->GetStringUTFChars(jpath, nullptr);
    g_state.prompt_cache_dir = path;
    env->ReleaseStringUTFChars(jpath, path);
    LOGI("Prompt cache dir set: %s", g_state.prompt_cache_dir.c_str());
}

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

// RAG engine state — separate from g_state, has its own embedding model.
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


// VT (Vision Token) cache state. Lazily initialised on first
// nativeVtCacheInit() call. Owned by Kotlin's GGMLEngine lifecycle.
static struct {
    vt_cache_t * cache = nullptr;
    std::mutex   mutex;
} g_vt;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheInit(
        JNIEnv * env, jobject, jstring jdir, jlong budgetBytes) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (g_vt.cache) {
        vt_cache_free(g_vt.cache);
        g_vt.cache = nullptr;
    }
    if (!jdir) return JNI_FALSE;
    const char * dir = env->GetStringUTFChars(jdir, nullptr);
    g_vt.cache = vt_cache_create(dir, (int64_t)budgetBytes);
    env->ReleaseStringUTFChars(jdir, dir);
    return g_vt.cache ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (g_vt.cache) {
        vt_cache_free(g_vt.cache);
        g_vt.cache = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheClear(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (g_vt.cache) vt_cache_clear(g_vt.cache);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheSetBudget(
        JNIEnv *, jobject, jlong bytes) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (g_vt.cache) vt_cache_set_budget(g_vt.cache, (int64_t)bytes);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheStatsJson(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    json info;
    if (!g_vt.cache) {
        info["initialized"] = false;
        return env->NewStringUTF(info.dump().c_str());
    }
    info["initialized"]  = true;
    info["total_bytes"]  = vt_cache_total_bytes(g_vt.cache);
    info["budget_bytes"] = vt_cache_get_budget(g_vt.cache);
    info["entry_count"]  = vt_cache_count(g_vt.cache);
    info["hits"]         = vt_cache_hits(g_vt.cache);
    info["misses"]       = vt_cache_misses(g_vt.cache);
    return env->NewStringUTF(info.dump().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheListEntriesJson(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (!g_vt.cache) return env->NewStringUTF("[]");
    int32_t count = 0;
    auto * entries = vt_cache_list(g_vt.cache, &count);
    json arr = json::array();
    for (int32_t i = 0; i < count; i++) {
        char hex[VT_CACHE_HASH_BYTES * 2 + 1];
        for (int j = 0; j < VT_CACHE_HASH_BYTES; j++) {
            snprintf(hex + j*2, 3, "%02x", entries[i].hash[j]);
        }
        arr.push_back({
            {"hash",           hex},
            {"n_tokens",       entries[i].n_tokens},
            {"n_embd",         entries[i].n_embd},
            {"size_bytes",     entries[i].size_bytes},
            {"last_access_ms", entries[i].last_access_ms},
        });
    }
    vt_cache_list_free(entries);
    return env->NewStringUTF(arr.dump().c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVtCacheRemove(
        JNIEnv * env, jobject, jbyteArray jhash) {
    std::lock_guard<std::mutex> lock(g_vt.mutex);
    if (!g_vt.cache || !jhash) return JNI_FALSE;
    if (env->GetArrayLength(jhash) != VT_CACHE_HASH_BYTES) return JNI_FALSE;
    uint8_t hash[VT_CACHE_HASH_BYTES];
    env->GetByteArrayRegion(jhash, 0, VT_CACHE_HASH_BYTES, (jbyte *)hash);
    return vt_cache_remove(g_vt.cache, hash) ? JNI_TRUE : JNI_FALSE;
}

// VLM-KV cache state. Stores the LLM context state (KV cache, n_past, etc.)
// captured at the post-image-chunk boundary so subsequent queries with the
// same image + system prompt + chat template skip both the vision encoder AND
// the LLM image-prefill — TTFT drops from ~9s to a few hundred ms.
static struct {
    vlm_kv_cache_t * cache = nullptr;
    std::mutex       mutex;
} g_vkv;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheInit(
        JNIEnv * env, jobject, jstring jdir, jlong budgetBytes) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (g_vkv.cache) {
        vlm_kv_cache_free(g_vkv.cache);
        g_vkv.cache = nullptr;
    }
    if (!jdir) return JNI_FALSE;
    const char * dir = env->GetStringUTFChars(jdir, nullptr);
    g_vkv.cache = vlm_kv_cache_create(dir, (int64_t)budgetBytes);
    env->ReleaseStringUTFChars(jdir, dir);
    return g_vkv.cache ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (g_vkv.cache) {
        vlm_kv_cache_free(g_vkv.cache);
        g_vkv.cache = nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheClear(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (g_vkv.cache) vlm_kv_cache_clear(g_vkv.cache);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheSetBudget(
        JNIEnv *, jobject, jlong bytes) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (g_vkv.cache) vlm_kv_cache_set_budget(g_vkv.cache, (int64_t)bytes);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheStatsJson(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    json info;
    if (!g_vkv.cache) {
        info["initialized"] = false;
        return env->NewStringUTF(info.dump().c_str());
    }
    info["initialized"]  = true;
    info["total_bytes"]  = vlm_kv_cache_total_bytes(g_vkv.cache);
    info["budget_bytes"] = vlm_kv_cache_get_budget(g_vkv.cache);
    info["entry_count"]  = vlm_kv_cache_count(g_vkv.cache);
    info["hits"]         = vlm_kv_cache_hits(g_vkv.cache);
    info["misses"]       = vlm_kv_cache_misses(g_vkv.cache);
    return env->NewStringUTF(info.dump().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheListEntriesJson(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (!g_vkv.cache) return env->NewStringUTF("[]");
    int32_t count = 0;
    auto * entries = vlm_kv_cache_list(g_vkv.cache, &count);
    json arr = json::array();
    for (int32_t i = 0; i < count; i++) {
        char hex[VLM_KV_CACHE_HASH_BYTES * 2 + 1];
        for (int j = 0; j < VLM_KV_CACHE_HASH_BYTES; j++) {
            snprintf(hex + j*2, 3, "%02x", entries[i].hash[j]);
        }
        arr.push_back({
            {"hash",           hex},
            {"n_tokens",       entries[i].n_tokens},
            {"size_bytes",     entries[i].size_bytes},
            {"last_access_ms", entries[i].last_access_ms},
        });
    }
    vlm_kv_cache_list_free(entries);
    return env->NewStringUTF(arr.dump().c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmKvCacheRemove(
        JNIEnv * env, jobject, jbyteArray jhash) {
    std::lock_guard<std::mutex> lock(g_vkv.mutex);
    if (!g_vkv.cache || !jhash) return JNI_FALSE;
    if (env->GetArrayLength(jhash) != VLM_KV_CACHE_HASH_BYTES) return JNI_FALSE;
    uint8_t hash[VLM_KV_CACHE_HASH_BYTES];
    env->GetByteArrayRegion(jhash, 0, VLM_KV_CACHE_HASH_BYTES, (jbyte *)hash);
    return vlm_kv_cache_remove(g_vkv.cache, hash) ? JNI_TRUE : JNI_FALSE;
}

// ── Backend diagnostics ─────────────────────────────────────────────────
//
// Lists every ggml backend that auto-registered at startup. Purely
// informational — does NOT route ops to GPU. Per-op routing requires
// upstream llama.cpp changes (ggml_backend_sched configured with the
// right backend set, plus an op-routing callback) and is intentionally
// not exposed here. See VLM.md "What's NOT shipping yet" for the
// trade-off discussion.
extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeListBackendsJson(JNIEnv * env, jobject) {
    ensure_backend_init();

    json regs   = json::array();
    json devs   = json::array();

    const size_t n_reg = ggml_backend_reg_count();
    for (size_t i = 0; i < n_reg; i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        if (!r) continue;
        const char * name = ggml_backend_reg_name(r);
        regs.push_back({ {"name", name ? name : "?"} });
    }

    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) continue;

        ggml_backend_dev_props p{};
        ggml_backend_dev_get_props(d, &p);

        const char * type_str = "?";
        switch (p.type) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   type_str = "cpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type_str = "gpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type_str = "igpu";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type_str = "accel"; break;
        }

        devs.push_back({
            {"name",         p.name        ? p.name        : "?"},
            {"description",  p.description ? p.description : ""},
            {"type",         type_str},
            {"memory_free",  (uint64_t)p.memory_free},
            {"memory_total", (uint64_t)p.memory_total},
            {"async",        p.caps.async},
            {"events",       p.caps.events},
        });
    }

    json out;
    out["backends"] = regs;
    out["devices"]  = devs;
    return env->NewStringUTF(out.dump().c_str());
}

// VLM (Vision Language Model) state. The mtmd projector context binds n_threads
// at init time; if the caller switches thread mode after loading, the projector
// keeps the old count. Reload via releaseVlmProjector() + loadVlmProjector() to
// pick up the new mode.
static struct {
    mtmd_context * ctx = nullptr;
    std::mutex     mutex;
} g_vlm;

// imageMinTokens / imageMaxTokens (-1 = model default) cap the mmproj token
// budget for the *overview* image. For LFM2-VL the per-tile count is a
// compile-time constant in clip.cpp; lowering imageMaxTokens does not cap it.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjector(
        JNIEnv * env, jobject, jstring jpath, jint nThreads,
        jint imageMinTokens, jint imageMaxTokens) {

    std::lock_guard<std::mutex> lock(g_vlm.mutex);

    if (!g_state.model) {
        LOGE("VLM: text model must be loaded first");
        return JNI_FALSE;
    }

    if (g_vlm.ctx) {
        mtmd_free(g_vlm.ctx);
        g_vlm.ctx = nullptr;
    }

    const char * path = env->GetStringUTFChars(jpath, nullptr);

    auto params = mtmd_context_params_default();
    // GPU dispatch is NOT controlled by this flag — it's a coarse hammer that
    // affects more than just the vision encoder. Real heterogeneous routing
    // (per-op CPU/GPU placement based on M) is wired through a custom
    // ggml_backend_sched in the load path. Leave this off.
    params.use_gpu       = false;
    params.n_threads     = nThreads > 0
        ? nThreads
        : tn_thread_config_for_mode((tn_thread_mode)g_state.thread_mode).n_threads_batch;
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

    LOGI("VLM: projector loaded (vision=%d, audio=%d, img_tokens=[%d..%d], threads=%d)",
         mtmd_support_vision(g_vlm.ctx), mtmd_support_audio(g_vlm.ctx),
         imageMinTokens, imageMaxTokens, params.n_threads);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjectorFromFd(
        JNIEnv * env, jobject thiz, jint fd, jint nThreads,
        jint imageMinTokens, jint imageMaxTokens) {

    if (fd < 0) {
        LOGE("VLM: invalid file descriptor: %d", fd);
        return JNI_FALSE;
    }

    int owned_fd = dup(fd);
    if (owned_fd < 0) {
        LOGE("VLM: dup() failed for fd %d: %s", fd, strerror(errno));
        return JNI_FALSE;
    }

    if (lseek(owned_fd, 0, SEEK_CUR) == (off_t)-1) {
        LOGE("VLM: fd %d is not seekable: %s", fd, strerror(errno));
        close(owned_fd);
        return JNI_FALSE;
    }

    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", owned_fd);
    jstring  jpath  = env->NewStringUTF(fd_path);
    jboolean result = Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmLoadProjector(
        env, thiz, jpath, nThreads, imageMinTokens, imageMaxTokens);
    env->DeleteLocalRef(jpath);

    close(owned_fd);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_vlm.mutex);
    if (g_vlm.ctx) {
        mtmd_free(g_vlm.ctx);
        g_vlm.ctx = nullptr;
        LOGI("VLM: projector released");
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmIsLoaded(JNIEnv *, jobject) {
    return g_vlm.ctx != nullptr ? JNI_TRUE : JNI_FALSE;
}

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

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmGetDefaultMarker(JNIEnv * env, jobject) {
    return env->NewStringUTF(mtmd_default_marker());
}

// Generates a response from text + images. messagesJson must include the
// default media marker where images appear; raw image bytes (JPEG/PNG) are
// passed in imageDataArray. The KV cache is cleared at the start — VLM does
// not support multi-turn context reuse here.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGUFNativeLib_nativeVlmGenerateStream(
        JNIEnv * env, jobject,
        jstring jmessagesJson,
        jobjectArray imageDataArray,
        jobjectArray vtKeysArray,    // optional byte[32][] parallel to images; null = no caching
        jbyteArray   vlmKvKeyArray,  // optional byte[32] for VLM-KV cache; null = no caching
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

    // VT cache integration. Each non-text chunk corresponds (by position) to
    // the next entry in imageDataArray / vtKeysArray. We hash-key per image,
    // not per chunk, because a single image always produces a single chunk
    // through mtmd_tokenize. Multi-image prompts get one cache slot each.
    const int32_t n_embd_inp = llama_model_n_embd_inp(g_state.model);
    int img_idx = 0;
    int32_t n_cache_hits = 0;
    int32_t n_cache_misses = 0;
    std::vector<float> cached_embd_buf;   // reused across image chunks on hit

    const size_t n_chunks_total = mtmd_input_chunks_size(chunks);

    // ── VLM-KV cache integration ──────────────────────────────────────────
    // The big TTFT win: cache the LLM context state *after* the last image
    // chunk has been decoded. On a hit, restore that state and skip everything
    // up through the last image — including the ~9s image-prefill llama_decode.
    //
    // Key derivation is the caller's job; we only require a 32-byte hash. The
    // canonical key derivation is in GGMLEngine.computeVlmKvKey() and includes
    // image bytes, projector path, image_max_tokens, system prompt, and chat
    // template prefix.
    uint8_t  vlm_kv_key[VLM_KV_CACHE_HASH_BYTES];
    bool     have_vlm_kv_key = false;
    if (vlmKvKeyArray && env->GetArrayLength(vlmKvKeyArray) == VLM_KV_CACHE_HASH_BYTES) {
        env->GetByteArrayRegion(vlmKvKeyArray, 0, VLM_KV_CACHE_HASH_BYTES,
                                (jbyte *)vlm_kv_key);
        have_vlm_kv_key = true;
    }

    // Stable snapshot of the cache pointer for the duration of this call.
    // Init/release on the host thread between calls is safe; mid-generation
    // release is undefined (host responsibility).
    vlm_kv_cache_t * vlm_kv = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_vkv.mutex);
        vlm_kv = g_vkv.cache;
    }

    // Boundary index: first chunk *after* the last image chunk. If there are
    // no image chunks, the cache is irrelevant and we leave this at 0.
    int resume_chunk_idx = 0;
    for (int i = (int)n_chunks_total - 1; i >= 0; i--) {
        if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i))
                != MTMD_INPUT_CHUNK_TYPE_TEXT) {
            resume_chunk_idx = i + 1;
            break;
        }
    }

    bool    vlm_kv_restored = false;
    int32_t vlm_kv_restored_tokens = 0;
    if (have_vlm_kv_key && vlm_kv && resume_chunk_idx > 0) {
        int32_t peek_n  = 0;
        size_t  peek_sz = 0;
        if (vlm_kv_cache_peek(vlm_kv, vlm_kv_key, &peek_n, &peek_sz) && peek_sz > 0) {
            std::vector<uint8_t> blob(peek_sz);
            size_t  got_sz = 0;
            int32_t got_n  = 0;
            if (vlm_kv_cache_lookup(vlm_kv, vlm_kv_key,
                                    blob.data(), peek_sz, &got_sz, &got_n)) {
                const size_t set_ret = llama_state_seq_set_data(
                    g_state.ctx, blob.data(), got_sz, /*seq_id=*/0);
                if (set_ret > 0) {
                    new_n_past             = (llama_pos)got_n;
                    vlm_kv_restored        = true;
                    vlm_kv_restored_tokens = got_n;
                    LOGI("VLM-KV cache HIT (restored %d tokens, %zu bytes) — "
                         "skipping chunks [0, %d)",
                         got_n, got_sz, resume_chunk_idx);
                } else {
                    LOGW("VLM-KV cache: state_set_data failed (geometry mismatch?), "
                         "falling back to fresh decode");
                }
            }
        }
    }

    if (g_onVlmKvCacheStatus && have_vlm_kv_key) {
        env->CallVoidMethod(callback, g_onVlmKvCacheStatus,
                            (jboolean)vlm_kv_restored, (jint)vlm_kv_restored_tokens);
    }

    // When restored, skip chunks [0, resume_chunk_idx) but still advance
    // img_idx past any image chunks we skipped — vtKeysArray slots align
    // by image-chunk index.
    const size_t start_ci = vlm_kv_restored ? (size_t)resume_chunk_idx : 0;
    if (vlm_kv_restored) {
        for (size_t i = 0; i < (size_t)resume_chunk_idx; i++) {
            const enum mtmd_input_chunk_type t = mtmd_input_chunk_get_type(
                mtmd_input_chunks_get(chunks, i));
            if (t != MTMD_INPUT_CHUNK_TYPE_TEXT) img_idx++;
        }
    }

    for (size_t ci = start_ci; ci < n_chunks_total && eval_result == 0; ci++) {
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
            const int32_t n_tok = (int32_t)mtmd_input_chunk_get_n_tokens(chunk);
            n_image_tokens += n_tok;
            const size_t n_floats = (size_t)n_tok * (size_t)n_embd_inp;

            // Pull this image's cache key (if any) before the slow path
            uint8_t  vt_key[VT_CACHE_HASH_BYTES];
            bool     have_key = false;
            if (vtKeysArray && img_idx < env->GetArrayLength(vtKeysArray)) {
                jbyteArray jkey = (jbyteArray)env->GetObjectArrayElement(vtKeysArray, img_idx);
                if (jkey && env->GetArrayLength(jkey) == VT_CACHE_HASH_BYTES) {
                    env->GetByteArrayRegion(jkey, 0, VT_CACHE_HASH_BYTES, (jbyte *)vt_key);
                    have_key = true;
                }
                if (jkey) env->DeleteLocalRef(jkey);
            }

            float * embd = nullptr;
            bool    cache_hit = false;

            if (have_key && g_vt.cache) {
                cached_embd_buf.assign(n_floats, 0.0f);
                int32_t hit_nt = 0, hit_ne = 0;
                if (vt_cache_lookup(g_vt.cache, vt_key,
                                    cached_embd_buf.data(), n_floats,
                                    &hit_nt, &hit_ne)) {
                    if (hit_nt == n_tok && hit_ne == n_embd_inp) {
                        embd = cached_embd_buf.data();
                        cache_hit = true;
                        n_cache_hits++;
                        LOGI("VT cache HIT for image %d (%d tokens × %d embd)",
                             img_idx, hit_nt, hit_ne);
                    } else {
                        // Geometry mismatch (different image_max_tokens, etc.).
                        // Fall through to encode path; treat as miss.
                        LOGW("VT cache geometry mismatch: cached %dx%d vs expected %dx%d",
                             hit_nt, hit_ne, n_tok, n_embd_inp);
                    }
                }
            }

            if (g_onVlmCacheStatus && have_key) {
                env->CallVoidMethod(callback, g_onVlmCacheStatus,
                                    (jboolean)cache_hit, (jint)n_tok, (jint)n_embd_inp);
            }

            if (!cache_hit) {
                // Vision / audio encoder forward
                const int64_t t_enc0 = llama_time_us();
                eval_result = mtmd_encode_chunk(g_vlm.ctx, chunk);
                t_encode_us += llama_time_us() - t_enc0;
                if (eval_result != 0) break;

                embd = mtmd_get_output_embd(g_vlm.ctx);
                n_cache_misses++;

                if (have_key && g_vt.cache && embd) {
                    vt_cache_store(g_vt.cache, vt_key, embd, n_tok, n_embd_inp);
                }
            }

            const int64_t t_dec0 = llama_time_us();
            eval_result = mtmd_helper_decode_image_chunk(
                g_vlm.ctx, g_state.ctx, chunk, embd,
                new_n_past, 0, vlm_n_batch, &new_n_past);
            t_decode_us += llama_time_us() - t_dec0;

            img_idx++;
        }

        if (g_onProgress && n_chunks_total > 1) {
            float p = 0.1f + 0.4f * ((float)(ci + 1) / (float)n_chunks_total);
            env->CallVoidMethod(callback, g_onProgress, p);
        }

        // Boundary save: just decoded the last image chunk on a fresh path.
        // Persist the LLM context state so the next call with the same key
        // can skip everything up through here.
        if (have_vlm_kv_key && !vlm_kv_restored && vlm_kv && eval_result == 0
                && (int)ci + 1 == resume_chunk_idx) {
            const size_t blob_size = llama_state_seq_get_size(g_state.ctx, /*seq_id=*/0);
            if (blob_size > 0) {
                std::vector<uint8_t> blob(blob_size);
                const size_t written = llama_state_seq_get_data(
                    g_state.ctx, blob.data(), blob_size, /*seq_id=*/0);
                if (written > 0) {
                    vlm_kv_cache_store(vlm_kv, vlm_kv_key,
                                       blob.data(), written, (int32_t)new_n_past);
                }
            }
        }
    }

    LOGI("VLM chunks: %d image chunks (%d cache hit, %d miss)",
         img_idx, n_cache_hits, n_cache_misses);

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
            size_t unsent_len   = generated_text.size() - unsent_start;
            // Hot path: this runs on every token. The std::string ctor copies
            // unsent_len bytes; antiprompt.find_stop only reads within that
            // window. Per-token cost is negligible (typically <128 bytes).
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

// KV Eviction Policy

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

// Error Tracker JNI

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
