/**
 * Optimized native_generate.cpp for llama.cpp JNI bindings
 *
 * Key optimizations:
 * 1. Immediate token streaming (no batching/buffering delay)
 * 2. Updated llama.cpp API usage (llama_memory_* instead of deprecated llama_kv_cache_*)
 * 3. Reduced JNI overhead with better caching
 * 4. Efficient UTF-8 handling with minimal buffering
 * 5. Optimized exception checking frequency
 *
 * Compatible with llama.cpp b7400+ (January 2026)
 */

#include "state/model_state.h"
#include "state/embedding_state.h"
#include "utils/jni_utils.h"
#include "utils/utf8_utils.h"
#include "chat/chat_template.h"

#include "llama.h"
#include "ggml-backend.h"
#include "cpu/cpu_helper.h"
#include "utils/logger.h"
#include "tool_calling/tool_call_state.h"

#include <jni.h>
#include <string>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

static std::mutex g_init_mtx;
static std::atomic<bool> g_stop_requested{false};

struct GenerationMetrics {
    int32_t total_tokens = 0;
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    int64_t time_to_first_token_ms = 0;
    int64_t total_time_ms = 0;
    float tokens_per_second = 0.0f;
};


namespace {

// Pre-cached JNI references for minimal lookup overhead
    struct JniCallbackCache {
        jclass cls = nullptr;
        jmethodID onToken = nullptr;
        jmethodID onError = nullptr;
        jmethodID onToolCall = nullptr;
        jmethodID onDone = nullptr;
        jmethodID onMetrics = nullptr;

        // Metrics class cache
        jclass metricsClass = nullptr;
        jmethodID metricsConstructor = nullptr;

        bool initialized = false;

        void init(JNIEnv *env, jobject callback) {
            if (initialized) return;

            jclass tempCls = env->GetObjectClass(callback);
            if (!tempCls) {
                LOG_ERROR("JniCallbackCache: Failed to get callback class");
                return;
            }

            cls = static_cast<jclass>(env->NewGlobalRef(tempCls));
            env->DeleteLocalRef(tempCls);

            onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
            onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
            onToolCall = env->GetMethodID(cls, "onToolCall",
                                          "(Ljava/lang/String;Ljava/lang/String;)V");
            onDone = env->GetMethodID(cls, "onDone", "()V");
            onMetrics = env->GetMethodID(cls, "onMetrics",
                                         "(Lcom/mp/ai_gguf/models/DecodingMetrics;)V");

            // Cache metrics class
            jclass tempMetricsCls = env->FindClass("com/mp/ai_gguf/models/DecodingMetrics");
            if (tempMetricsCls) {
                metricsClass = static_cast<jclass>(env->NewGlobalRef(tempMetricsCls));
                metricsConstructor = env->GetMethodID(metricsClass, "<init>", "(IIIFJJ)V");
                env->DeleteLocalRef(tempMetricsCls);
            }

            initialized = true;
        }

        void release(JNIEnv *env) {
            if (cls) {
                env->DeleteGlobalRef(cls);
                cls = nullptr;
            }
            if (metricsClass) {
                env->DeleteGlobalRef(metricsClass);
                metricsClass = nullptr;
            }
            initialized = false;
        }
    };

// Thread-local callback cache for multi-threaded safety
    static thread_local JniCallbackCache g_callback_cache;

/**
 * Send a single token immediately to the Java callback
 * This is the core streaming function - no buffering, immediate delivery
 */
    inline void send_token_immediate(JNIEnv *env, jobject callback, const std::string &token) {
        if (token.empty() || !callback) return;

        g_callback_cache.init(env, callback);
        if (!g_callback_cache.onToken) return;

        // Convert UTF-8 to Java string
        // Note: We use NewStringUTF for ASCII-compatible tokens (most cases)
        // For full UTF-8 with surrogates, we need proper conversion
        jstring jtoken = nullptr;

        // Fast path for ASCII-only tokens (most common case)
        bool is_ascii = true;
        for (unsigned char c: token) {
            if (c >= 0x80) {
                is_ascii = false;
                break;
            }
        }

        if (is_ascii) {
            jtoken = env->NewStringUTF(token.c_str());
        } else {
            // Full UTF-8 to UTF-16 conversion for non-ASCII
            jtoken = utf8::to_jstring_immediate(env, token);
        }

        if (jtoken) {
            env->CallVoidMethod(callback, g_callback_cache.onToken, jtoken);
            env->DeleteLocalRef(jtoken);
        }
    }

    inline void send_error(JNIEnv *env, jobject callback, const char *msg) {
        if (!callback) return;

        g_callback_cache.init(env, callback);
        if (!g_callback_cache.onError) return;

        jstring jmsg = env->NewStringUTF(msg ? msg : "<unknown error>");
        env->CallVoidMethod(callback, g_callback_cache.onError, jmsg);
        env->DeleteLocalRef(jmsg);
    }

    inline void send_toolcall(JNIEnv *env, jobject callback, const std::string &name,
                              const std::string &payload) {
        if (!callback) return;

        g_callback_cache.init(env, callback);
        if (!g_callback_cache.onToolCall) return;

        jstring jname = env->NewStringUTF(name.c_str());
        jstring jpayload = utf8::to_jstring_immediate(env, payload);

        env->CallVoidMethod(callback, g_callback_cache.onToolCall, jname, jpayload);

        env->DeleteLocalRef(jname);
        env->DeleteLocalRef(jpayload);
    }

    inline void send_done(JNIEnv *env, jobject callback) {
        if (!callback) return;

        g_callback_cache.init(env, callback);
        if (!g_callback_cache.onDone) return;

        env->CallVoidMethod(callback, g_callback_cache.onDone);
    }

    inline void send_metrics(JNIEnv *env, jobject callback, const GenerationMetrics &metrics) {
        if (!callback) return;

        g_callback_cache.init(env, callback);
        if (!g_callback_cache.onMetrics || !g_callback_cache.metricsClass) return;

        jobject metricsObj = env->NewObject(g_callback_cache.metricsClass,
                                            g_callback_cache.metricsConstructor,
                                            metrics.total_tokens, metrics.prompt_tokens,
                                            metrics.generated_tokens, metrics.tokens_per_second,
                                            metrics.time_to_first_token_ms, metrics.total_time_ms);

        if (metricsObj) {
            env->CallVoidMethod(callback, g_callback_cache.onMetrics, metricsObj);
            env->DeleteLocalRef(metricsObj);
        }
    }

} // anonymous namespace

class Utf8StreamDecoder {
public:
    void reset() {
        pending_bytes_.clear();
    }

    /**
     * Process raw token bytes and return complete UTF-8 characters
     * Incomplete sequences are buffered until the next token completes them
     */
    std::string decode(const std::string &raw_bytes) {
        if (raw_bytes.empty()) return {};

        // Prepend any pending bytes from previous tokens
        std::string input;
        if (!pending_bytes_.empty()) {
            input = pending_bytes_ + raw_bytes;
            pending_bytes_.clear();
        } else {
            input = raw_bytes;
        }

        std::string complete;
        complete.reserve(input.size());

        size_t i = 0;
        while (i < input.size()) {
            unsigned char c = static_cast<unsigned char>(input[i]);
            size_t char_len = utf8_char_length(c);

            if (char_len == 0) {
                // Invalid start byte - skip
                ++i;
                continue;
            }

            // Check if we have all bytes for this character
            if (i + char_len > input.size()) {
                // Incomplete sequence - save for next token
                pending_bytes_.assign(input.data() + i, input.size() - i);
                break;
            }

            // Validate continuation bytes
            bool valid = true;
            for (size_t j = 1; j < char_len; ++j) {
                unsigned char cont = static_cast<unsigned char>(input[i + j]);
                if ((cont & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                complete.append(input.data() + i, char_len);
                i += char_len;
            } else {
                // Invalid sequence - skip start byte
                ++i;
            }
        }

        return complete;
    }

    /**
     * Flush any remaining pending bytes (call at end of generation)
     */
    std::string flush() {
        std::string result;
        if (!pending_bytes_.empty()) {
            // Return replacement character for incomplete sequence
            result = "\xEF\xBF\xBD"; // U+FFFD
            pending_bytes_.clear();
        }
        return result;
    }

    bool has_pending() const { return !pending_bytes_.empty(); }

private:
    std::string pending_bytes_;

    static size_t utf8_char_length(unsigned char c) {
        if ((c & 0x80) == 0x00) return 1;      // 0xxxxxxx - ASCII
        if ((c & 0xE0) == 0xC0) return 2;      // 110xxxxx
        if ((c & 0xF0) == 0xE0) return 3;      // 1110xxxx
        if ((c & 0xF8) == 0xF0) return 4;      // 11110xxx
        return 0; // Invalid start byte
    }
};

/**
 * Initialize or update grammar sampler for tool calls
 * Uses caching to avoid rebuilds when tools haven't changed
 */
static void maybe_init_grammar() {
    if (!g_state.tools_enabled) return;

    // Use cached grammar management
    g_state.update_grammar_if_needed();
}

static const char *get_model_architecture(llama_model *model) {
    if (!model) return nullptr;

    static char arch_buf[128] = {0};
    int32_t len = llama_model_meta_val_str(model, "general.architecture", arch_buf,
                                           sizeof(arch_buf));

    return (len > 0) ? arch_buf : nullptr;
}

static const char *get_model_name(llama_model *model) {
    if (!model) return nullptr;

    static char name_buf[256] = {0};
    int32_t len = llama_model_meta_val_str(model, "general.name", name_buf, sizeof(name_buf));

    return (len > 0) ? name_buf : nullptr;
}

static const char *get_model_description(llama_model *model) {
    if (!model) return nullptr;

    static char desc_buf[512] = {0};
    int32_t len = llama_model_meta_val_str(model, "general.description", desc_buf,
                                           sizeof(desc_buf));

    return (len > 0) ? desc_buf : nullptr;
}


extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeGenerateStream(JNIEnv *env, jobject, jstring jprompt,
                                                        jint max_tokens, jobject jcallback) {
    // Validate model state
    if (!g_state.is_ready()) {
        send_error(env, jcallback, "Model not initialized");
        return JNI_FALSE;
    }

    // Prepare for new generation
    LOG_INFO("Starting new generation, calling prepare_for_generation");
    g_state.prepare_for_generation();
    LOG_INFO("prepare_for_generation completed");
    g_stop_requested.store(false, std::memory_order_relaxed);

    static std::mutex g_generate_mtx;
    std::lock_guard<std::mutex> lock(g_generate_mtx);

    // Initialize metrics
    GenerationMetrics metrics;
    auto start_time = std::chrono::steady_clock::now();
    bool first_token_generated = false;

    // Get user message
    const std::string user_msg = utf8::from_jstring(env, jprompt);

    // Get vocab
    const llama_vocab *vocab = llama_model_get_vocab(g_state.model);
    if (!vocab) {
        send_error(env, jcallback, "Failed to get vocab");
        return JNI_FALSE;
    }

    // Build system prompt with tool preamble if needed
    std::string system = g_state.system_prompt;
    if (g_state.tools_enabled && !g_state.tools_json.empty()) {
        system.reserve(system.size() + g_state.tools_json.size() + 256);
        system += "\n";
        system += chat::build_tool_preamble(g_state.tools_json);
    }

    // Apply chat template
    const std::string prompt = chat::apply_template(g_state.model, system, user_msg,
                                                    g_state.chat_template_override,
                                                    true // add generation prompt
    );

    LOG_INFO("Rendered prompt size=%zu", prompt.size());

    // Tokenize prompt
    std::vector<llama_token> prompt_toks = g_state.tokenize(prompt);
    if (prompt_toks.empty()) {
        send_error(env, jcallback, "Tokenization failed");
        return JNI_FALSE;
    }

    metrics.prompt_tokens = static_cast<int32_t>(prompt_toks.size());
    metrics.total_tokens = metrics.prompt_tokens;

    // Check context size
    int32_t available = g_state.ctx_size - metrics.prompt_tokens - 8;
    if (available <= 0) {
        send_error(env, jcallback, "Context overflow - shorten your prompt");
        return JNI_TRUE;
    }

    int32_t to_generate = (max_tokens > 0) ? static_cast<int32_t>(max_tokens) : 128;
    to_generate = std::min(to_generate, available);

    // Decode prompt (prefill phase)
    if (!g_state.decode_prompt(prompt_toks)) {
        jni::on_error(env, jcallback, "Decoding prompt failed");
        return JNI_TRUE;
    }

    // Verify we have logits available
    float *logits = llama_get_logits(g_state.ctx);
    if (!logits) {
        LOG_ERROR("No logits available after prompt decode");
        jni::on_error(env, jcallback, "No logits available");
        return JNI_TRUE;
    }

    // Initialize streaming components
    ToolCallState tool_state;
    Utf8StreamDecoder utf8_decoder;

    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);

    // Single-token batch for autoregressive generation
    llama_batch single = llama_batch_init(1, 0, 1);

    // Exception check interval - less frequent for better performance
    // Check every 64 tokens or so
    constexpr int EXCEPTION_CHECK_INTERVAL = 64;
    bool has_exception = false;

    // ========================================================================
    // LAZY TOOL DETECTION OPTIMIZATION
    // Only engage tool call parsing after seeing potential tool call start
    // This reduces overhead when generating normal text
    // ========================================================================
    bool tool_detection_active = g_state.tools_enabled;
    bool seen_non_whitespace = false;
    bool definitely_not_tool_call = false;

    // ========================================================================
    // MAIN GENERATION LOOP - IMMEDIATE TOKEN STREAMING
    // ========================================================================
    for (int i = 0; i < to_generate && !g_stop_requested.load(std::memory_order_relaxed); ++i) {
        // Use -1 which means "last token with logits enabled"
        // BUT we must ensure decode succeeded first
        int current_pos = static_cast<int>(prompt_toks.size()) + i;
        if (current_pos >= g_state.ctx_size - 1) {
            LOG_ERROR("Context overflow at pos %d, ctx_size %d", current_pos, g_state.ctx_size);
            jni::on_error(env, jcallback, "Context size exceeded");
            break;
        }

        llama_token tok = llama_sampler_sample(g_state.sampler, g_state.ctx, -1);

        // Check for invalid token
        if (tok < 0) {
            LOG_ERROR("llama_sampler_sample returned invalid token");
            jni::on_error(env, jcallback, "Sampling failed");
            break;
        }

        llama_sampler_accept(g_state.sampler, tok);

        // Handle first-token edge case
        if (i == 0 && (tok == eos || tok == eot)) {
            tok = g_state.space_token();
        }

        // Check for end of generation
        if (tok == eos || tok == eot) {
            break;
        }

        // Record time to first token
        if (!first_token_generated) {
            auto first_token_time = std::chrono::steady_clock::now();
            metrics.time_to_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    first_token_time - start_time).count();
            first_token_generated = true;
        }

        // Update metrics
        metrics.generated_tokens++;
        metrics.total_tokens++;

        // Detokenize and decode UTF-8
        std::string raw_piece = g_state.detokenize_single(tok);
        std::string complete_chars = utf8_decoder.decode(raw_piece);

        // ====================================================================
        // IMMEDIATE TOKEN STREAMING - NO BUFFERING
        // ====================================================================
        if (!complete_chars.empty()) {
            bool tool_complete = false;

            // Check for tool calls if tools are enabled
            if (g_state.tools_enabled) {
                tool_complete = tool_state.accumulate(complete_chars);
                if (tool_complete) {
                    std::string name, payload;
                    if (tool_state.extract_tool_call(name, payload)) {
                        send_toolcall(env, jcallback, name, payload);
                        break;
                    }
                    tool_state.reset();
                }
            }

            // Stream token immediately (unless collecting a tool call)
            if (!tool_state.is_collecting()) {
                send_token_immediate(env, jcallback, complete_chars);
            }
        }

        // Prepare batch for next token prediction
        single.n_tokens = 1;
        single.token[0] = tok;
        single.pos[0] = static_cast<int32_t>(prompt_toks.size() + i);
        single.n_seq_id[0] = 1;
        single.seq_id[0][0] = 0;
        single.logits[0] = true;

        // Decode (forward pass for next token)
        int decode_result = llama_decode(g_state.ctx, single);
        if (decode_result != 0) {
            LOG_ERROR("llama_decode failed with code %d at token %d, pos %d", decode_result, i,
                      (int) (prompt_toks.size() + i));
            jni::on_error(env, jcallback, "llama_decode failed during generation");
            break;
        }

        // Periodic exception check (less frequent for performance)
        if ((i & (EXCEPTION_CHECK_INTERVAL - 1)) == 0) {
            if (env->ExceptionCheck()) {
                LOG_ERROR("Java exception during callback - aborting");
                env->ExceptionClear();
                has_exception = true;
                break;
            }
        }
    }

    // ========================================================================
    // CLEANUP AND FINAL OUTPUT
    // ========================================================================

    // Flush any remaining UTF-8 bytes
    std::string remaining = utf8_decoder.flush();
    if (!remaining.empty()) {
        send_token_immediate(env, jcallback, remaining);
    }

    // Calculate final metrics
    auto end_time = std::chrono::steady_clock::now();
    metrics.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

    if (metrics.total_time_ms > 0 && metrics.generated_tokens > 0) {
        metrics.tokens_per_second =
                (metrics.generated_tokens * 1000.0f) / static_cast<float>(metrics.total_time_ms);
    }

    // Clean up batch
    llama_batch_free(single);

    // Send completion callbacks (unless exception occurred)
    if (!has_exception) {
        send_metrics(env, jcallback, metrics);
        send_done(env, jcallback);
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeLoadModelFromFd(JNIEnv *env, jobject, jint fd,
                                                         jint jthreads, jint ctxSize, jfloat temp,
                                                         jint topK, jfloat topP, jfloat minP,
                                                         jint mirostat, jfloat mirostatTau,
                                                         jfloat mirostatEta, jint seed) {
    std::lock_guard<std::mutex> lk(g_init_mtx);

    g_state.release();
    llama_backend_init();

    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;

    LOG_INFO("Initializing model from fd=%d (threads=%d, ctx=%d)", fd, nthreads, ctxSize);

    // Get file size via fstat
    struct stat st;
    if (fstat(fd, &st) != 0) {
        LOG_ERROR("fstat failed: %s", strerror(errno));
        return JNI_FALSE;
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    LOG_INFO("File size: %zu bytes", file_size);

    // Model parameters - no mmap for FD-based loading
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;  // FD loading doesn't support mmap
    mparams.use_mlock = false;
    mparams.check_tensors = false;  // Skip tensor validation for faster load

    // Use the native FD loading API (added to llama.cpp for Android SAF support)
    // This avoids the /proc/self/fd/ workaround that fails on Android
    g_state.model = llama_model_load_from_fd(fd, file_size, mparams);

    if (!g_state.model) {
        LOG_ERROR("llama_model_load_from_fd failed");
        g_state.release();
        return JNI_FALSE;
    }

    LOG_INFO("Model loaded successfully from fd");

    // Context setup
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 512;
    cparams.n_ubatch = 256;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOG_ERROR("Failed to create context");
        g_state.release();
        return JNI_FALSE;
    }

    g_state.ctx_size = ctxSize;
    g_state.batch_size = cparams.n_batch;

    g_state.rebuild_sampler(static_cast<int>(topK), topP, temp, minP, mirostat, mirostatTau,
                            mirostatEta, seed);
    g_state.warmup_context();
    maybe_init_grammar();

    LOG_INFO("Model initialized successfully from fd");
    return JNI_TRUE;
}




extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeLoadModel(JNIEnv *env, jobject, jstring jpath,
                                                   jint jthreads, jint ctxSize, jfloat temp,
                                                   jint topK, jfloat topP, jfloat minP,
                                                   jint mirostat, jfloat mirostatTau,
                                                   jfloat mirostatEta, jint seed) {
    std::lock_guard<std::mutex> lk(g_init_mtx);

    const std::string path = utf8::from_jstring(env, jpath);
    g_state.release();
    llama_backend_init();

    // Detect optimal thread count
    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;

    LOG_INFO("Initializing model '%s' (threads=%d, ctx=%d)", path.c_str(), nthreads, ctxSize);

    // Model parameters
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;       // CPU-only for Android
    mparams.use_mmap = true;        // Memory-map for efficiency
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    // Load model
    g_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_state.model) {
        LOG_ERROR("Failed to load model '%s'", path.c_str());
        g_state.release();
        return JNI_FALSE;
    }

    // Context parameters
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = 512;
    cparams.n_ubatch = 256;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;    // CPU-only
    cparams.n_seq_max = 1;
    cparams.no_perf = false;

    // Create context
    g_state.ctx = llama_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        LOG_ERROR("Failed to create context");
        g_state.release();
        return JNI_FALSE;
    }

    g_state.ctx_size = ctxSize;
    g_state.batch_size = cparams.n_batch;

    // Build sampler chain
    g_state.rebuild_sampler(static_cast<int>(topK), topP, temp, minP, mirostat, mirostatTau,
                            mirostatEta, seed);

    // Warm up context
    g_state.warmup_context();

    // Initialize grammar if tools are enabled
    maybe_init_grammar();

    LOG_INFO("Model initialized successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeRelease(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_state.release();
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeSetSystemPrompt(JNIEnv *env, jobject, jstring jprompt) {
    g_state.system_prompt = utf8::from_jstring(env, jprompt);
    LOG_INFO("System prompt updated (%zu bytes)", g_state.system_prompt.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeSetChatTemplate(JNIEnv *env, jobject, jstring jtemplate) {
    g_state.chat_template_override = utf8::from_jstring(env, jtemplate);
    LOG_INFO("Chat template override set (%zu bytes)", g_state.chat_template_override.size());
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeSetToolsJson(JNIEnv *env, jobject, jstring jtools) {
    g_state.tools_json = utf8::from_jstring(env, jtools);
    g_state.tools_enabled = !g_state.tools_json.empty();
    LOG_INFO("Tools JSON set (%zu bytes), enabled=%d", g_state.tools_json.size(),
             static_cast<int>(g_state.tools_enabled));
    maybe_init_grammar();
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeStopGeneration(JNIEnv *, jobject) {
    g_stop_requested.store(true, std::memory_order_relaxed);
    LOG_INFO("Stop generation requested");
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeClearMemory(JNIEnv *, jobject) {
    if (g_state.ctx) {
        // Updated API: llama_memory_* instead of llama_kv_cache_*
        llama_memory_t mem = llama_get_memory(g_state.ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        LOG_INFO("KV cache cleared");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_llamaPrintTimings(JNIEnv *, jobject) {
    llama_print_system_info();
    llama_perf_context_print(g_state.ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeGetModelInfo(JNIEnv *env, jobject thiz) {
    if (!g_state.model) return env->NewStringUTF("{}");

    const llama_vocab *vocab = llama_model_get_vocab(g_state.model);
    std::ostringstream json;
    json << "{";
    bool first = true;

    auto add_string_field = [&](const char *key, const char *value) {
        if (value && *value) {
            if (!first) json << ",";
            json << "\"" << key << "\":\"" << chat::json_escape(value) << "\"";
            first = false;
        }
    };

    auto add_int_field = [&](const char *key, int value) {
        if (value > 0) {
            if (!first) json << ",";
            json << "\"" << key << "\":" << value;
            first = false;
        }
    };

    // Model identity - only if they exist
    const char *arch = get_model_architecture(g_state.model);
    const char *name = get_model_name(g_state.model);
    const char *desc = get_model_description(g_state.model);

    add_string_field("architecture", arch);
    add_string_field("name", name);
    add_string_field("description", desc);

    // Model dimensions - only positive values
    if (vocab) {
        add_int_field("n_vocab", llama_vocab_n_tokens(vocab));
    }

    add_int_field("n_ctx_train", llama_model_n_ctx_train(g_state.model));
    add_int_field("n_embd", llama_model_n_embd(g_state.model));
    add_int_field("n_layer", llama_model_n_layer(g_state.model));
    add_int_field("n_head", llama_model_n_head(g_state.model));
    add_int_field("n_head_kv", llama_model_n_head_kv(g_state.model));

    // Vocabulary tokens - only if vocab exists
    if (vocab) {
        add_int_field("bos", llama_vocab_bos(vocab));
        add_int_field("eos", llama_vocab_eos(vocab));
        add_int_field("eot", llama_vocab_eot(vocab));
        add_int_field("nl", llama_vocab_nl(vocab));

        // Vocab type - only known types
        const char *vocab_type = nullptr;
        switch (llama_vocab_type(vocab)) {
            case LLAMA_VOCAB_TYPE_SPM:
                vocab_type = "spm";
                break;
            case LLAMA_VOCAB_TYPE_BPE:
                vocab_type = "bpe";
                break;
            case LLAMA_VOCAB_TYPE_WPM:
                vocab_type = "wpm";
                break;
            case LLAMA_VOCAB_TYPE_NONE:
                vocab_type = "NONE";
                break;
            case LLAMA_VOCAB_TYPE_UGM:
                vocab_type = "UGM";
                break;
            case LLAMA_VOCAB_TYPE_RWKV:
                vocab_type = "RWKV";
                break;
            case LLAMA_VOCAB_TYPE_PLAMO2:
                vocab_type = "PLAMO2";
                break;
        }
        add_string_field("vocab_type", vocab_type);
    }

    // Chat template - only if it exists in model
    const char *tmpl = llama_model_chat_template(g_state.model, nullptr);
    if (tmpl && *tmpl) {
        add_string_field("chat_template", tmpl);

        // Detect template type only from existing template
        std::string template_str(tmpl);
        const char *template_type = nullptr;

        if (template_str.find("<|im_start|>") != std::string::npos) {
            template_type = "chatml";
        } else if (template_str.find("<start_of_turn>") != std::string::npos) {
            template_type = "gemma";
        } else if (template_str.find("[INST]") != std::string::npos) {
            template_type = "llama";
        } else if (template_str.find("<|system|>") != std::string::npos) {
            template_type = "phi";
        }

        add_string_field("template_type", template_type);
    }

    // System info - only if it exists
    const char *sys_info = llama_print_system_info();
    add_string_field("system", sys_info);

    json << "}";
    return env->NewStringUTF(json.str().c_str());
}

// ============================================================================
// EMBEDDING MODEL FUNCTIONS
// ============================================================================

namespace {
    // Pre-cached JNI references for embedding callbacks
    struct EmbeddingCallbackCache {
        jclass cls = nullptr;
        jmethodID onProgress = nullptr;
        jmethodID onComplete = nullptr;
        jmethodID onError = nullptr;

        // EmbeddingResult class cache
        jclass resultClass = nullptr;
        jmethodID resultConstructor = nullptr;

        bool initialized = false;

        void init(JNIEnv *env, jobject callback) {
            if (initialized) return;

            jclass tempCls = env->GetObjectClass(callback);
            if (!tempCls) {
                LOG_ERROR("EmbeddingCallbackCache: Failed to get callback class");
                return;
            }

            cls = static_cast<jclass>(env->NewGlobalRef(tempCls));
            env->DeleteLocalRef(tempCls);

            onProgress = env->GetMethodID(cls, "onProgress", "(FII)V");
            onComplete = env->GetMethodID(cls, "onComplete",
                                          "(Lcom/mp/ai_gguf/models/EmbeddingResult;)V");
            onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");

            // Cache EmbeddingResult class
            jclass tempResultCls = env->FindClass("com/mp/ai_gguf/models/EmbeddingResult");
            if (tempResultCls) {
                resultClass = static_cast<jclass>(env->NewGlobalRef(tempResultCls));
                resultConstructor = env->GetMethodID(resultClass, "<init>",
                                                     "([FILjava/lang/String;IJ)V");
                env->DeleteLocalRef(tempResultCls);
            }

            initialized = true;
        }

        void release(JNIEnv *env) {
            if (cls) {
                env->DeleteGlobalRef(cls);
                cls = nullptr;
            }
            if (resultClass) {
                env->DeleteGlobalRef(resultClass);
                resultClass = nullptr;
            }
            initialized = false;
        }
    };

    static thread_local EmbeddingCallbackCache g_embedding_callback_cache;

    inline void send_embedding_progress(JNIEnv *env, jobject callback,
                                        float progress, int32_t current, int32_t total) {
        if (!callback) return;

        g_embedding_callback_cache.init(env, callback);
        if (!g_embedding_callback_cache.onProgress) return;

        env->CallVoidMethod(callback, g_embedding_callback_cache.onProgress,
                            progress, current, total);
    }

    inline void send_embedding_complete(JNIEnv *env, jobject callback,
                                        const EmbeddingOutput &output) {
        if (!callback) return;

        g_embedding_callback_cache.init(env, callback);
        if (!g_embedding_callback_cache.onComplete ||
            !g_embedding_callback_cache.resultClass) return;

        // Convert embeddings to jfloatArray
        jfloatArray jembeddings = env->NewFloatArray(output.dimension);
        if (!jembeddings) {
            LOG_ERROR("Failed to create float array for embeddings");
            return;
        }
        env->SetFloatArrayRegion(jembeddings, 0, output.dimension, output.embeddings.data());

        // Get pooling type string
        const char *pooling_str = "mean";
        switch (output.pooling) {
            case PoolingType::NONE:
                pooling_str = "none";
                break;
            case PoolingType::MEAN:
                pooling_str = "mean";
                break;
            case PoolingType::CLS:
                pooling_str = "cls";
                break;
            case PoolingType::LAST:
                pooling_str = "last";
                break;
            case PoolingType::MAX:
                pooling_str = "max";
                break;
        }
        jstring jpooling = env->NewStringUTF(pooling_str);

        // Create EmbeddingResult object
        jobject result = env->NewObject(g_embedding_callback_cache.resultClass,
                                        g_embedding_callback_cache.resultConstructor,
                                        jembeddings, output.dimension, jpooling,
                                        output.num_tokens, output.time_ms);

        if (result) {
            env->CallVoidMethod(callback, g_embedding_callback_cache.onComplete, result);
            env->DeleteLocalRef(result);
        }

        env->DeleteLocalRef(jembeddings);
        env->DeleteLocalRef(jpooling);
    }

    inline void send_embedding_error(JNIEnv *env, jobject callback, const char *msg) {
        if (!callback) return;

        g_embedding_callback_cache.init(env, callback);
        if (!g_embedding_callback_cache.onError) return;

        jstring jmsg = env->NewStringUTF(msg ? msg : "<unknown error>");
        env->CallVoidMethod(callback, g_embedding_callback_cache.onError, jmsg);
        env->DeleteLocalRef(jmsg);
    }

} // anonymous namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeLoadEmbeddingModelFromFd(JNIEnv *env, jobject,
                                                                   jint fd,
                                                                   jint jthreads,
                                                                   jint ctxSize) {
    std::lock_guard<std::mutex> lk(g_init_mtx);

    g_embedding_state.release();
    llama_backend_init();

    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;

    LOG_INFO("Loading embedding model from fd=%d (threads=%d, ctx=%d)", fd, nthreads, ctxSize);

    // Get file size via fstat
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        LOG_ERROR("fstat failed: %s", strerror(errno));
        return JNI_FALSE;
    }
    auto file_size = static_cast<size_t>(st.st_size);
    LOG_INFO("File size: %zu bytes", file_size);

    // Model parameters - no mmap for FD-based loading
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;  // FD loading doesn't support mmap
    mparams.use_mlock = false;
    mparams.check_tensors = false;  // Skip tensor validation for faster load

    // Load model from FD
    g_embedding_state.model = llama_model_load_from_fd(fd, file_size, mparams);
    if (!g_embedding_state.model) {
        LOG_ERROR("llama_model_load_from_fd failed for embedding model");
        g_embedding_state.release();
        return JNI_FALSE;
    }

    LOG_INFO("Embedding model loaded successfully from fd");

    // Context parameters - optimized for embeddings
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = g_embedding_state.batch_size;
    cparams.n_ubatch = g_embedding_state.batch_size;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;
    cparams.embeddings = true;  // CRITICAL: Enable embeddings mode

    // Create context
    g_embedding_state.ctx = llama_init_from_model(g_embedding_state.model, cparams);
    if (!g_embedding_state.ctx) {
        LOG_ERROR("Failed to create embedding context");
        g_embedding_state.release();
        return JNI_FALSE;
    }

    g_embedding_state.ctx_size = ctxSize;
    g_embedding_state.n_threads = nthreads;

    // Get embedding dimension
    g_embedding_state.n_embd = g_embedding_state.get_embedding_dimension();
    LOG_INFO("Embedding dimension: %d", g_embedding_state.n_embd);

    // Detect pooling type from model
    g_embedding_state.pooling_type = g_embedding_state.detect_pooling_type();

    LOG_INFO("Embedding model initialized successfully from fd");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeLoadEmbeddingModel(JNIEnv *env, jobject,
                                                             jstring jpath,
                                                             jint jthreads,
                                                             jint ctxSize) {
    std::lock_guard<std::mutex> lk(g_init_mtx);

    const std::string path = utf8::from_jstring(env, jpath);
    g_embedding_state.release();
    llama_backend_init();

    // Detect optimal thread count
    int phys = count_physical_cores();
    int nthreads = (jthreads > 0) ? static_cast<int>(jthreads) : phys;

    LOG_INFO("Loading embedding model '%s' (threads=%d, ctx=%d)", path.c_str(), nthreads,
             ctxSize);

    // Model parameters - optimized for embeddings
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only for Android
    mparams.use_mmap = true;   // Memory-map for efficiency
    mparams.use_mlock = false;
    mparams.check_tensors = true;

    // Load model
    g_embedding_state.model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_embedding_state.model) {
        LOG_ERROR("Failed to load embedding model '%s'", path.c_str());
        g_embedding_state.release();
        return JNI_FALSE;
    }

    // Context parameters - optimized for embeddings
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = ctxSize;
    cparams.n_batch = g_embedding_state.batch_size;
    cparams.n_ubatch = g_embedding_state.batch_size;
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    cparams.offload_kqv = false;
    cparams.n_seq_max = 1;
    cparams.no_perf = false;
    cparams.embeddings = true;  // CRITICAL: Enable embeddings mode

    // Create context
    g_embedding_state.ctx = llama_init_from_model(g_embedding_state.model, cparams);
    if (!g_embedding_state.ctx) {
        LOG_ERROR("Failed to create embedding context");
        g_embedding_state.release();
        return JNI_FALSE;
    }

    g_embedding_state.ctx_size = ctxSize;
    g_embedding_state.n_threads = nthreads;

    // Get embedding dimension
    g_embedding_state.n_embd = g_embedding_state.get_embedding_dimension();
    LOG_INFO("Embedding dimension: %d", g_embedding_state.n_embd);

    // Detect pooling type from model
    g_embedding_state.pooling_type = g_embedding_state.detect_pooling_type();

    LOG_INFO("Embedding model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeEncodeText(JNIEnv *env, jobject, jstring jtext,
                                                     jboolean normalize, jobject jcallback) {
    if (!g_embedding_state.is_ready()) {
        send_embedding_error(env, jcallback, "Embedding model not initialized");
        return JNI_FALSE;
    }

    const std::string text = utf8::from_jstring(env, jtext);
    if (text.empty()) {
        send_embedding_error(env, jcallback, "Empty text provided");
        return JNI_FALSE;
    }

    LOG_INFO("Encoding text (%zu bytes)", text.size());

    // Create progress callback that forwards to Java
    auto progress_callback = [env, jcallback](float progress, int32_t current, int32_t total) {
        send_embedding_progress(env, jcallback, progress, current, total);
    };

    // Encode text
    EmbeddingOutput output = g_embedding_state.encode(text, normalize, progress_callback);

    // Check if encoding succeeded
    if (output.embeddings.empty()) {
        send_embedding_error(env, jcallback, "Encoding failed");
        return JNI_FALSE;
    }

    // Send result to callback
    send_embedding_complete(env, jcallback, output);

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeReleaseEmbeddingModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_embedding_state.release();
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeGetEmbeddingModelInfo(JNIEnv *env, jobject) {
    if (!g_embedding_state.model) return env->NewStringUTF("{}");

    std::ostringstream json;
    json << "{";
    bool first = true;

    auto add_string_field = [&](const char *key, const char *value) {
        if (value && *value) {
            if (!first) json << ",";
            json << "\"" << key << "\":\"" << chat::json_escape(value) << "\"";
            first = false;
        }
    };

    auto add_int_field = [&](const char *key, int value) {
        if (value > 0) {
            if (!first) json << ",";
            json << "\"" << key << "\":" << value;
            first = false;
        }
    };

    // Model identity
    const char *arch = get_model_architecture(g_embedding_state.model);
    const char *name = get_model_name(g_embedding_state.model);
    const char *desc = get_model_description(g_embedding_state.model);

    add_string_field("architecture", arch);
    add_string_field("name", name);
    add_string_field("description", desc);

    // Embedding-specific info
    add_int_field("n_embd", g_embedding_state.n_embd);
    add_int_field("n_ctx", g_embedding_state.ctx_size);

    // Pooling type
    const char *pooling_str = "unknown";
    switch (g_embedding_state.pooling_type) {
        case PoolingType::NONE:
            pooling_str = "none";
            break;
        case PoolingType::MEAN:
            pooling_str = "mean";
            break;
        case PoolingType::CLS:
            pooling_str = "cls";
            break;
        case PoolingType::LAST:
            pooling_str = "last";
            break;
        case PoolingType::MAX:
            pooling_str = "max";
            break;
    }
    add_string_field("pooling", pooling_str);

    json << "}";
    return env->NewStringUTF(json.str().c_str());
}

// ============================================================================
// TOOL CALLING SDK FUNCTIONS
// ============================================================================

extern "C" JNIEXPORT jstring JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeGetModelArchitecture(JNIEnv *env, jobject) {
    if (!g_state.model) {
        return env->NewStringUTF("");
    }

    const char *arch = get_model_architecture(g_state.model);
    return env->NewStringUTF(arch ? arch : "");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeIsToolCallingSupported(JNIEnv *env, jobject) {
    if (!g_state.model) {
        return JNI_FALSE;
    }

    const char *arch = get_model_architecture(g_state.model);
    if (!arch) {
        return JNI_FALSE;
    }

    // Only Qwen models support tool calling
    std::string arch_str(arch);
    std::transform(arch_str.begin(), arch_str.end(), arch_str.begin(), ::tolower);

    return (arch_str.find("qwen") != std::string::npos) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeEnableToolCalling(JNIEnv *env, jobject, jstring jtools) {
    if (!g_state.model) {
        LOG_ERROR("Cannot enable tool calling: model not loaded");
        return JNI_FALSE;
    }

    // Check if model supports tool calling
    const char *arch = get_model_architecture(g_state.model);
    if (!arch) {
        LOG_ERROR("Cannot enable tool calling: failed to get model architecture");
        return JNI_FALSE;
    }

    std::string arch_str(arch);
    std::transform(arch_str.begin(), arch_str.end(), arch_str.begin(), ::tolower);

    if (arch_str.find("qwen") == std::string::npos) {
        LOG_ERROR("Tool calling only supported for Qwen models, got: %s", arch);
        return JNI_FALSE;
    }

    // Set tools JSON
    const std::string tools_json = utf8::from_jstring(env, jtools);
    g_state.tools_json = tools_json;
    g_state.tools_enabled = !tools_json.empty();

    // Set tool calling system prompt
    const std::string tool_system_prompt =
        "You are a function-calling assistant. When tools are available, respond ONLY with a JSON object in this EXACT format:\n"
        "\n"
        "{\n"
        "  \"tool_calls\": [{\n"
        "    \"name\": \"toolName\",\n"
        "    \"arguments\": {\n"
        "      \"param1\": \"value1\",\n"
        "      \"param2\": \"value2\"\n"
        "    }\n"
        "  }]\n"
        "}\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. Use \"arguments\" as an object containing all parameters\n"
        "2. NEVER put parameters directly in the tool_calls object\n"
        "3. NEVER include any text before or after the JSON\n"
        "4. The \"arguments\" field must be a JSON object, not a string\n"
        "5. Match parameter names exactly as defined in the tool schema\n"
        "\n"
        "If no tool is needed, respond with plain text.";

    g_state.system_prompt = tool_system_prompt;

    // Set Qwen chat template with tool calling support
    const std::string qwen_template =
        "{%- if professional is defined or emotional is defined -%}\n"
        "<|im_start|>system\n"
        "The assistant should modulate style accordingly while staying accurate.\n"
        "<|im_end|>\n"
        "{%- endif -%}\n"
        "{%- if gbnf is defined and gbnf|length > 0 -%}\n"
        "<|im_start|>system\n"
        "The assistant's NEXT message MUST conform to the following GBNF grammar.\n"
        "If a token would violate the grammar, do not emit it.\n"
        "<GBNF>\n"
        "{{ gbnf }}\n"
        "</GBNF>\n"
        "<|im_end|>\n"
        "{%- endif -%}\n"
        "{%- for m in messages -%}\n"
        "<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}\n"
        "<|im_end|>\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "<|im_start|>assistant\n"
        "{%- endif -%}";

    g_state.chat_template_override = qwen_template;

    // Initialize grammar
    maybe_init_grammar();

    LOG_INFO("Tool calling enabled for Qwen model (%zu bytes of tools JSON)", tools_json.size());
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeDisableToolCalling(JNIEnv *env, jobject) {
    g_state.tools_json.clear();
    g_state.tools_enabled = false;
    g_state.system_prompt.clear();
    g_state.chat_template_override.clear();

    LOG_INFO("Tool calling disabled, reverted to default model settings");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_mp_ai_1gguf_GGUFNativeLib_nativeIsToolCallingEnabled(JNIEnv *env, jobject) {
    return g_state.tools_enabled ? JNI_TRUE : JNI_FALSE;
}
