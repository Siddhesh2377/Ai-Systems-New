#include "engine.h"
#include "utils/jni_utils.h"
#include "cpu/cpu_helper.h"

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <jni.h>
#include <android/log.h>
#include <dlfcn.h>

#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#define TAG "ai_gguf"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define JNI_FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_mp_ai_1gguf_GGUFNativeLib_##name

using json = nlohmann::json;

// ── Global State ─────────────────────────────────────────────────────────

namespace {

std::unique_ptr<engine::Context> g_ctx;
std::unique_ptr<engine::Context> g_embed_ctx;
std::mutex g_init_mtx;
std::mutex g_gen_mtx;
std::atomic<bool> g_stop{false};
std::atomic<bool> g_backends_loaded{false};

std::string g_system_prompt;
std::string g_chat_template;
std::string g_tools_json;
std::string g_tool_choice = "auto";
bool g_tool_calling = false;
int g_grammar_mode = 0;
std::vector<std::string> g_stop_strings;
engine::SamplingConfig g_sampling;
std::string g_grammar;

void ensure_backends(const std::string& hint = "") {
    if (g_backends_loaded.exchange(true)) return;

    if (!hint.empty()) {
        LOGI("Loading backends from: %s", hint.c_str());
        ggml_backend_load_all_from_path(hint.c_str());
        return;
    }

    Dl_info info;
    if (dladdr((void*)&g_ctx, &info) && info.dli_fname) {
        std::string p(info.dli_fname);
        auto pos = p.rfind('/');
        if (pos != std::string::npos) {
            std::string dir = p.substr(0, pos);
            LOGI("Loading backends from: %s", dir.c_str());
            ggml_backend_load_all_from_path(dir.c_str());
            return;
        }
    }
    ggml_backend_load_all();
}

engine::CompletionRequest build_request(const std::string& prompt, int n_predict) {
    engine::CompletionRequest req;
    req.prompt = prompt;
    req.sampling = g_sampling;
    req.sampling.stop = g_stop_strings;
    if (!g_grammar.empty()) req.sampling.grammar = g_grammar;
    req.n_predict = n_predict;
    if (g_tool_calling && !g_tools_json.empty()) {
        req.tools = g_tools_json;
        req.tool_choice = g_tool_choice;
    }
    req.chat_template = g_chat_template;
    return req;
}

engine::CompletionRequest build_request_multi(const std::string& messages_json, int n_predict) {
    engine::CompletionRequest req;
    req.sampling = g_sampling;
    req.sampling.stop = g_stop_strings;
    if (!g_grammar.empty()) req.sampling.grammar = g_grammar;
    req.n_predict = n_predict;

    try {
        auto arr = json::parse(messages_json);
        if (!g_system_prompt.empty()) {
            bool has_system = false;
            for (auto& m : arr) {
                if (m.value("role", "") == "system") { has_system = true; break; }
            }
            if (!has_system) {
                json sys = {{"role", "system"}, {"content", g_system_prompt}};
                arr.insert(arr.begin(), sys);
            }
        }
        for (auto& m : arr) {
            engine::ChatMessage cm;
            cm.role = m.value("role", "");
            cm.content = m.value("content", "");
            cm.tool_call_id = m.value("tool_call_id", "");
            cm.tool_name = m.value("name", "");
            req.messages.push_back(std::move(cm));
        }
    } catch (...) {
        LOGE("Failed to parse messages JSON");
    }

    if (g_tool_calling && !g_tools_json.empty()) {
        req.tools = g_tools_json;
        req.tool_choice = g_tool_choice;
    }
    req.chat_template = g_chat_template;
    return req;
}

// ── Thermal Monitoring ──────────────────────────────────────────────────

struct ThermalZone {
    std::string path;
    std::string type;
};

std::vector<ThermalZone> g_thermal_zones;
bool g_thermal_scanned = false;

void scan_thermal_zones() {
    if (g_thermal_scanned) return;
    g_thermal_scanned = true;

    DIR* dir = opendir("/sys/class/thermal");
    if (!dir) return;
    struct dirent* e;
    while ((e = readdir(dir))) {
        if (strncmp(e->d_name, "thermal_zone", 12) != 0) continue;
        std::string zp = std::string("/sys/class/thermal/") + e->d_name;
        std::ifstream tf(zp + "/type");
        if (!tf) continue;
        std::string zt;
        std::getline(tf, zt);
        bool gpu = zt.find("gpu") != std::string::npos || zt.find("GPU") != std::string::npos ||
                   zt.find("G3D") != std::string::npos || zt.find("gpuss") != std::string::npos;
        bool npu = zt.find("nsp") != std::string::npos || zt.find("npu") != std::string::npos ||
                   zt.find("NPU") != std::string::npos || zt.find("cdsp") != std::string::npos;
        if (gpu || npu) g_thermal_zones.push_back({zp, zt});
    }
    closedir(dir);
}

int read_temp(const std::string& path) {
    std::ifstream f(path + "/temp");
    int t = -1;
    if (f) f >> t;
    return t;
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// MODEL LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jboolean, nativeLoadModel)(
    JNIEnv* env, jobject,
    jstring jpath, jint nCtx, jint nBatch, jint nUbatch,
    jint nThreads,
    jboolean flashAttn, jboolean useMmap, jboolean useMlock,
    jstring jCacheTypeK, jstring jCacheTypeV,
    jstring jBackendPath)
{
    std::lock_guard<std::mutex> lk(g_init_mtx);

    std::string path = jni::to_string(env, jpath);
    std::string backend_path = jni::to_string(env, jBackendPath);
    std::string cache_k = jni::to_string(env, jCacheTypeK);
    std::string cache_v = jni::to_string(env, jCacheTypeV);

    ensure_backends(backend_path);
    llama_backend_init();

    g_ctx = std::make_unique<engine::Context>();

    engine::EngineConfig cfg;
    cfg.model_path   = path;
    cfg.n_ctx        = nCtx;
    cfg.n_batch      = nBatch;
    cfg.n_ubatch     = nUbatch;
    cfg.n_threads    = nThreads;
    cfg.n_gpu_layers = 0; // CPU-only (OpenCL disabled for GGUF)
    cfg.flash_attn   = flashAttn;
    cfg.use_mmap     = useMmap;
    cfg.use_mlock    = useMlock;
    cfg.cache_type_k = cache_k.empty() ? "q8_0" : cache_k;
    cfg.cache_type_v = cache_v.empty() ? "q8_0" : cache_v;

    LOGI("Loading model: %s (ctx=%d, threads=%d, flash=%d, kv=%s/%s)",
         path.c_str(), nCtx, nThreads, (int)flashAttn,
         cfg.cache_type_k.c_str(), cfg.cache_type_v.c_str());

    bool ok = g_ctx->load(cfg, [](float p) {
        LOGI("Loading: %.0f%%", p * 100.0f);
    });

    if (!ok) {
        LOGE("Failed to load model: %s", path.c_str());
        g_ctx.reset();
        return JNI_FALSE;
    }

    LOGI("Model loaded: %s", g_ctx->model_info().c_str());
    return JNI_TRUE;
}

JNI_FN(jboolean, nativeLoadModelFromFd)(
    JNIEnv* env, jobject,
    jint fd, jint nCtx, jint nBatch, jint nUbatch,
    jint nThreads,
    jboolean flashAttn,
    jstring jCacheTypeK, jstring jCacheTypeV,
    jstring jBackendPath)
{
    std::lock_guard<std::mutex> lk(g_init_mtx);

    // Get file size via fstat
    struct stat st;
    if (fstat(fd, &st) != 0) {
        LOGE("fstat(fd=%d) failed: %s", fd, strerror(errno));
        return JNI_FALSE;
    }
    size_t file_size = static_cast<size_t>(st.st_size);

    std::string backend_path = jni::to_string(env, jBackendPath);
    std::string cache_k = jni::to_string(env, jCacheTypeK);
    std::string cache_v = jni::to_string(env, jCacheTypeV);

    ensure_backends(backend_path);
    llama_backend_init();

    g_ctx = std::make_unique<engine::Context>();

    engine::EngineConfig cfg;
    cfg.n_ctx        = nCtx;
    cfg.n_batch      = nBatch;
    cfg.n_ubatch     = nUbatch;
    cfg.n_threads    = nThreads;
    cfg.flash_attn   = flashAttn;
    cfg.cache_type_k = cache_k.empty() ? "q8_0" : cache_k;
    cfg.cache_type_v = cache_v.empty() ? "q8_0" : cache_v;

    LOGI("Loading model from FD %d (size=%zu, ctx=%d, threads=%d, flash=%d, kv=%s/%s)",
         fd, file_size, nCtx, nThreads, (int)flashAttn,
         cfg.cache_type_k.c_str(), cfg.cache_type_v.c_str());

    bool ok = g_ctx->load_from_fd(fd, file_size, cfg, [](float p) {
        LOGI("Loading: %.0f%%", p * 100.0f);
    });

    if (!ok) {
        LOGE("Failed to load model from FD %d", fd);
        g_ctx.reset();
        return JNI_FALSE;
    }

    LOGI("Model loaded from FD: %s", g_ctx->model_info().c_str());
    return JNI_TRUE;
}

JNI_FN(jboolean, nativeRelease)(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_ctx.reset();
    g_system_prompt.clear();
    g_chat_template.clear();
    g_tools_json.clear();
    g_tool_calling = false;
    g_stop_strings.clear();
    g_grammar.clear();
    return JNI_TRUE;
}

JNI_FN(jboolean, nativeIsLoaded)(JNIEnv*, jobject) {
    return g_ctx && g_ctx->is_loaded() ? JNI_TRUE : JNI_FALSE;
}

// ═══════════════════════════════════════════════════════════════════════════
// SAMPLING CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(void, nativeSetSampling)(
    JNIEnv*, jobject,
    jfloat temp, jint topK, jfloat topP, jfloat minP,
    jfloat repeatPenalty, jint penaltyLastN,
    jfloat freqPenalty, jfloat presPenalty, jint seed,
    jfloat dryMult, jfloat dryBase, jint dryAllowed, jint dryLastN,
    jfloat xtcProb, jfloat xtcThresh,
    jint mirostat, jfloat miroTau, jfloat miroEta)
{
    g_sampling.temperature      = temp;
    g_sampling.top_k            = topK;
    g_sampling.top_p            = topP;
    g_sampling.min_p            = minP;
    g_sampling.repeat_penalty   = repeatPenalty;
    g_sampling.penalty_last_n   = penaltyLastN;
    g_sampling.frequency_penalty = freqPenalty;
    g_sampling.presence_penalty = presPenalty;
    g_sampling.seed             = static_cast<uint32_t>(seed);
    g_sampling.dry_multiplier   = dryMult;
    g_sampling.dry_base         = dryBase;
    g_sampling.dry_allowed_length = dryAllowed;
    g_sampling.dry_penalty_last_n = dryLastN;
    g_sampling.xtc_probability  = xtcProb;
    g_sampling.xtc_threshold    = xtcThresh;
    g_sampling.mirostat         = mirostat;
    g_sampling.mirostat_tau     = miroTau;
    g_sampling.mirostat_eta     = miroEta;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jboolean, nativeGenerateStream)(
    JNIEnv* env, jobject, jstring jprompt, jint maxTokens, jobject jcallback)
{
    if (!g_ctx || !g_ctx->is_loaded()) {
        LOGE("nativeGenerateStream: g_ctx=%p is_loaded=%d — aborting",
             g_ctx.get(), g_ctx ? g_ctx->is_loaded() : 0);
        return JNI_FALSE;
    }

    jni::Callback cb;
    if (!cb.init(env, jcallback)) {
        LOGE("nativeGenerateStream: callback init failed — aborting");
        return JNI_FALSE;
    }

    std::lock_guard<std::mutex> lk(g_gen_mtx);
    g_stop.store(false);

    std::string prompt = jni::to_string(env, jprompt);
    auto req = build_request(prompt, maxTokens);

    auto t_start = ggml_time_us();
    int64_t t_first = 0;
    int n_tokens = 0;

    auto result = g_ctx->completion(req, [&](const std::string& token) -> bool {
        if (g_stop.load(std::memory_order_relaxed)) return false;
        if (n_tokens == 0) t_first = ggml_time_us();
        n_tokens++;
        cb.token(token);
        return !env->ExceptionCheck();
    });

    auto t_end = ggml_time_us();
    float total_ms = (float)(t_end - t_start) / 1000.0f;
    float ttft_ms = t_first > 0 ? (float)(t_first - t_start) / 1000.0f : 0.0f;
    float tps = total_ms > 0 ? (float)result.tokens_predicted / (total_ms / 1000.0f) : 0.0f;

    for (auto& tc : result.tool_calls) {
        cb.tool_call(tc.name, tc.arguments);
    }

    cb.metrics(tps, ttft_ms, total_ms, result.tokens_evaluated, result.tokens_predicted,
               0, 0, 0, 0);
    cb.done();
    return JNI_TRUE;
}

JNI_FN(jboolean, nativeGenerateStreamMultiTurn)(
    JNIEnv* env, jobject, jstring jmessages, jint maxTokens, jobject jcallback)
{
    LOGI("nativeGenerateStreamMultiTurn: entry (maxTokens=%d)", maxTokens);

    if (!g_ctx || !g_ctx->is_loaded()) {
        LOGE("nativeGenerateStreamMultiTurn: g_ctx=%p is_loaded=%d — aborting",
             g_ctx.get(), g_ctx ? g_ctx->is_loaded() : 0);
        return JNI_FALSE;
    }

    jni::Callback cb;
    if (!cb.init(env, jcallback)) {
        LOGE("nativeGenerateStreamMultiTurn: callback init failed — aborting");
        return JNI_FALSE;
    }

    LOGI("nativeGenerateStreamMultiTurn: acquiring gen mutex...");
    std::lock_guard<std::mutex> lk(g_gen_mtx);
    g_stop.store(false);

    std::string messages_json = jni::to_string(env, jmessages);
    LOGI("nativeGenerateStreamMultiTurn: messages_json length=%zu", messages_json.size());
    auto req = build_request_multi(messages_json, maxTokens);
    LOGI("nativeGenerateStreamMultiTurn: request built (messages=%zu, prompt='%s')",
         req.messages.size(), req.prompt.substr(0, 100).c_str());

    LOGI("nativeGenerateStreamMultiTurn: calling completion...");
    auto t_start = ggml_time_us();
    int64_t t_first = 0;
    int n_tokens = 0;

    auto result = g_ctx->completion(req, [&](const std::string& token) -> bool {
        if (g_stop.load(std::memory_order_relaxed)) return false;
        if (n_tokens == 0) t_first = ggml_time_us();
        n_tokens++;
        cb.token(token);
        return !env->ExceptionCheck();
    });

    auto t_end = ggml_time_us();
    float total_ms = (float)(t_end - t_start) / 1000.0f;
    float ttft_ms = t_first > 0 ? (float)(t_first - t_start) / 1000.0f : 0.0f;
    float tps = total_ms > 0 ? (float)result.tokens_predicted / (total_ms / 1000.0f) : 0.0f;

    LOGI("nativeGenerateStreamMultiTurn: done — eval=%d pred=%d tps=%.1f total=%.0fms",
         result.tokens_evaluated, result.tokens_predicted, tps, total_ms);

    for (auto& tc : result.tool_calls) {
        cb.tool_call(tc.name, tc.arguments);
    }

    cb.metrics(tps, ttft_ms, total_ms, result.tokens_evaluated, result.tokens_predicted,
               0, 0, 0, 0);
    cb.done();
    return JNI_TRUE;
}

JNI_FN(void, nativeStopGeneration)(JNIEnv*, jobject) {
    g_stop.store(true, std::memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(void, nativeSetSystemPrompt)(JNIEnv* env, jobject, jstring jp) {
    g_system_prompt = jni::to_string(env, jp);
}

JNI_FN(void, nativeSetChatTemplate)(JNIEnv* env, jobject, jstring jt) {
    g_chat_template = jni::to_string(env, jt);
}

JNI_FN(void, nativeSetToolsJson)(JNIEnv* env, jobject, jstring jt) {
    g_tools_json = jni::to_string(env, jt);
}

JNI_FN(void, nativeSetToolChoice)(JNIEnv* env, jobject, jstring jc) {
    g_tool_choice = jni::to_string(env, jc);
}

JNI_FN(void, nativeEnableToolCalling)(JNIEnv*, jobject, jboolean enabled) {
    g_tool_calling = enabled;
}

JNI_FN(jboolean, nativeIsToolCallingEnabled)(JNIEnv*, jobject) {
    return g_tool_calling ? JNI_TRUE : JNI_FALSE;
}

JNI_FN(void, nativeSetGrammarMode)(JNIEnv*, jobject, jint mode) {
    g_grammar_mode = mode;
}

JNI_FN(void, nativeSetGrammar)(JNIEnv* env, jobject, jstring jg) {
    g_grammar = jni::to_string(env, jg);
}

JNI_FN(void, nativeSetStopStrings)(JNIEnv* env, jobject, jobjectArray jstops) {
    g_stop_strings.clear();
    if (!jstops) return;
    int len = env->GetArrayLength(jstops);
    for (int i = 0; i < len; i++) {
        auto js = (jstring)env->GetObjectArrayElement(jstops, i);
        if (js) {
            g_stop_strings.push_back(jni::to_string(env, js));
            env->DeleteLocalRef(js);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERVENTIONS
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(void, nativeSetLogitBias)(JNIEnv* env, jobject, jintArray jids, jfloatArray jbiases) {
    g_sampling.logit_bias.clear();
    if (!jids || !jbiases) return;
    int len = env->GetArrayLength(jids);
    jint* ids = env->GetIntArrayElements(jids, nullptr);
    jfloat* biases = env->GetFloatArrayElements(jbiases, nullptr);
    for (int i = 0; i < len; i++) {
        g_sampling.logit_bias[ids[i]] = biases[i];
    }
    env->ReleaseIntArrayElements(jids, ids, JNI_ABORT);
    env->ReleaseFloatArrayElements(jbiases, biases, JNI_ABORT);
}

JNI_FN(void, nativeSetHeadScales)(JNIEnv* env, jobject, jfloatArray jscales) {
    if (!g_ctx || !jscales) return;
    int len = env->GetArrayLength(jscales);
    jfloat* data = env->GetFloatArrayElements(jscales, nullptr);
    engine::InterventionConfig ic;
    ic.head_scales.assign(data, data + len);
    g_ctx->set_interventions(ic);
    env->ReleaseFloatArrayElements(jscales, data, JNI_ABORT);
}

JNI_FN(void, nativeResetHeadScales)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

JNI_FN(void, nativeSetAttentionTemperatureProfile)(JNIEnv* env, jobject, jfloatArray jtemps) {
    if (!g_ctx || !jtemps) return;
    int len = env->GetArrayLength(jtemps);
    jfloat* data = env->GetFloatArrayElements(jtemps, nullptr);
    engine::InterventionConfig ic;
    ic.attn_temperatures.assign(data, data + len);
    g_ctx->set_interventions(ic);
    env->ReleaseFloatArrayElements(jtemps, data, JNI_ABORT);
}

JNI_FN(void, nativeResetAttentionTemperatures)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

JNI_FN(void, nativeSetResidualGates)(JNIEnv* env, jobject, jfloatArray jattn, jfloatArray jffn) {
    if (!g_ctx) return;
    engine::InterventionConfig ic;
    if (jattn) {
        int len = env->GetArrayLength(jattn);
        jfloat* data = env->GetFloatArrayElements(jattn, nullptr);
        ic.residual_gates_attn.assign(data, data + len);
        env->ReleaseFloatArrayElements(jattn, data, JNI_ABORT);
    }
    if (jffn) {
        int len = env->GetArrayLength(jffn);
        jfloat* data = env->GetFloatArrayElements(jffn, nullptr);
        ic.residual_gates_ffn.assign(data, data + len);
        env->ReleaseFloatArrayElements(jffn, data, JNI_ABORT);
    }
    g_ctx->set_interventions(ic);
}

JNI_FN(void, nativeResetResidualGates)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

JNI_FN(void, nativeSetNormOffsets)(JNIEnv* env, jobject, jint layer, jfloatArray joffsets) {
    if (!g_ctx || !joffsets) return;
    int len = env->GetArrayLength(joffsets);
    jfloat* data = env->GetFloatArrayElements(joffsets, nullptr);
    engine::InterventionConfig ic;
    ic.norm_offsets.resize(layer + 1);
    ic.norm_offsets[layer].assign(data, data + len);
    g_ctx->set_interventions(ic);
    env->ReleaseFloatArrayElements(joffsets, data, JNI_ABORT);
}

JNI_FN(void, nativeResetNormOffsets)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

JNI_FN(void, nativeSetAttentionBias)(JNIEnv* env, jobject, jint layer, jfloatArray jbiases) {
    if (!g_ctx || !jbiases) return;
    int len = env->GetArrayLength(jbiases);
    jfloat* data = env->GetFloatArrayElements(jbiases, nullptr);
    engine::InterventionConfig ic;
    ic.attn_biases.resize(layer + 1);
    ic.attn_biases[layer].assign(data, data + len);
    g_ctx->set_interventions(ic);
    env->ReleaseFloatArrayElements(jbiases, data, JNI_ABORT);
}

JNI_FN(void, nativeClearAttentionBias)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

JNI_FN(void, nativeClearAllInterventions)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_interventions();
}

// ═══════════════════════════════════════════════════════════════════════════
// LORA
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jboolean, nativeApplyLora)(JNIEnv* env, jobject, jstring jpath, jfloat scale) {
    if (!g_ctx) return JNI_FALSE;
    std::string path = jni::to_string(env, jpath);
    return g_ctx->apply_lora(path, scale) ? JNI_TRUE : JNI_FALSE;
}

JNI_FN(void, nativeRemoveLora)(JNIEnv* env, jobject, jstring jpath) {
    if (!g_ctx) return;
    g_ctx->remove_lora(jni::to_string(env, jpath));
}

JNI_FN(void, nativeClearLora)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_lora();
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jboolean, nativeSaveState)(JNIEnv* env, jobject, jstring jpath) {
    if (!g_ctx) return JNI_FALSE;
    return g_ctx->save_state(jni::to_string(env, jpath)) ? JNI_TRUE : JNI_FALSE;
}

JNI_FN(jboolean, nativeLoadState)(JNIEnv* env, jobject, jstring jpath) {
    if (!g_ctx) return JNI_FALSE;
    return g_ctx->load_state(jni::to_string(env, jpath)) ? JNI_TRUE : JNI_FALSE;
}

// ═══════════════════════════════════════════════════════════════════════════
// CACHE
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(void, nativeClearCache)(JNIEnv*, jobject) {
    if (g_ctx) g_ctx->clear_cache();
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDINGS
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jboolean, nativeLoadEmbeddingModel)(
    JNIEnv* env, jobject, jstring jpath,
    jint nCtx, jint nThreads, jstring jBackendPath)
{
    std::lock_guard<std::mutex> lk(g_init_mtx);

    std::string path = jni::to_string(env, jpath);
    std::string backend_path = jni::to_string(env, jBackendPath);

    ensure_backends(backend_path);
    llama_backend_init();

    g_embed_ctx = std::make_unique<engine::Context>();

    engine::EngineConfig cfg;
    cfg.model_path = path;
    cfg.n_ctx      = nCtx > 0 ? nCtx : 512;
    cfg.n_batch    = cfg.n_ctx;
    cfg.n_threads  = nThreads;
    cfg.flash_attn = true;

    bool ok = g_embed_ctx->load(cfg);
    if (!ok) {
        LOGE("Failed to load embedding model: %s", path.c_str());
        g_embed_ctx.reset();
        return JNI_FALSE;
    }

    LOGI("Embedding model loaded: %s", path.c_str());
    return JNI_TRUE;
}

JNI_FN(jboolean, nativeLoadEmbeddingModelFromFd)(
    JNIEnv* env, jobject thiz,
    jint fd, jint nCtx, jint nThreads, jstring jBackendPath)
{
    int dup_fd = dup(fd);
    if (dup_fd < 0) return JNI_FALSE;

    char path_buf[64];
    snprintf(path_buf, sizeof(path_buf), "/proc/self/fd/%d", dup_fd);

    jstring jpath = env->NewStringUTF(path_buf);
    jboolean result = Java_com_mp_ai_1gguf_GGUFNativeLib_nativeLoadEmbeddingModel(
        env, thiz, jpath, nCtx, nThreads, jBackendPath);
    env->DeleteLocalRef(jpath);

    close(dup_fd);
    return result;
}

JNI_FN(jfloatArray, nativeEmbed)(JNIEnv* env, jobject, jstring jtext) {
    auto* ctx = g_embed_ctx ? g_embed_ctx.get() : g_ctx.get();
    if (!ctx || !ctx->is_loaded()) return nullptr;

    std::string text = jni::to_string(env, jtext);
    auto embd = ctx->embed(text);
    if (embd.empty()) return nullptr;

    jfloatArray result = env->NewFloatArray((jsize)embd.size());
    env->SetFloatArrayRegion(result, 0, (jsize)embd.size(), embd.data());
    return result;
}

JNI_FN(void, nativeReleaseEmbeddingModel)(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    g_embed_ctx.reset();
}

JNI_FN(jstring, nativeGetEmbeddingModelInfo)(JNIEnv* env, jobject) {
    if (!g_embed_ctx) return env->NewStringUTF("{}");
    return env->NewStringUTF(g_embed_ctx->model_info().c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// INFO
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jstring, nativeModelInfo)(JNIEnv* env, jobject) {
    if (!g_ctx) return env->NewStringUTF("{}");
    return env->NewStringUTF(g_ctx->model_info().c_str());
}

JNI_FN(jstring, nativeBackendInfo)(JNIEnv* env, jobject) {
    ensure_backends();
    if (!g_ctx) {
        std::ostringstream ss;
        ss << "{\"backends\":[";
        auto n = ggml_backend_dev_count();
        for (size_t i = 0; i < n; i++) {
            auto* dev = ggml_backend_dev_get(i);
            if (i > 0) ss << ",";
            ss << "{\"name\":\"" << ggml_backend_dev_name(dev)
               << "\",\"description\":\"" << ggml_backend_dev_description(dev)
               << "\",\"type\":" << (int)ggml_backend_dev_type(dev) << "}";
        }
        ss << "]}";
        return env->NewStringUTF(ss.str().c_str());
    }
    return env->NewStringUTF(g_ctx->backend_info().c_str());
}

JNI_FN(jint, nativeContextSize)(JNIEnv*, jobject) {
    return g_ctx ? g_ctx->n_ctx() : 0;
}

JNI_FN(jint, nativeVocabSize)(JNIEnv*, jobject) {
    return g_ctx ? g_ctx->n_vocab() : 0;
}

JNI_FN(jint, nativeLayerCount)(JNIEnv*, jobject) {
    return g_ctx ? g_ctx->n_layers() : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jstring, nativeBench)(JNIEnv* env, jobject, jint pp, jint tg, jint pl, jint nr) {
    if (!g_ctx) return env->NewStringUTF("{}");

    auto r = g_ctx->bench(pp, tg, pl, nr);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "{\"pp\":" << r.pp << ",\"tg\":" << r.tg
       << ",\"pp_avg\":" << r.pp_avg << ",\"tg_avg\":" << r.tg_avg << "}";
    return env->NewStringUTF(ss.str().c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// THERMAL
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jstring, nativeGetThermalState)(JNIEnv* env, jobject) {
    scan_thermal_zones();
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < g_thermal_zones.size(); i++) {
        auto& tz = g_thermal_zones[i];
        int t = read_temp(tz.path);
        float tc = t / 1000.0f;
        if (i > 0) ss << ",";
        ss << "{\"zone\":\"" << tz.type << "\",\"temp_c\":"
           << std::fixed << std::setprecision(1) << tc
           << ",\"throttled\":" << (tc >= 85.0f ? "true" : "false") << "}";
    }
    ss << "]";
    return env->NewStringUTF(ss.str().c_str());
}

JNI_FN(jint, nativeGetThermalLevel)(JNIEnv*, jobject) {
    scan_thermal_zones();
    int max_t = 0;
    for (auto& tz : g_thermal_zones) {
        int t = read_temp(tz.path);
        if (t > max_t) max_t = t;
    }
    float tc = max_t / 1000.0f;
    if (tc >= 95.0f) return 3;
    if (tc >= 85.0f) return 2;
    if (tc >= 70.0f) return 1;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKEND CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(void, nativeSetOpenCLCacheDir)(JNIEnv* env, jobject, jstring jpath) {
    std::string path = jni::to_string(env, jpath);
    mkdir(path.c_str(), 0755);
    setenv("GGML_OPENCL_CACHE_DIR", path.c_str(), 1);
    LOGI("OpenCL cache dir: %s", path.c_str());
}

JNI_FN(void, nativeSetGPUCacheDir)(JNIEnv* env, jobject, jstring jpath) {
    std::string base = jni::to_string(env, jpath);
    mkdir(base.c_str(), 0755);
    std::string cl_dir = base + "/opencl";
    mkdir(cl_dir.c_str(), 0755);
    setenv("GGML_OPENCL_CACHE_DIR", cl_dir.c_str(), 1);
    LOGI("GPU cache dirs set under: %s", base.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZATION
// ═══════════════════════════════════════════════════════════════════════════

JNI_FN(jintArray, nativeTokenize)(JNIEnv* env, jobject, jstring jtext, jboolean addSpecial) {
    if (!g_ctx) return nullptr;
    std::string text = jni::to_string(env, jtext);
    auto tokens = g_ctx->tokenize(text, addSpecial);
    if (tokens.empty()) return nullptr;
    jintArray result = env->NewIntArray((jsize)tokens.size());
    env->SetIntArrayRegion(result, 0, (jsize)tokens.size(), tokens.data());
    return result;
}

JNI_FN(jstring, nativeDetokenize)(JNIEnv* env, jobject, jintArray jtokens) {
    if (!g_ctx || !jtokens) return env->NewStringUTF("");
    int len = env->GetArrayLength(jtokens);
    jint* data = env->GetIntArrayElements(jtokens, nullptr);
    std::vector<int32_t> tokens(data, data + len);
    env->ReleaseIntArrayElements(jtokens, data, JNI_ABORT);
    std::string text = g_ctx->detokenize(tokens);
    return jni::to_jstring(env, text);
}
