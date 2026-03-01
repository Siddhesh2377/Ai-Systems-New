#include <jni.h>
#include <string>
#include <android/log.h>

#include "ggml-engine.h"
#include "tool-manager.h"
#include "character-engine.h"
#include "kv-cache-manager.h"

#define LOG_TAG "ToolNeuron"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Store global references
static JavaVM * g_jvm = nullptr;

JNIEXPORT jint JNI_OnLoad(JavaVM * vm, void *) {
    g_jvm = vm;
    LOGI("Tool-Neuron JNI loaded");
    return JNI_VERSION_1_6;
}

// ==================== GGMLEngine ====================

extern "C" JNIEXPORT jlong JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeCreate(
        JNIEnv * env, jobject,
        jint nCtx, jint nBatch, jint nThreads, jboolean useMmap, jboolean flashAttn) {
    auto params = ggml_engine_default_params();
    params.n_ctx = nCtx;
    params.n_batch = nBatch;
    params.n_threads = nThreads;
    params.use_mmap = useMmap;
    params.flash_attn = flashAttn;

    auto * engine = ggml_engine_create(params);
    LOGI("Engine created: ctx=%d batch=%d threads=%d", nCtx, nBatch, nThreads);
    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeDestroy(JNIEnv *, jobject, jlong handle) {
    ggml_engine_free(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeLoadModel(JNIEnv * env, jobject, jlong handle, jstring path) {
    auto * engine = reinterpret_cast<ggml_engine_t *>(handle);
    const char * cpath = env->GetStringUTFChars(path, nullptr);
    auto status = ggml_engine_load_model(engine, cpath);
    LOGI("Model load: path=%s status=%d", cpath, status);
    env->ReleaseStringUTFChars(path, cpath);
    return static_cast<jint>(status);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeLoadModelFromFd(JNIEnv *, jobject, jlong handle, jint fd) {
    auto * engine = reinterpret_cast<ggml_engine_t *>(handle);
    auto status = ggml_engine_load_model_from_fd(engine, fd);
    LOGI("Model load from fd=%d status=%d", fd, status);
    return static_cast<jint>(status);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeUnloadModel(JNIEnv *, jobject, jlong handle) {
    ggml_engine_unload_model(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeIsLoaded(JNIEnv *, jobject, jlong handle) {
    return ggml_engine_is_loaded(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeGetModelInfo(JNIEnv * env, jobject, jlong handle) {
    auto * engine = reinterpret_cast<ggml_engine_t *>(handle);
    char * json = ggml_engine_model_info_json(engine);
    jstring result = env->NewStringUTF(json);
    ggml_engine_free_string(json);
    return result;
}

// Streaming callback data
struct jni_callback_data {
    JNIEnv * env;
    jobject  callback;
    jmethodID method;
    bool     should_stop;
};

static bool jni_token_callback(const char * text, void * user_data) {
    auto * data = static_cast<jni_callback_data *>(user_data);
    if (data->should_stop) return false;

    JNIEnv * env = data->env;
    jstring jtext = env->NewStringUTF(text);
    jboolean cont = env->CallBooleanMethod(data->callback, data->method, jtext);
    env->DeleteLocalRef(jtext);

    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        data->should_stop = true;
        return false;
    }

    return cont;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeGenerate(
        JNIEnv * env, jobject, jlong handle,
        jstring prompt, jfloat temperature, jint topK, jfloat topP,
        jfloat minP, jfloat repeatPenalty, jint repeatLastN,
        jint nPredict, jint seed, jobject callback) {

    auto * engine = reinterpret_cast<ggml_engine_t *>(handle);
    const char * cprompt = env->GetStringUTFChars(prompt, nullptr);

    auto sampling = ggml_engine_default_sampling();
    sampling.temperature = temperature;
    sampling.top_k = topK;
    sampling.top_p = topP;
    sampling.min_p = minP;
    sampling.repeat_penalty = repeatPenalty;
    sampling.repeat_last_n = repeatLastN;
    sampling.n_predict = nPredict;
    sampling.seed = static_cast<uint32_t>(seed);

    ggml_engine_token_callback cb = nullptr;
    jni_callback_data cb_data = {};

    if (callback != nullptr) {
        jclass cls = env->GetObjectClass(callback);
        jmethodID method = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)Z");
        if (method) {
            cb_data.env = env;
            cb_data.callback = callback;
            cb_data.method = method;
            cb_data.should_stop = false;
            cb = jni_token_callback;
        }
    }

    auto status = ggml_engine_generate(engine, cprompt, sampling,
                                        cb, cb ? &cb_data : nullptr);

    env->ReleaseStringUTFChars(prompt, cprompt);
    return static_cast<jint>(status);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeCancel(JNIEnv *, jobject, jlong handle) {
    ggml_engine_cancel(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeGetResponse(JNIEnv * env, jobject, jlong handle) {
    auto * engine = reinterpret_cast<ggml_engine_t *>(handle);
    char * resp = ggml_engine_get_response(engine);
    jstring result = env->NewStringUTF(resp);
    ggml_engine_free_string(resp);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeClearContext(JNIEnv *, jobject, jlong handle) {
    ggml_engine_clear_context(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeContextUsed(JNIEnv *, jobject, jlong handle) {
    return ggml_engine_context_used(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeContextSize(JNIEnv *, jobject, jlong handle) {
    return ggml_engine_context_size(reinterpret_cast<ggml_engine_t *>(handle));
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_dark_gguf_1lib_GGMLEngine_nativeGetPerf(JNIEnv * env, jobject, jlong handle) {
    auto perf = ggml_engine_get_perf(reinterpret_cast<ggml_engine_t *>(handle));
    jfloatArray result = env->NewFloatArray(6);
    float vals[6] = {
        (float)perf.prompt_eval_ms,
        (float)perf.generation_ms,
        (float)perf.prompt_tokens,
        (float)perf.generated_tokens,
        (float)perf.prompt_tokens_per_sec,
        (float)perf.generation_tokens_per_sec
    };
    env->SetFloatArrayRegion(result, 0, 6, vals);
    return result;
}

// ==================== ToolManager ====================

extern "C" JNIEXPORT jlong JNICALL
Java_com_dark_gguf_1lib_ToolManager_nativeCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(tool_manager_create());
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_ToolManager_nativeDestroy(JNIEnv *, jobject, jlong handle) {
    tool_manager_free(reinterpret_cast<tool_manager_t *>(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_ToolManager_nativeRegisterTool(
        JNIEnv * env, jobject, jlong handle,
        jstring name, jstring description, jobjectArray paramNames,
        jobjectArray paramDescs, jintArray paramTypes, jbooleanArray paramRequired) {

    auto * tm = reinterpret_cast<tool_manager_t *>(handle);

    const char * cname = env->GetStringUTFChars(name, nullptr);
    const char * cdesc = env->GetStringUTFChars(description, nullptr);

    int n_params = paramNames ? env->GetArrayLength(paramNames) : 0;
    std::vector<tool_param_def> params(n_params);
    std::vector<std::string> names_store(n_params);
    std::vector<std::string> descs_store(n_params);

    jint * types = n_params > 0 ? env->GetIntArrayElements(paramTypes, nullptr) : nullptr;
    jboolean * required = n_params > 0 ? env->GetBooleanArrayElements(paramRequired, nullptr) : nullptr;

    for (int i = 0; i < n_params; i++) {
        auto jn = (jstring)env->GetObjectArrayElement(paramNames, i);
        auto jd = (jstring)env->GetObjectArrayElement(paramDescs, i);
        names_store[i] = env->GetStringUTFChars(jn, nullptr);
        descs_store[i] = env->GetStringUTFChars(jd, nullptr);
        params[i].name = names_store[i].c_str();
        params[i].description = descs_store[i].c_str();
        params[i].type = static_cast<tool_param_type>(types[i]);
        params[i].required = required[i];
        env->ReleaseStringUTFChars(jn, names_store[i].c_str());
        env->ReleaseStringUTFChars(jd, descs_store[i].c_str());
    }

    tool_def def = {cname, cdesc, params.data(), n_params};
    tool_manager_register(tm, &def);

    if (types) env->ReleaseIntArrayElements(paramTypes, types, 0);
    if (required) env->ReleaseBooleanArrayElements(paramRequired, required, 0);
    env->ReleaseStringUTFChars(name, cname);
    env->ReleaseStringUTFChars(description, cdesc);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_ToolManager_nativeGetPrompt(JNIEnv * env, jobject, jlong handle) {
    char * prompt = tool_manager_get_prompt(reinterpret_cast<tool_manager_t *>(handle));
    jstring result = env->NewStringUTF(prompt);
    tool_manager_free_string(prompt);
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_ToolManager_nativeParseOutput(JNIEnv * env, jobject, jlong handle, jstring output) {
    auto * tm = reinterpret_cast<tool_manager_t *>(handle);
    const char * cout = env->GetStringUTFChars(output, nullptr);
    auto result = tool_manager_parse_output(tm, cout);
    env->ReleaseStringUTFChars(output, cout);

    if (!result.is_valid) {
        return nullptr;
    }

    // Return as JSON: {"tool": "name", "arguments": {...}}
    std::string json = "{\"tool\": \"" + std::string(result.tool_name) +
                       "\", \"arguments\": " + std::string(result.arguments_json) + "}";
    free((void *)result.tool_name);
    free((void *)result.arguments_json);
    return env->NewStringUTF(json.c_str());
}

// ==================== CharacterEngine ====================

extern "C" JNIEXPORT jlong JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(character_engine_create());
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeDestroy(JNIEnv *, jobject, jlong handle) {
    character_engine_free(reinterpret_cast<character_engine_t *>(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeSetPersonality(
        JNIEnv * env, jobject, jlong handle,
        jstring name, jstring persona,
        jfloat temperature, jfloat topP, jfloat repPenalty,
        jfloat creativity, jfloat verbosity, jfloat formality) {

    auto * ce = reinterpret_cast<character_engine_t *>(handle);
    const char * cname = env->GetStringUTFChars(name, nullptr);
    const char * cpersona = env->GetStringUTFChars(persona, nullptr);

    char_personality p = {};
    p.name = cname;
    p.persona = cpersona;
    p.temperature = temperature;
    p.top_p = topP;
    p.repetition_penalty = repPenalty;
    p.creativity = creativity;
    p.verbosity = verbosity;
    p.formality = formality;

    character_engine_set_personality(ce, &p);

    env->ReleaseStringUTFChars(name, cname);
    env->ReleaseStringUTFChars(persona, cpersona);
}

extern "C" JNIEXPORT void JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeSetMood(JNIEnv *, jobject, jlong handle, jint mood) {
    character_engine_set_mood(reinterpret_cast<character_engine_t *>(handle),
                              static_cast<char_mood>(mood));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeGetContext(JNIEnv * env, jobject, jlong handle) {
    char * ctx = character_engine_get_context(reinterpret_cast<character_engine_t *>(handle));
    jstring result = env->NewStringUTF(ctx);
    character_engine_free_string(ctx);
    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_dark_gguf_1lib_CharacterEngine_nativeGetParams(JNIEnv * env, jobject, jlong handle) {
    auto params = character_engine_get_params(reinterpret_cast<character_engine_t *>(handle));
    jfloatArray result = env->NewFloatArray(5);
    float vals[5] = {params.temperature, params.top_p, params.min_p,
                     params.repetition_penalty, (float)params.top_k};
    env->SetFloatArrayRegion(result, 0, 5, vals);
    return result;
}
