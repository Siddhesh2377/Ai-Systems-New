#include <jni.h>
#include <android/log.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "engine.h"
#include "log.h"

namespace {

void android_log_sink(rmg::LogLevel level, const char* msg) {
    int prio;
    switch (level) {
        case rmg::LOG_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        case rmg::LOG_INFO:  prio = ANDROID_LOG_INFO;  break;
        case rmg::LOG_WARN:  prio = ANDROID_LOG_WARN;  break;
        case rmg::LOG_ERROR: prio = ANDROID_LOG_ERROR; break;
        default:             prio = ANDROID_LOG_INFO;  break;
    }
    __android_log_print(prio, "rmg", "%s", msg);
}

inline rmg::Engine* fromHandle(jlong h) {
    return reinterpret_cast<rmg::Engine*>(static_cast<uintptr_t>(h));
}

inline jlong toHandle(rmg::Engine* e) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(e));
}

struct StreamCtx {
    JNIEnv*      env;
    jobject      callback;
    jmethodID    on_token_mid;
    rmg::Engine* engine;
    bool         aborted;
};

// Invoked by engine_generate_stream on the calling (JNI) thread.
// Returns 0 to continue, non-zero to abort.
int kotlin_token_callback(int token, void* ud) {
    auto* ctx = static_cast<StreamCtx*>(ud);
    if (ctx->aborted) return 1;

    size_t len = 0;
    const char* p = rmg::engine_token_bytes(*ctx->engine, token, &len);

    jbyteArray bytes = nullptr;
    if (p && len > 0) {
        bytes = ctx->env->NewByteArray(static_cast<jsize>(len));
        if (bytes) {
            ctx->env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(len),
                                         reinterpret_cast<const jbyte*>(p));
        }
    }

    const jboolean cont = ctx->env->CallBooleanMethod(
            ctx->callback, ctx->on_token_mid, static_cast<jint>(token), bytes);

    if (bytes) ctx->env->DeleteLocalRef(bytes);

    if (ctx->env->ExceptionCheck()) {
        ctx->env->ExceptionDescribe();
        ctx->env->ExceptionClear();
        ctx->aborted = true;
        return 1;
    }
    return cont == JNI_TRUE ? 0 : 1;
}

}

extern "C" {

JNIEXPORT void JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeInit(JNIEnv*, jclass) {
    rmg::log_set_sink(android_log_sink);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeSetLogLevel(JNIEnv*, jclass, jint level) {
    if (level < rmg::LOG_DEBUG || level > rmg::LOG_ERROR) return;
    rmg::log_set_level(static_cast<rmg::LogLevel>(level));
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeOpen(JNIEnv* env, jclass, jstring jpath) {
    if (!jpath) return 0;
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return 0;

    auto* e = new (std::nothrow) rmg::Engine();
    if (!e) {
        env->ReleaseStringUTFChars(jpath, path);
        return 0;
    }

    const int rc = rmg::engine_open(*e, path);
    env->ReleaseStringUTFChars(jpath, path);

    if (rc < 0) {
        rmg::engine_close(*e);
        delete e;
        return 0;
    }
    return toHandle(e);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeClose(JNIEnv*, jclass, jlong handle) {
    auto* e = fromHandle(handle);
    if (!e) return;
    rmg::engine_close(*e);
    delete e;
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeReset(JNIEnv*, jclass, jlong handle) {
    auto* e = fromHandle(handle);
    if (!e) return;
    rmg::engine_reset(*e);
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeSeqPos(JNIEnv*, jclass, jlong handle) {
    auto* e = fromHandle(handle);
    return e ? e->seq_pos : 0;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeForward(JNIEnv* env, jclass, jlong handle,
                                                  jint token_id, jfloatArray logits_out) {
    auto* e = fromHandle(handle);
    if (!e || !logits_out) return JNI_FALSE;

    const jsize n = env->GetArrayLength(logits_out);
    if (n < e->dims.vocab_size) return JNI_FALSE;

    jfloat* buf = env->GetFloatArrayElements(logits_out, nullptr);
    if (!buf) return JNI_FALSE;

    rmg::engine_forward(*e, token_id, buf);
    env->ReleaseFloatArrayElements(logits_out, buf, 0);
    return JNI_TRUE;
}

JNIEXPORT jintArray JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeGenerate(JNIEnv* env, jclass, jlong handle,
                                                   jintArray prompt_ids,
                                                   jint max_new, jint stop_id) {
    auto* e = fromHandle(handle);
    if (!e || !prompt_ids || max_new < 0) return nullptr;

    const jsize n_prompt = env->GetArrayLength(prompt_ids);
    if (n_prompt <= 0) return nullptr;

    jint* prompt = env->GetIntArrayElements(prompt_ids, nullptr);
    if (!prompt) return nullptr;

    std::vector<jint> out(static_cast<size_t>(max_new));
    int n_written = 0;
    if (max_new > 0) {
        n_written = rmg::engine_generate(*e, prompt, n_prompt,
                                         max_new, stop_id, out.data());
    }
    env->ReleaseIntArrayElements(prompt_ids, prompt, JNI_ABORT);

    if (n_written < 0) n_written = 0;
    jintArray result = env->NewIntArray(n_written);
    if (!result) return nullptr;
    if (n_written > 0) {
        env->SetIntArrayRegion(result, 0, n_written, out.data());
    }
    return result;
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeGenerateStream(JNIEnv* env, jclass, jlong handle,
                                                         jintArray prompt_ids,
                                                         jint max_new, jint stop_id,
                                                         jobject callback) {
    auto* e = fromHandle(handle);
    if (!e || !prompt_ids || max_new < 0 || !callback) return -1;

    const jsize n_prompt = env->GetArrayLength(prompt_ids);
    if (n_prompt <= 0) return -1;

    jclass cb_cls = env->GetObjectClass(callback);
    if (!cb_cls) return -1;
    const jmethodID on_token_mid = env->GetMethodID(cb_cls, "onToken", "(I[B)Z");
    env->DeleteLocalRef(cb_cls);
    if (!on_token_mid) return -1;

    jint* prompt = env->GetIntArrayElements(prompt_ids, nullptr);
    if (!prompt) return -1;

    StreamCtx ctx{env, callback, on_token_mid, e, false};
    const int n = rmg::engine_generate_stream(*e, prompt, n_prompt,
                                              max_new, stop_id,
                                              kotlin_token_callback, &ctx);
    env->ReleaseIntArrayElements(prompt_ids, prompt, JNI_ABORT);
    return n;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeHasTokenizer(JNIEnv*, jclass, jlong handle) {
    auto* e = fromHandle(handle);
    return (e && e->tokenizer.vocab_size > 0 && e->tokenizer.offsets != nullptr)
           ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeTokenBytes(JNIEnv* env, jclass, jlong handle, jint token_id) {
    auto* e = fromHandle(handle);
    if (!e) return nullptr;

    size_t len = 0;
    const char* p = rmg::engine_token_bytes(*e, token_id, &len);
    if (!p) return nullptr;

    jbyteArray result = env->NewByteArray(static_cast<jsize>(len));
    if (!result) return nullptr;
    if (len > 0) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(len),
                                reinterpret_cast<const jbyte*>(p));
    }
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeDecodeTokens(JNIEnv* env, jclass, jlong handle,
                                                      jintArray token_ids) {
    auto* e = fromHandle(handle);
    if (!e || !token_ids) return nullptr;
    if (e->tokenizer.vocab_size <= 0 || e->tokenizer.offsets == nullptr) return nullptr;

    const jsize n = env->GetArrayLength(token_ids);
    if (n <= 0) return env->NewByteArray(0);

    jint* ids = env->GetIntArrayElements(token_ids, nullptr);
    if (!ids) return nullptr;

    const size_t need = rmg::engine_decode_tokens(*e, ids, n, nullptr, 0);
    std::vector<char> buf(need);
    if (need > 0) {
        rmg::engine_decode_tokens(*e, ids, n, buf.data(), buf.size());
    }
    env->ReleaseIntArrayElements(token_ids, ids, JNI_ABORT);

    jbyteArray result = env->NewByteArray(static_cast<jsize>(need));
    if (!result) return nullptr;
    if (need > 0) {
        env->SetByteArrayRegion(result, 0, static_cast<jsize>(need),
                                reinterpret_cast<const jbyte*>(buf.data()));
    }
    return result;
}

// Layout shared with RmgEngine.kt — keep in sync:
//   [0..7]  d_model, n_layers, n_heads, n_kv_heads, d_head, d_ff, vocab_size, max_seq
//   [8..9]  rope_theta, rms_eps  (Float.fromBits)
//   [10]    rope_interleaved
//   [11]    tie_word_embeddings
JNIEXPORT jintArray JNICALL
Java_com_dark_ai_1rmg_RmgNativeLib_nativeGetDims(JNIEnv* env, jclass, jlong handle) {
    auto* e = fromHandle(handle);
    if (!e) return nullptr;
    const auto& d = e->dims;

    jint vals[12];
    vals[0]  = d.d_model;
    vals[1]  = d.n_layers;
    vals[2]  = d.n_heads;
    vals[3]  = d.n_kv_heads;
    vals[4]  = d.d_head;
    vals[5]  = d.d_ff;
    vals[6]  = d.vocab_size;
    vals[7]  = d.max_seq;

    jint rope_bits, eps_bits;
    std::memcpy(&rope_bits, &d.rope_theta, sizeof(jint));
    std::memcpy(&eps_bits,  &d.rms_eps,    sizeof(jint));
    vals[8]  = rope_bits;
    vals[9]  = eps_bits;
    vals[10] = d.rope_interleaved    ? 1 : 0;
    vals[11] = d.tie_word_embeddings ? 1 : 0;

    jintArray r = env->NewIntArray(12);
    if (!r) return nullptr;
    env->SetIntArrayRegion(r, 0, 12, vals);
    return r;
}

}
