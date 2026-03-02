/**
 * JNI callback utilities for Stable Diffusion pipeline
 *
 * Thread-local caching of callback method IDs for minimal JNI overhead.
 * Pattern proven in production by ai_gguf module.
 */

#include "jni_utils.h"
#include "sd_logger.h"

#include <jni.h>
#include <string>
#include <atomic>

namespace sd_jni {

namespace {

struct CallbackCache {
    jclass cls = nullptr;
    jmethodID onProgress = nullptr;       // (II)V
    jmethodID onImageProgress = nullptr;  // (II[BII)V
    jmethodID onComplete = nullptr;       // ([BIIJI)V
    jmethodID onError = nullptr;          // (Ljava/lang/String;)V
    bool initialized = false;

    // Perf 13: Cached jbyteArray for preview steps (avoids 786KB alloc per step)
    jbyteArray cachedPreviewArray = nullptr;
    int cachedPreviewSize = 0;

    void init(JNIEnv* env, jobject callback) {
        if (initialized) return;

        jclass tempCls = env->GetObjectClass(callback);
        if (!tempCls) {
            SD_LOG_ERROR("sd_jni: unable to find callback class");
            return;
        }

        cls = static_cast<jclass>(env->NewGlobalRef(tempCls));
        env->DeleteLocalRef(tempCls);

        onProgress = env->GetMethodID(cls, "onProgress", "(II)V");
        onImageProgress = env->GetMethodID(cls, "onImageProgress", "(II[BII)V");
        onComplete = env->GetMethodID(cls, "onComplete", "([BIIJI)V");
        onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");

        if (!onProgress || !onImageProgress || !onComplete || !onError) {
            SD_LOG_ERROR("sd_jni: failed to find one or more callback methods");
        }

        initialized = true;
    }

    void release(JNIEnv* env) {
        if (cls) {
            env->DeleteGlobalRef(cls);
            cls = nullptr;
        }
        if (cachedPreviewArray) {
            env->DeleteGlobalRef(cachedPreviewArray);
            cachedPreviewArray = nullptr;
            cachedPreviewSize = 0;
        }
        onProgress = nullptr;
        onImageProgress = nullptr;
        onComplete = nullptr;
        onError = nullptr;
        initialized = false;
    }
};

static thread_local CallbackCache g_cache;
static std::atomic<bool> g_cache_reset_requested{false};

} // anonymous namespace

void on_progress(JNIEnv* env, jobject cb, int step, int totalSteps) {
    if (!cb) return;

    if (g_cache_reset_requested.exchange(false)) {
        g_cache.release(env);
    }

    g_cache.init(env, cb);
    if (!g_cache.onProgress) return;

    env->CallVoidMethod(cb, g_cache.onProgress, (jint)step, (jint)totalSteps);
}

void on_image_progress(JNIEnv* env, jobject cb, int step, int totalSteps,
                       const uint8_t* rgbData, int dataLen, int width, int height) {
    if (!cb) return;

    g_cache.init(env, cb);
    if (!g_cache.onImageProgress) return;

    // Perf 13: Reuse cached jbyteArray across steps (same size within a generation)
    if (!g_cache.cachedPreviewArray || g_cache.cachedPreviewSize != dataLen) {
        if (g_cache.cachedPreviewArray) {
            env->DeleteGlobalRef(g_cache.cachedPreviewArray);
            g_cache.cachedPreviewArray = nullptr;
        }
        jbyteArray local = env->NewByteArray(dataLen);
        if (!local) {
            SD_LOG_ERROR("on_image_progress: NewByteArray(%d) failed — likely OOM", dataLen);
            on_error(env, cb, "Out of memory allocating image progress buffer");
            return;
        }
        g_cache.cachedPreviewArray = static_cast<jbyteArray>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        g_cache.cachedPreviewSize = dataLen;
    }

    env->SetByteArrayRegion(g_cache.cachedPreviewArray, 0, dataLen,
                            reinterpret_cast<const jbyte*>(rgbData));

    env->CallVoidMethod(cb, g_cache.onImageProgress,
                        (jint)step, (jint)totalSteps, g_cache.cachedPreviewArray,
                        (jint)width, (jint)height);
}

void on_complete(JNIEnv* env, jobject cb, const uint8_t* rgbData, int dataLen,
                 int width, int height, long seed, int generationTimeMs) {
    if (!cb) return;

    g_cache.init(env, cb);
    if (!g_cache.onComplete) return;

    jbyteArray jdata = env->NewByteArray(dataLen);
    if (!jdata) {
        SD_LOG_ERROR("on_complete: NewByteArray(%d) failed — likely OOM", dataLen);
        on_error(env, cb, "Out of memory allocating completion image buffer");
        return;
    }
    env->SetByteArrayRegion(jdata, 0, dataLen, reinterpret_cast<const jbyte*>(rgbData));

    env->CallVoidMethod(cb, g_cache.onComplete,
                        jdata, (jint)width, (jint)height, (jlong)seed, (jint)generationTimeMs);
    env->DeleteLocalRef(jdata);
}

void on_error(JNIEnv* env, jobject cb, const char* msg) {
    if (!cb) return;

    g_cache.init(env, cb);
    if (!g_cache.onError) return;

    jstring jmsg = env->NewStringUTF(msg ? msg : "<unknown error>");
    env->CallVoidMethod(cb, g_cache.onError, jmsg);
    env->DeleteLocalRef(jmsg);
}

void reset_cache() {
    g_cache_reset_requested.store(true, std::memory_order_relaxed);
}

} // namespace sd_jni
