/**
 * JNI callback utilities for Stable Diffusion pipeline.
 *
 * Caches the SDCallback class + four method IDs + a reusable preview
 * jbyteArray as process-global state. Global refs are cross-thread; the
 * cache is reused across every generation and every worker thread that
 * delivers callbacks, with a mutex around mutation so reset and per-class
 * invalidation are safe.
 */

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "jni_utils.h"
#include "sd_logger.h"

#include <jni.h>
#include <atomic>
#include <mutex>
#include <string>

namespace sd_jni {

namespace {

struct CallbackCache {
    jclass cls = nullptr;
    jmethodID onProgress = nullptr;       // (II)V
    jmethodID onImageProgress = nullptr;  // (II[BII)V
    jmethodID onComplete = nullptr;       // ([BIIJI)V
    jmethodID onError = nullptr;          // (Ljava/lang/String;)V
    bool initialized = false;

    jbyteArray cachedPreviewArray = nullptr;
    int cachedPreviewSize = 0;
};

CallbackCache g_cache;
std::mutex g_cache_mtx;
std::atomic<bool> g_cache_reset_requested{false};

void release_locked(JNIEnv* env) {
    if (g_cache.cls) {
        env->DeleteGlobalRef(g_cache.cls);
        g_cache.cls = nullptr;
    }
    if (g_cache.cachedPreviewArray) {
        env->DeleteGlobalRef(g_cache.cachedPreviewArray);
        g_cache.cachedPreviewArray = nullptr;
        g_cache.cachedPreviewSize = 0;
    }
    g_cache.onProgress = nullptr;
    g_cache.onImageProgress = nullptr;
    g_cache.onComplete = nullptr;
    g_cache.onError = nullptr;
    g_cache.initialized = false;
}

// Look up method IDs for `callback`'s actual class. Different SDCallback
// subclasses produce different jclass objects with different method-ID
// slots; reusing one set against another is undefined behavior and was
// the source of the 1024² SIGSEGV in RenderThread.
void ensure_init_locked(JNIEnv* env, jobject callback) {
    jclass curCls = env->GetObjectClass(callback);
    if (!curCls) {
        SD_LOG_ERROR("sd_jni: GetObjectClass returned null");
        if (env->ExceptionCheck()) env->ExceptionClear();
        return;
    }

    if (g_cache.initialized && g_cache.cls &&
        env->IsSameObject(g_cache.cls, curCls)) {
        env->DeleteLocalRef(curCls);
        return;
    }

    if (g_cache.initialized) {
        release_locked(env);
    }

    g_cache.cls = static_cast<jclass>(env->NewGlobalRef(curCls));
    env->DeleteLocalRef(curCls);
    if (!g_cache.cls) {
        SD_LOG_ERROR("sd_jni: NewGlobalRef(callback class) failed");
        if (env->ExceptionCheck()) env->ExceptionClear();
        return;
    }

    g_cache.onProgress      = env->GetMethodID(g_cache.cls, "onProgress",      "(II)V");
    g_cache.onImageProgress = env->GetMethodID(g_cache.cls, "onImageProgress", "(II[BII)V");
    g_cache.onComplete      = env->GetMethodID(g_cache.cls, "onComplete",      "([BIIJI)V");
    g_cache.onError         = env->GetMethodID(g_cache.cls, "onError",         "(Ljava/lang/String;)V");

    if (!g_cache.onProgress || !g_cache.onImageProgress ||
        !g_cache.onComplete || !g_cache.onError) {
        SD_LOG_ERROR("sd_jni: missing one or more SDCallback methods on class");
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteGlobalRef(g_cache.cls);
        g_cache.cls = nullptr;
        return;
    }

    g_cache.initialized = true;
}

// Any pending Java exception left after a CallVoidMethod will abort the
// next JNI call with a hard process abort. Clear once per callback so the
// next step survives even if user code threw. Emit a structured tn_security
// error first so callers can surface "Kotlin callback X threw" instead of
// swallowing it silently.
void check_clear_exception(JNIEnv* env, const char* where) {
    if (env->ExceptionCheck()) {
        TN_ERR(TN_CODE_PLUGIN_EXEC_FAIL, TN_STAGE_UNSPECIFIED,
               "Kotlin callback threw in %s", where ? where : "<unknown>");
        SD_LOG_ERROR("sd_jni: pending Java exception after %s — clearing", where);
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
}

void honor_reset_request_locked(JNIEnv* env) {
    if (g_cache_reset_requested.exchange(false, std::memory_order_acq_rel)) {
        release_locked(env);
    }
}

} // anonymous namespace

void on_progress(JNIEnv* env, jobject cb, int step, int totalSteps) {
    if (!cb) return;
    std::lock_guard<std::mutex> lock(g_cache_mtx);

    honor_reset_request_locked(env);
    ensure_init_locked(env, cb);
    if (!g_cache.onProgress) return;

    env->CallVoidMethod(cb, g_cache.onProgress, (jint)step, (jint)totalSteps);
    check_clear_exception(env, "onProgress");
}

void on_image_progress(JNIEnv* env, jobject cb, int step, int totalSteps,
                       const uint8_t* rgbData, int dataLen, int width, int height) {
    if (!cb) return;
    std::lock_guard<std::mutex> lock(g_cache_mtx);

    honor_reset_request_locked(env);
    ensure_init_locked(env, cb);
    if (!g_cache.onImageProgress) return;

    if (!g_cache.cachedPreviewArray || g_cache.cachedPreviewSize != dataLen) {
        if (g_cache.cachedPreviewArray) {
            env->DeleteGlobalRef(g_cache.cachedPreviewArray);
            g_cache.cachedPreviewArray = nullptr;
        }
        jbyteArray local = env->NewByteArray(dataLen);
        if (!local) {
            SD_LOG_ERROR("on_image_progress: NewByteArray(%d) failed — likely OOM", dataLen);
            if (env->ExceptionCheck()) env->ExceptionClear();
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
    check_clear_exception(env, "onImageProgress");
}

void on_complete(JNIEnv* env, jobject cb, const uint8_t* rgbData, int dataLen,
                 int width, int height, long seed, int generationTimeMs) {
    if (!cb) return;
    std::lock_guard<std::mutex> lock(g_cache_mtx);

    honor_reset_request_locked(env);
    ensure_init_locked(env, cb);
    if (!g_cache.onComplete) return;

    jbyteArray jdata = env->NewByteArray(dataLen);
    if (!jdata) {
        SD_LOG_ERROR("on_complete: NewByteArray(%d) failed — likely OOM", dataLen);
        if (env->ExceptionCheck()) env->ExceptionClear();
        return;
    }
    env->SetByteArrayRegion(jdata, 0, dataLen, reinterpret_cast<const jbyte*>(rgbData));

    env->CallVoidMethod(cb, g_cache.onComplete,
                        jdata, (jint)width, (jint)height,
                        (jlong)seed, (jint)generationTimeMs);
    check_clear_exception(env, "onComplete");
    env->DeleteLocalRef(jdata);
}

void on_error(JNIEnv* env, jobject cb, const char* msg) {
    if (!cb) return;
    std::lock_guard<std::mutex> lock(g_cache_mtx);

    honor_reset_request_locked(env);
    ensure_init_locked(env, cb);
    if (!g_cache.onError) return;

    jstring jmsg = env->NewStringUTF(msg ? msg : "<unknown error>");
    if (!jmsg) {
        SD_LOG_ERROR("on_error: NewStringUTF failed");
        if (env->ExceptionCheck()) env->ExceptionClear();
        return;
    }
    env->CallVoidMethod(cb, g_cache.onError, jmsg);
    check_clear_exception(env, "onError");
    env->DeleteLocalRef(jmsg);
}

void reset_cache() {
    g_cache_reset_requested.store(true, std::memory_order_release);
}

} // namespace sd_jni
