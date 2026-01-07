/**
 * Optimized JNI utility functions for token streaming
 * 
 * Optimizations:
 * 1. Thread-local caching of JNI method IDs
 * 2. Global reference for callback class (prevents repeated lookups)
 * 3. Fast path for ASCII tokens
 * 4. Immediate delivery - no buffering
 */

#include "jni_utils.h"
#include "utf8_utils.h"
#include "logger.h"

#include <jni.h>
#include <string>
#include <atomic>

namespace jni {

// ============================================================================
// JNI CALLBACK CACHE
// Thread-local for multi-threaded safety
// ============================================================================

    namespace {

        struct CallbackCache {
            jclass cls = nullptr;
            jmethodID onToken = nullptr;
            jmethodID onError = nullptr;
            jmethodID onToolCall = nullptr;
            jmethodID onDone = nullptr;
            bool initialized = false;

            void init(JNIEnv* env, jobject callback) {
                if (initialized) return;

                jclass tempCls = env->GetObjectClass(callback);
                if (!tempCls) {
                    LOG_ERROR("jni_utils: unable to find callback class");
                    return;
                }

                // Create global reference to class (survives across JNI calls)
                cls = static_cast<jclass>(env->NewGlobalRef(tempCls));
                env->DeleteLocalRef(tempCls);

                // Cache method IDs (these don't change for a class)
                onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
                onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
                onToolCall = env->GetMethodID(cls, "onToolCall", "(Ljava/lang/String;Ljava/lang/String;)V");
                onDone = env->GetMethodID(cls, "onDone", "()V");

                if (!onToken || !onError || !onToolCall || !onDone) {
                    LOG_ERROR("jni_utils: failed to find callback methods");
                }

                initialized = true;
            }

            void release(JNIEnv* env) {
                if (cls) {
                    env->DeleteGlobalRef(cls);
                    cls = nullptr;
                }
                onToken = nullptr;
                onError = nullptr;
                onToolCall = nullptr;
                onDone = nullptr;
                initialized = false;
            }
        };

// Thread-local cache for multi-threaded environments
        static thread_local CallbackCache g_cache;

// Global cache reset flag
        static std::atomic<bool> g_cache_reset_requested{false};

    } // anonymous namespace

// ============================================================================
// PUBLIC API
// ============================================================================

    void on_token(JNIEnv* env, jobject cb, const std::string& txt) {
        if (!cb || txt.empty()) return;

        // Check for reset request
        if (g_cache_reset_requested.exchange(false)) {
            g_cache.release(env);
        }

        g_cache.init(env, cb);
        if (!g_cache.onToken) return;

        // Create Java string from UTF-8
        jstring jstr = nullptr;

        // Fast path: check if ASCII-only (most common case for LLM tokens)
        bool is_ascii = true;
        for (unsigned char c : txt) {
            if (c >= 0x80) {
                is_ascii = false;
                break;
            }
        }

        if (is_ascii) {
            // Fast path: ASCII-only, use NewStringUTF directly
            jstr = env->NewStringUTF(txt.c_str());
        } else {
            // Slow path: full UTF-8 to UTF-16 conversion
            jstr = utf8::to_jstring_immediate(env, txt);
        }

        if (jstr) {
            // Call Java callback
            env->CallVoidMethod(cb, g_cache.onToken, jstr);

            // Clean up local reference
            env->DeleteLocalRef(jstr);
        }
    }

    void on_error(JNIEnv* env, jobject cb, const char* msg) {
        if (!cb) return;

        g_cache.init(env, cb);
        if (!g_cache.onError) return;

        jstring jmsg = env->NewStringUTF(msg ? msg : "<unknown error>");
        env->CallVoidMethod(cb, g_cache.onError, jmsg);
        env->DeleteLocalRef(jmsg);
    }

    void on_toolcall(JNIEnv* env, jobject cb,
                     const std::string& name,
                     const std::string& payload) {
        if (!cb) return;

        g_cache.init(env, cb);
        if (!g_cache.onToolCall) return;

        jstring jname = env->NewStringUTF(name.c_str());
        jstring jpayload = utf8::to_jstring_immediate(env, payload);

        env->CallVoidMethod(cb, g_cache.onToolCall, jname, jpayload);

        env->DeleteLocalRef(jname);
        env->DeleteLocalRef(jpayload);
    }

    void on_done(JNIEnv* env, jobject cb) {
        if (!cb) return;

        g_cache.init(env, cb);
        if (!g_cache.onDone) return;

        env->CallVoidMethod(cb, g_cache.onDone);
    }

    void reset_cache() {
        g_cache_reset_requested.store(true, std::memory_order_relaxed);
    }

} // namespace jni