#pragma once

/**
 * Optimized JNI utility functions for token streaming
 * 
 * Features:
 * - Cached JNI method IDs for minimal lookup overhead
 * - Thread-safe caching
 * - Immediate token delivery without buffering
 */

#include <jni.h>
#include <string>

namespace jni {

/**
 * Send a token to the Java callback immediately
 * No buffering - each token is delivered as soon as it's decoded
 */
    void on_token(JNIEnv* env, jobject cb, const std::string& txt);

/**
 * Send an error message to the Java callback
 */
    void on_error(JNIEnv* env, jobject cb, const char* msg);

/**
 * Send a tool call to the Java callback
 */
    void on_toolcall(JNIEnv* env, jobject cb,
                     const std::string& name,
                     const std::string& payload);

/**
 * Signal completion to the Java callback
 */
    void on_done(JNIEnv* env, jobject cb);

/**
 * Reset cached JNI references
 * Call this if the callback class might have changed
 */
    void reset_cache();

} // namespace jni