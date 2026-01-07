#pragma once

/**
 * Optimized UTF-8 utilities for JNI string conversion
 * 
 * Key features:
 * - Fast ASCII path for common case
 * - Proper surrogate pair handling for emojis/extended Unicode
 * - Immediate conversion without buffering for streaming
 */

#include <jni.h>
#include <string>
#include <cstdint>

namespace utf8 {

/**
 * Convert Java jstring to UTF-8 std::string
 * Handles all Unicode including emoji (surrogate pairs)
 */
    std::string from_jstring(JNIEnv* env, jstring js);

/**
 * Convert UTF-8 string to jstring - IMMEDIATE (no buffering)
 * This is the optimized version for streaming - converts immediately
 * without any carry buffer logic
 */
    jstring to_jstring_immediate(JNIEnv* env, const std::string& utf8);

/**
 * Legacy buffered conversion for backwards compatibility
 * Handles incomplete UTF-8 sequences across calls
 */
    jstring to_jstring(JNIEnv* env, const std::string& utf8, std::string& carry_buffer);

/**
 * Flush any remaining carry buffer (legacy)
 */
    void flush_carry(JNIEnv* env, jobject cb);

/**
 * Clear thread-local carry buffer
 */
    void clear_carry_buffer();

} // namespace utf8