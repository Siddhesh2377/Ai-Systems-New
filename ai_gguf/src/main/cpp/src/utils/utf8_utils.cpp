/**
 * Optimized UTF-8 utilities for JNI string conversion
 *
 * Optimizations:
 * 1. Fast ASCII path (most tokens are ASCII-only)
 * 2. SIMD-friendly loop structure
 * 3. Proper surrogate pair handling for emojis
 * 4. Minimal allocations via reserve()
 * 5. Immediate conversion mode for streaming (no buffering)
 */

#include "utf8_utils.h"
#include "logger.h"

#include <cassert>
#include <cstring>

namespace utf8 {

// Thread-local carry buffer for legacy API
    static thread_local std::string t_carry;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

    namespace {

// Check if a UTF-16 code unit is a high surrogate (first half of emoji)
        constexpr bool is_high_surrogate(char16_t c) {
            return (c >= 0xD800 && c <= 0xDBFF);
        }

// Check if a UTF-16 code unit is a low surrogate (second half of emoji)
        constexpr bool is_low_surrogate(char16_t c) {
            return (c >= 0xDC00 && c <= 0xDFFF);
        }

// Combine surrogate pair into Unicode codepoint
        constexpr uint32_t surrogate_to_codepoint(char16_t high, char16_t low) {
            return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
        }

// Get expected length of UTF-8 sequence from first byte
        constexpr size_t utf8_char_length(unsigned char c) {
            if ((c & 0x80) == 0x00) return 1;      // 0xxxxxxx - ASCII
            if ((c & 0xE0) == 0xC0) return 2;      // 110xxxxx
            if ((c & 0xF0) == 0xE0) return 3;      // 1110xxxx
            if ((c & 0xF8) == 0xF0) return 4;      // 11110xxx
            return 0; // Invalid start byte
        }

// Encode a Unicode codepoint to UTF-8
// Optimized with minimal branches
        inline void encode_utf8(uint32_t cp, std::string& out) {
            if (cp <= 0x7F) {
                out.push_back(static_cast<char>(cp));
            } else if (cp <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp <= 0xFFFF) {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp <= 0x10FFFF) {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                // Invalid codepoint - emit replacement character
                out.append("\xEF\xBF\xBD");
            }
        }

// Check if string is ASCII-only (fast path check)
        inline bool is_ascii_only(const std::string& s) {
            for (size_t i = 0; i < s.size(); ++i) {
                if (static_cast<unsigned char>(s[i]) >= 0x80) {
                    return false;
                }
            }
            return true;
        }

// Check if UTF-16 string is BMP-only (no surrogates)
        inline bool is_bmp_only(const jchar* chars, jsize len) {
            for (jsize i = 0; i < len; ++i) {
                if (chars[i] >= 0xD800 && chars[i] <= 0xDFFF) {
                    return false;
                }
            }
            return true;
        }

    } // anonymous namespace

// ============================================================================
// PUBLIC API: from_jstring
// Convert Java String (UTF-16) to UTF-8 std::string
// ============================================================================
    std::string from_jstring(JNIEnv* env, jstring js) {
        if (!js) return {};

        jsize len = env->GetStringLength(js);
        if (len == 0) return {};

        const jchar* chars = env->GetStringChars(js, nullptr);
        if (!chars) return {};

        std::string out;

        // Fast path: BMP-only strings (no surrogate pairs)
        if (is_bmp_only(chars, len)) {
            out.reserve(len * 3); // Worst case: all 3-byte chars

            for (jsize i = 0; i < len; ++i) {
                encode_utf8(static_cast<uint32_t>(chars[i]), out);
            }
        } else {
            // Slow path: handle surrogate pairs (emojis, etc.)
            out.reserve(len * 4);

            for (jsize i = 0; i < len; ++i) {
                char16_t unit = static_cast<char16_t>(chars[i]);

                if (is_high_surrogate(unit)) {
                    if (i + 1 < len && is_low_surrogate(static_cast<char16_t>(chars[i + 1]))) {
                        uint32_t codepoint = surrogate_to_codepoint(unit, static_cast<char16_t>(chars[i + 1]));
                        encode_utf8(codepoint, out);
                        ++i;
                    } else {
                        // Unpaired high surrogate - emit replacement
                        out.append("\xEF\xBF\xBD");
                    }
                } else if (is_low_surrogate(unit)) {
                    // Unpaired low surrogate - emit replacement
                    out.append("\xEF\xBF\xBD");
                } else {
                    encode_utf8(static_cast<uint32_t>(unit), out);
                }
            }
        }

        env->ReleaseStringChars(js, chars);
        return out;
    }

// ============================================================================
// PUBLIC API: to_jstring_immediate
// Convert UTF-8 to Java String - IMMEDIATE (no buffering)
// Optimized for streaming: converts immediately without carry buffer
// ============================================================================
    jstring to_jstring_immediate(JNIEnv* env, const std::string& utf8) {
        if (utf8.empty()) {
            return env->NewStringUTF("");
        }

        // Fast path: ASCII-only strings
        if (is_ascii_only(utf8)) {
            return env->NewStringUTF(utf8.c_str());
        }

        // Full UTF-8 to UTF-16 conversion
        std::u16string u16;
        u16.reserve(utf8.size());

        size_t i = 0;
        while (i < utf8.size()) {
            unsigned char c = static_cast<unsigned char>(utf8[i]);
            size_t char_len = utf8_char_length(c);

            if (char_len == 0) {
                // Invalid start byte - emit replacement and skip
                u16.push_back(0xFFFD);
                ++i;
                continue;
            }

            // Check if we have enough bytes
            if (i + char_len > utf8.size()) {
                // Incomplete sequence at end - emit replacement
                u16.push_back(0xFFFD);
                break;
            }

            // Decode UTF-8 to codepoint
            uint32_t cp = 0;
            bool valid = true;

            switch (char_len) {
                case 1:
                    cp = c;
                    break;
                case 2:
                    cp = (c & 0x1F) << 6;
                    if ((static_cast<unsigned char>(utf8[i + 1]) & 0xC0) == 0x80) {
                        cp |= (static_cast<unsigned char>(utf8[i + 1]) & 0x3F);
                    } else {
                        valid = false;
                    }
                    break;
                case 3:
                    cp = (c & 0x0F) << 12;
                    if ((static_cast<unsigned char>(utf8[i + 1]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(utf8[i + 2]) & 0xC0) == 0x80) {
                        cp |= ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 6);
                        cp |= (static_cast<unsigned char>(utf8[i + 2]) & 0x3F);
                    } else {
                        valid = false;
                    }
                    break;
                case 4:
                    cp = (c & 0x07) << 18;
                    if ((static_cast<unsigned char>(utf8[i + 1]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(utf8[i + 2]) & 0xC0) == 0x80 &&
                        (static_cast<unsigned char>(utf8[i + 3]) & 0xC0) == 0x80) {
                        cp |= ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 12);
                        cp |= ((static_cast<unsigned char>(utf8[i + 2]) & 0x3F) << 6);
                        cp |= (static_cast<unsigned char>(utf8[i + 3]) & 0x3F);
                    } else {
                        valid = false;
                    }
                    break;
                default:
                    valid = false;
            }

            if (!valid) {
                u16.push_back(0xFFFD);
                ++i;
                continue;
            }

            // Encode to UTF-16
            if (cp > 0xFFFF) {
                // Need surrogate pair
                cp -= 0x10000;
                u16.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
                u16.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
            } else {
                u16.push_back(static_cast<char16_t>(cp));
            }

            i += char_len;
        }

        return env->NewString(reinterpret_cast<const jchar*>(u16.data()),
                              static_cast<jsize>(u16.size()));
    }

// ============================================================================
// LEGACY API: to_jstring with carry buffer
// For backwards compatibility with buffered streaming
// ============================================================================
    jstring to_jstring(JNIEnv* env, const std::string& utf8, std::string& carry_buffer) {
        std::u16string u16;
        u16.reserve(utf8.size());

        size_t i = 0;
        while (i < utf8.size()) {
            unsigned char c = static_cast<unsigned char>(utf8[i]);

            if (c <= 0x7F) {
                u16.push_back(static_cast<char16_t>(c));
                ++i;
            } else if ((c & 0xE0) == 0xC0) {
                if (i + 1 < utf8.size()) {
                    uint32_t cp = ((c & 0x1F) << 6) |
                                  (static_cast<unsigned char>(utf8[i + 1]) & 0x3F);
                    u16.push_back(static_cast<char16_t>(cp));
                    i += 2;
                } else {
                    carry_buffer.assign(utf8.data() + i, utf8.size() - i);
                    break;
                }
            } else if ((c & 0xF0) == 0xE0) {
                if (i + 2 < utf8.size()) {
                    uint32_t cp = ((c & 0x0F) << 12) |
                                  ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 6) |
                                  (static_cast<unsigned char>(utf8[i + 2]) & 0x3F);
                    u16.push_back(static_cast<char16_t>(cp));
                    i += 3;
                } else {
                    carry_buffer.assign(utf8.data() + i, utf8.size() - i);
                    break;
                }
            } else if ((c & 0xF8) == 0xF0) {
                if (i + 3 < utf8.size()) {
                    uint32_t cp = ((c & 0x07) << 18) |
                                  ((static_cast<unsigned char>(utf8[i + 1]) & 0x3F) << 12) |
                                  ((static_cast<unsigned char>(utf8[i + 2]) & 0x3F) << 6) |
                                  (static_cast<unsigned char>(utf8[i + 3]) & 0x3F);

                    if (cp > 0xFFFF) {
                        cp -= 0x10000;
                        u16.push_back(static_cast<char16_t>(0xD800 + (cp >> 10)));
                        u16.push_back(static_cast<char16_t>(0xDC00 + (cp & 0x3FF)));
                    } else {
                        u16.push_back(static_cast<char16_t>(cp));
                    }
                    i += 4;
                } else {
                    carry_buffer.assign(utf8.data() + i, utf8.size() - i);
                    break;
                }
            } else {
                u16.push_back(0xFFFD);
                ++i;
            }
        }

        return env->NewString(reinterpret_cast<const jchar*>(u16.data()),
                              static_cast<jsize>(u16.size()));
    }

// ============================================================================
// LEGACY API: flush_carry
// ============================================================================
    void flush_carry(JNIEnv* env, jobject cb) {
        if (t_carry.empty()) return;

        t_carry.clear();

        jclass cls = env->GetObjectClass(cb);
        if (!cls) return;

        jmethodID mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
        if (!mid) return;

        // Emit replacement character for incomplete sequence
        static const char replacement[] = "\xEF\xBF\xBD";
        jstring js = env->NewStringUTF(replacement);
        env->CallVoidMethod(cb, mid, js);
        env->DeleteLocalRef(js);
    }

    void clear_carry_buffer() {
        t_carry.clear();
    }

} // namespace utf8