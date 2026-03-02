#pragma once

/**
 * Production logging for ai_sd.
 *
 * Level hierarchy: NONE < ERROR < WARN < INFO < DEBUG < TRACE
 * Compile-time gate: SD_LOG_LEVEL_MAX (default 3=INFO)
 * Runtime gate: sd_log_set_level()
 * Zero heap allocation: 512-char stack buffer
 * Structured tags: [CLIP] [UNET] [VAE] [SCHED] [TILE] [JNI] [SAFETY] [LOAD] [GEN] [UPSCL]
 * Timing macros: SD_TIMER_START(name) / SD_TIMER_END(name, tag)
 */

#include <cstdio>
#include <cstdarg>
#include <atomic>
#include <chrono>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

// Compile-time maximum log level (strips dead code in release)
// 0=NONE, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE
#ifndef SD_LOG_LEVEL_MAX
#define SD_LOG_LEVEL_MAX 3
#endif

#define SD_LOG_TAG "ai_sd"

namespace sd_log {

enum Level : int {
    NONE  = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    DEBUG = 4,
    TRACE = 5
};

inline std::atomic<int>& runtime_level() {
    static std::atomic<int> lvl{INFO};
    return lvl;
}

inline void set_level(int l) { runtime_level().store(l, std::memory_order_relaxed); }
inline int  get_level()      { return runtime_level().load(std::memory_order_relaxed); }

inline void log_msg(int level, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
inline void log_msg(int level, const char* fmt, ...) {
    if (level > get_level()) return;

    va_list ap;
    va_start(ap, fmt);

#if defined(__ANDROID__)
    int prio;
    switch (level) {
        case ERROR: prio = ANDROID_LOG_ERROR; break;
        case WARN:  prio = ANDROID_LOG_WARN;  break;
        case INFO:  prio = ANDROID_LOG_INFO;  break;
        case DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default:    prio = ANDROID_LOG_VERBOSE; break;
    }
    __android_log_vprint(prio, SD_LOG_TAG, fmt, ap);
#else
    char buf[512];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    if (level <= WARN)
        fprintf(stderr, "%s\n", buf);
    else
        fprintf(stdout, "%s\n", buf);
#endif

    va_end(ap);
}

} // namespace sd_log

// --- Level macros with compile-time gate ---

#if SD_LOG_LEVEL_MAX >= 1
#define SD_LOG_ERROR(...) ::sd_log::log_msg(::sd_log::ERROR, __VA_ARGS__)
#else
#define SD_LOG_ERROR(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 2
#define SD_LOG_WARN(...) ::sd_log::log_msg(::sd_log::WARN, __VA_ARGS__)
#else
#define SD_LOG_WARN(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 3
#define SD_LOG_INFO(...) ::sd_log::log_msg(::sd_log::INFO, __VA_ARGS__)
#else
#define SD_LOG_INFO(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 4
#define SD_LOG_DEBUG(...) ::sd_log::log_msg(::sd_log::DEBUG, __VA_ARGS__)
#else
#define SD_LOG_DEBUG(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 5
#define SD_LOG_TRACE(...) ::sd_log::log_msg(::sd_log::TRACE, __VA_ARGS__)
#else
#define SD_LOG_TRACE(...) ((void)0)
#endif

// --- Timing macros ---
// Usage:
//   SD_TIMER_START(clip);
//   ... work ...
//   SD_TIMER_END(clip, "[CLIP]");  // logs: [CLIP] 12ms

#if SD_LOG_LEVEL_MAX >= 4
#define SD_TIMER_START(name) \
    auto _sd_timer_##name = std::chrono::steady_clock::now()

#define SD_TIMER_END(name, tag) \
    do { \
        auto _sd_timer_end_##name = std::chrono::steady_clock::now(); \
        auto _sd_timer_ms_##name = std::chrono::duration_cast<std::chrono::milliseconds>( \
            _sd_timer_end_##name - _sd_timer_##name).count(); \
        SD_LOG_DEBUG("%s %ldms", tag, (long)_sd_timer_ms_##name); \
    } while (0)

#define SD_TIMER_END_INFO(name, tag, fmt, ...) \
    do { \
        auto _sd_timer_end_##name = std::chrono::steady_clock::now(); \
        auto _sd_timer_ms_##name = std::chrono::duration_cast<std::chrono::milliseconds>( \
            _sd_timer_end_##name - _sd_timer_##name).count(); \
        SD_LOG_INFO(fmt, __VA_ARGS__, (long)_sd_timer_ms_##name); \
    } while (0)
#else
#define SD_TIMER_START(name) ((void)0)
#define SD_TIMER_END(name, tag) ((void)0)
#define SD_TIMER_END_INFO(name, tag, fmt, ...) ((void)0)
#endif

// --- Convenience: set runtime level from JNI ---
inline void sd_log_set_level(int level) { sd_log::set_level(level); }
