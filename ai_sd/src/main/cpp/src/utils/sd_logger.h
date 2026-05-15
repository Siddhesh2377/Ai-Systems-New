#pragma once

/**
 * Production logging for ai_sd.
 *
 * Routes through tn_security under TN_MODULE_AI_SD so every log line lands
 * in the unified sink pipeline (logcat + JSON crash file + Kotlin facade).
 * The historical SD_LOG_* surface is preserved as a thin compatibility shim
 * for the ~400 existing call sites; new code should prefer the TN_I/W/E/D
 * macros from <tn_security/tn_security_macros.h>.
 *
 * Level hierarchy: NONE < ERROR < WARN < INFO < DEBUG < TRACE
 * Compile-time gate: SD_LOG_LEVEL_MAX (default 3=INFO) — kept for source
 * compatibility; the actual level filter lives in tn_security.
 */

#include <cstdio>
#include <chrono>

#include <tn_security/tn_security.h>

// Compile-time maximum log level — historical knob, retained so call-site
// gates still work. The real filtering happens in tn_sec_log.
// 0=NONE, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE
#ifndef SD_LOG_LEVEL_MAX
#define SD_LOG_LEVEL_MAX 3
#endif

#define SD_LOG_TAG "ai_sd"

// --- Level macros with compile-time gate ---
//
// Each macro hard-codes TN_MODULE_AI_SD as the source module. .cpp files that
// also want to emit structured errors via TN_ERR should additionally:
//
//     #define TN_MODULE TN_MODULE_AI_SD
//     #define TN_TAG    "ai_sd"
//     #include <tn_security/tn_security_macros.h>

#if SD_LOG_LEVEL_MAX >= 1
#define SD_LOG_ERROR(...) tn_sec_log(TN_LEVEL_ERROR, TN_MODULE_AI_SD, SD_LOG_TAG, \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SD_LOG_ERROR(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 2
#define SD_LOG_WARN(...) tn_sec_log(TN_LEVEL_WARN, TN_MODULE_AI_SD, SD_LOG_TAG, \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SD_LOG_WARN(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 3
#define SD_LOG_INFO(...) tn_sec_log(TN_LEVEL_INFO, TN_MODULE_AI_SD, SD_LOG_TAG, \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SD_LOG_INFO(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 4
#define SD_LOG_DEBUG(...) tn_sec_log(TN_LEVEL_DEBUG, TN_MODULE_AI_SD, SD_LOG_TAG, \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SD_LOG_DEBUG(...) ((void)0)
#endif

#if SD_LOG_LEVEL_MAX >= 5
#define SD_LOG_TRACE(...) tn_sec_log(TN_LEVEL_TRACE, TN_MODULE_AI_SD, SD_LOG_TAG, \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, __VA_ARGS__)
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

// --- Convenience: runtime level shim ---
// Retained for source compatibility; tn_security has its own runtime gate.
inline void sd_log_set_level(int /*level*/) {}
