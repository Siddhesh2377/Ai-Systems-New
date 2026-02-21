#pragma once

/**
 * Header-only logger for ai_sd
 * Android logcat output with runtime level control.
 */

#include <cstdio>
#include <cstdarg>
#include <atomic>

#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_PLATFORM_ANDROID 1
#else
#define LOG_PLATFORM_ANDROID 0
#endif

namespace sd_log {
    enum class Level : int {
        Error   = 1,
        Warning = 2,
        Info    = 3,
        Debug   = 4
    };

    inline std::atomic<Level>& level_ref() {
        static std::atomic<Level> lvl{Level::Info};
        return lvl;
    }

    inline Level get_level() {
        return level_ref().load(std::memory_order_relaxed);
    }

    inline void set_level(Level l) {
        level_ref().store(l, std::memory_order_relaxed);
    }

    inline void logf(Level level, const char* fmt, ...) {
        if (level > get_level()) return;

#if LOG_PLATFORM_ANDROID
        int android_lvl = ANDROID_LOG_INFO;
        switch (level) {
            case Level::Error:   android_lvl = ANDROID_LOG_ERROR;   break;
            case Level::Warning: android_lvl = ANDROID_LOG_WARN;    break;
            case Level::Info:    android_lvl = ANDROID_LOG_INFO;    break;
            case Level::Debug:   android_lvl = ANDROID_LOG_DEBUG;   break;
        }
        va_list ap;
        va_start(ap, fmt);
        __android_log_vprint(android_lvl, "ai_sd", fmt, ap);
        va_end(ap);
#else
        va_list ap;
        va_start(ap, fmt);
        if (level == Level::Error || level == Level::Warning)
            std::vfprintf(stderr, fmt, ap);
        else
            std::vfprintf(stdout, fmt, ap);
        std::fprintf(stdout, "\n");
        va_end(ap);
#endif
    }
} // namespace sd_log

#define SD_LOG_ERROR(...)  ::sd_log::logf(::sd_log::Level::Error,   __VA_ARGS__)
#define SD_LOG_WARN(...)   ::sd_log::logf(::sd_log::Level::Warning, __VA_ARGS__)
#define SD_LOG_INFO(...)   ::sd_log::logf(::sd_log::Level::Info,    __VA_ARGS__)
#define SD_LOG_DEBUG(...)  ::sd_log::logf(::sd_log::Level::Debug,   __VA_ARGS__)

#ifndef NDEBUG
#else
#undef SD_LOG_DEBUG
#define SD_LOG_DEBUG(...)
#endif
