package com.dark.gguf_lib

/**
 * Process-wide native error tracker for the gguf_lib SDK.
 *
 * Pairs the C++ `error_tracker.h/.cpp` machinery (signal handlers, last-op
 * JSON, crash log path) with a public Kotlin surface so consumers can wire
 * crash diagnostics without reaching into the internal JNI bridge.
 *
 * Typical usage from a host service / process:
 * ```kotlin
 * ErrorTracker.init()
 * ErrorTracker.setCrashLogPath(File(filesDir, "gguf_crash.json").absolutePath)
 * ```
 *
 * After a crash or a failed native op, read [getLastErrorJson] to retrieve a
 * structured error blob (op name, detail, message, code). Idempotent — safe
 * to call [init] more than once.
 */
object ErrorTracker {

    /** Install signal handlers (SIGSEGV/SIGABRT/...). Idempotent. */
    fun init() = GGUFNativeLib.nativeErrorInit()

    /** Direct the crash handler to write a structured JSON blob to [path]. */
    fun setCrashLogPath(path: String) = GGUFNativeLib.nativeErrorSetCrashLogPath(path)

    /** Clear the last-error state. Does not affect crash log files on disk. */
    fun clear() = GGUFNativeLib.nativeErrorClear()

    /** Last-error JSON (op + detail + message + code), or "{}" if none. */
    fun getLastErrorJson(): String = GGUFNativeLib.nativeErrorGetLastJson()
}
