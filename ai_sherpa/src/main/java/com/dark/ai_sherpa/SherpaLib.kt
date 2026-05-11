// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * Native library bootstrap and process-wide error/crash bridge.
 *
 * Loads `libai_sherpa.so` lazily on first reference. All other classes in this
 * module call [init] in their `companion object` initializer to make sure the
 * library is loaded before any JNI method is invoked.
 *
 * The error/crash API is designed to be polled from the Kotlin side after a
 * native call fails or after process restart following a tombstone:
 *
 *  - Call [nativeErrorInit] once, early, to install signal handlers.
 *  - Call [nativeErrorSetCrashLogPath] with a writable file in your app's
 *    private storage; the handler writes a small JSON blob there before
 *    re-raising so the upstream tombstone path still runs.
 *  - After any failure, [nativeErrorGetLastJson] returns the last error as
 *    a JSON object with shape `{code, category, message, op_at_time, timestamp}`.
 *  - [nativeErrorClear] resets both the last-error and current-op slots.
 */
object SherpaLib {

    init {
        System.loadLibrary("ai_sherpa")
    }

    /**
     * Forces the library to load (no-op if already loaded). Call from
     * `Application.onCreate()` if you want loading to be observable on the
     * main thread rather than deferred to first use.
     */
    fun init() { /* triggers companion init */ }

    /**
     * Installs native crash handlers (SIGSEGV, SIGABRT, SIGFPE, SIGBUS, SIGILL).
     * Idempotent — safe to call multiple times. Must be called before any
     * recognizer/TTS load if you want crash JSON written to disk.
     */
    external fun nativeErrorInit()

    /**
     * Sets the path where crash JSON is written when a fatal signal is caught.
     * Use a path inside `context.filesDir` so the app has write permission.
     * The file is overwritten on each crash. The native handler always
     * re-raises after writing, so the kernel still produces a tombstone.
     */
    external fun nativeErrorSetCrashLogPath(path: String)

    /**
     * Returns the most recent non-fatal error as a JSON string, or `"{}"` if
     * none. Shape: `{"code":int,"category":string,"message":string,"op_at_time":{...},"timestamp":long}`.
     */
    external fun nativeErrorGetLastJson(): String

    /**
     * Clears the last-error and current-op slots. Call after surfacing the
     * error to the user so the next failure isn't conflated with this one.
     */
    external fun nativeErrorClear()
}
