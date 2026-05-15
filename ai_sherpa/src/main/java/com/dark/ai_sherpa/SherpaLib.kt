// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

/**
 * Native library bootstrap.
 *
 * Loads `libai_sherpa.so` lazily on first reference. All other classes in this
 * module call [init] in their `companion object` initializer to make sure the
 * library is loaded before any JNI method is invoked.
 *
 * Error reporting and crash capture are owned by the `:tn_security` module —
 * the host app (ToolNeuron) installs signal handlers and the crash-file
 * pattern once via `com.dark.tn_security.TnSecurity`. Every log line, error,
 * and cancellation from this SDK (plus the upstream sherpa-onnx library)
 * funnels through that same sink and shares one unified event stream with
 * the other on-device SDKs (gguf_lib, ai_sd, …).
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
}
