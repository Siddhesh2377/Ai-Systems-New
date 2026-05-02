// Copyright (c) 2025 Dark Matter Labs
package com.dark.ai_sherpa

object SherpaLib {
    init {
        System.loadLibrary("ai_sherpa")
    }

    /** Call from Application.onCreate() or before first use to ensure the native library is loaded. */
    fun init() { /* triggers the companion object init block */ }

    external fun nativeErrorInit()
    external fun nativeErrorSetCrashLogPath(path: String)
    external fun nativeErrorGetLastJson(): String
    external fun nativeErrorClear()
}
