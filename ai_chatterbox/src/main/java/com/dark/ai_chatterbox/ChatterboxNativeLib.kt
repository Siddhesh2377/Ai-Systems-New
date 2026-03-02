package com.dark.ai_chatterbox

class ChatterboxNativeLib {
    external fun nativePing(): Boolean

    companion object {
        init {
            System.loadLibrary("ai_chatterbox")
        }
    }
}
