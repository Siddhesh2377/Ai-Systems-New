package com.dark.ai_sherpa

class NativeLib {

    /**
     * A native method that is implemented by the 'ai_sherpa' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'ai_sherpa' library on application startup.
        init {
            System.loadLibrary("ai_sherpa")
        }
    }
}