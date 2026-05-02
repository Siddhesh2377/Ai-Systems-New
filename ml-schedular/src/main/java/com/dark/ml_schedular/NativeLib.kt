package com.dark.ml_schedular

class NativeLib {

    /**
     * A native method that is implemented by the 'ml_schedular' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'ml_schedular' library on application startup.
        init {
            System.loadLibrary("ml_schedular")
        }
    }
}