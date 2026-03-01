package com.dark.backend_manager

class NativeLib {

    /**
     * A native method that is implemented by the 'backend_manager' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'backend_manager' library on application startup.
        init {
            System.loadLibrary("backend_manager")
        }
    }
}