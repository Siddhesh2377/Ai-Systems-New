package com.dark.backend_plugin_api.callback

interface ImageGenCallback {
    fun onProgress(step: Int, totalSteps: Int)
    fun onImageReady(rgbData: ByteArray, width: Int, height: Int)
    fun onError(message: String)
}
