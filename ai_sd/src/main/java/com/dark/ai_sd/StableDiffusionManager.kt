package com.dark.ai_sd

import android.content.Context

// Initialize in a coroutine or background thread
class StableDiffusionManager(private val context: Context) {
    private val nativeLib = NativeLibStableDiffusion()
    private var serverThread: Thread? = null
    
    fun initialize(modelPaths: ModelPaths): Boolean {
        val result = nativeLib.initialize(
            clipPath = modelPaths.clipPath,
            unetPath = modelPaths.unetPath,
            vaeDecoderPath = modelPaths.vaeDecoderPath,
            vaeEncoderPath = modelPaths.vaeEncoderPath,
            tokenizerPath = modelPaths.tokenizerPath,
            backendPath = modelPaths.backendPath,
            systemLibraryPath = modelPaths.systemLibraryPath,
            safetyCheckerPath = modelPaths.safetyCheckerPath,
            patchPath = modelPaths.patchPath,
            port = 8081,
            useMnn = false,
            useMnnClip = false,
            ponyv55 = false,
            useSafetyChecker = false,
            upscalerMode = false
        )
        return result == 0
    }
    
    fun startServer() {
        serverThread = Thread {
            nativeLib.startServer()
        }.apply {
            name = "SD-Server-Thread"
            start()
        }
    }
    
    fun cleanup() {
        nativeLib.cleanup()
        serverThread?.interrupt()
        serverThread = null
    }
}

data class ModelPaths(
    val clipPath: String,
    val unetPath: String,
    val vaeDecoderPath: String,
    val vaeEncoderPath: String? = null,
    val tokenizerPath: String,
    val backendPath: String? = null,
    val systemLibraryPath: String? = null,
    val safetyCheckerPath: String? = null,
    val patchPath: String? = null
)