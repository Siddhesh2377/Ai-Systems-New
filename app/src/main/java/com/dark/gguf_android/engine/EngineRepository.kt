package com.dark.gguf_android.engine

import com.mp.ai_gguf_android.GGUFAndroidLib
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

object EngineRepository {
    private var lib: GGUFAndroidLib? = null

    private val _isCreated = MutableStateFlow(false)
    val isCreated: StateFlow<Boolean> = _isCreated.asStateFlow()

    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()

    fun create(threads: Int = 4): GGUFAndroidLib {
        if (lib != null) return lib!!
        val instance = GGUFAndroidLib(threads)
        lib = instance
        _isCreated.value = true
        return instance
    }

    fun loadModel(fd: Int, maxCtx: Int = 2048): Boolean {
        val result = lib?.loadModel(fd, maxCtx) ?: false
        _isModelLoaded.value = result
        return result
    }

    fun reset() {
        lib?.reset()
        _isModelLoaded.value = false
    }

    fun release() {
        lib?.release()
        lib = null
        _isCreated.value = false
        _isModelLoaded.value = false
    }

    fun getLib(): GGUFAndroidLib {
        return lib ?: throw IllegalStateException("Engine not created. Call create() first.")
    }

    fun getLibOrNull(): GGUFAndroidLib? = lib
}
