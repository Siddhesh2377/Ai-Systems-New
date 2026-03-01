package com.dark.backend_plugin_api.core

import com.dark.backend_plugin_api.model.ModelMetadata
import kotlinx.coroutines.flow.StateFlow

/**
 * The main contract every backend must implement.
 * Each backend plugin (ai_gguf, ai_sd, ai_tts, future engines) provides
 * a class implementing this interface as the entry point.
 *
 * The entry class name is declared in manifest.json and instantiated
 * by the plugin loader via reflection.
 */
interface BackendPlugin {

    /** Unique backend ID — must match manifest.json "id" field */
    val id: String

    /** Human-readable backend name */
    val name: String

    /** What this backend can do */
    val capabilities: Set<Capability>

    /** Observable lifecycle state */
    val state: StateFlow<BackendState>

    /**
     * Estimate RAM needed to load this model (bytes).
     * Called BEFORE loadModel() to check if the device has enough memory.
     * Return 0 if estimation is not possible.
     */
    fun estimateMemory(model: ModelMetadata): Long

    /**
     * Load a model. This is a heavy operation (can take seconds).
     * Must transition state: UNLOADED -> LOADING -> READY (or ERROR).
     */
    suspend fun loadModel(model: ModelMetadata): Result<Unit>

    /** Unload model and free all native memory. State -> UNLOADED. */
    suspend fun unloadModel()

    /**
     * Pause active generation but keep the model in memory.
     * State: RUNNING -> PAUSED. KV cache / intermediate state preserved.
     */
    suspend fun pause(): Result<Unit>

    /**
     * Resume paused generation.
     * State: PAUSED -> RUNNING.
     */
    suspend fun resume(): Result<Unit>

    /** Check if this backend can handle the given model (file format, capabilities, etc.) */
    fun canHandle(model: ModelMetadata): Boolean

    /** Get the currently active task type, or null if idle (state is READY) */
    fun activeTask(): TaskType?

    /** Backend-specific info: model name, loaded quant, perf stats, etc. */
    fun getInfo(): Map<String, String>

    /** Clean shutdown — release everything, free all native resources. */
    suspend fun release()
}
