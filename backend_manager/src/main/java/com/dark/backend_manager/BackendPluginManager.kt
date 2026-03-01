package com.dark.backend_manager

import android.util.Log
import com.dark.backend_manager.hardware.NativeHardwareObserver
import com.dark.backend_manager.loader.PluginLoader
import com.dark.backend_manager.registry.InstallStatus
import com.dark.backend_manager.registry.PluginRegistry
import com.dark.backend_plugin_api.callback.ConflictCallback
import com.dark.backend_plugin_api.core.BackendPlugin
import com.dark.backend_plugin_api.core.BackendState
import com.dark.backend_plugin_api.core.HardwareObserver
import com.dark.backend_plugin_api.core.PluginManager
import com.dark.backend_plugin_api.model.BackendManifest
import com.dark.backend_plugin_api.model.ModelMetadata
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File

/**
 * Concrete implementation of [PluginManager].
 * Manages the full lifecycle: discovery, loading, conflict resolution, memory checks.
 *
 * Usage from ToolNeuron:
 * ```
 * val manager = BackendPluginManager(context.cacheDir)
 * manager.init(filesDir.resolve("plugins").path, conflictCallback)
 *
 * // When user picks a model:
 * val backend = manager.requestBackend(modelMetadata).getOrThrow()
 * val textGen = backend as TextGenBackend
 * textGen.generateStream(messagesJson, maxTokens, callback)
 * ```
 */
class BackendPluginManager(cacheDir: File) : PluginManager {

    companion object {
        private const val TAG = "BackendPluginManager"

        // Minimum free RAM to keep after loading a backend (256MB safety margin)
        private const val RAM_SAFETY_MARGIN_BYTES = 256L * 1024 * 1024
    }

    private val registry = PluginRegistry()
    private val loader = PluginLoader(cacheDir)
    private val _hardware = NativeHardwareObserver()

    private val _loadedBackends = MutableStateFlow<Map<String, BackendPlugin>>(emptyMap())
    private val mutex = Mutex()
    private var conflictCallback: ConflictCallback? = null

    // --- PluginManager interface ---

    override val installedBackends: StateFlow<List<BackendManifest>>
        get() = MutableStateFlow(registry.getManifests()).asStateFlow()

    override val loadedBackends: StateFlow<Map<String, BackendPlugin>> =
        _loadedBackends.asStateFlow()

    override val hardware: HardwareObserver get() = _hardware

    override suspend fun init(pluginsDir: String, conflictCallback: ConflictCallback) {
        this.conflictCallback = conflictCallback
        registry.scanPluginsDir(pluginsDir)
        Log.i(TAG, "Initialized with ${registry.entries.value.size} installed backends")
    }

    override suspend fun requestBackend(model: ModelMetadata): Result<BackendPlugin> = mutex.withLock {
        runCatching {
            val backendId = model.requiredBackend

            // 1. Check if already loaded with this model
            val existing = _loadedBackends.value[backendId]
            if (existing != null && existing.state.value == BackendState.READY) {
                return@runCatching existing
            }

            // 2. Check if installed
            val entry = registry.getEntry(backendId)
                ?: throw IllegalStateException("Backend '$backendId' is not installed. Download it first.")
            require(entry.status == InstallStatus.INSTALLED) {
                "Backend '$backendId' is ${entry.status}. Re-download required."
            }

            // 3. Check for conflicts — is another backend running?
            val runningBackend = findRunningBackend()
            if (runningBackend != null && runningBackend.id != backendId) {
                val runningTask = runningBackend.activeTask()
                    ?: throw IllegalStateException("Backend ${runningBackend.id} in inconsistent state")

                val shouldPause = conflictCallback?.onConflict(
                    runningBackendId = runningBackend.id,
                    runningTask = runningTask,
                    requestedBackendId = backendId,
                    requestedTask = model.capabilities.firstOrNull()?.let { cap ->
                        com.dark.backend_plugin_api.core.TaskType.entries.firstOrNull { it.name.contains(cap.name, ignoreCase = true) }
                    } ?: com.dark.backend_plugin_api.core.TaskType.TEXT_GENERATION
                ) ?: false

                if (shouldPause) {
                    Log.i(TAG, "User chose to pause ${runningBackend.id}")
                    runningBackend.pause()
                } else {
                    throw CancellationException("User cancelled: ${runningBackend.id} is still running")
                }
            }

            // 4. Memory check
            val needed = estimateLoadMemory(model, entry)
            val available = _hardware.getAvailableRamBytes()
            if (needed > 0 && (available - needed) < RAM_SAFETY_MARGIN_BYTES) {
                // Try unloading idle backends to free memory
                freeIdleBackends(backendId)
                val availableAfterFree = _hardware.getAvailableRamBytes()
                if ((availableAfterFree - needed) < RAM_SAFETY_MARGIN_BYTES) {
                    throw OutOfMemoryError(
                        "Not enough RAM: need ${needed.toMB()}MB, " +
                                "available ${availableAfterFree.toMB()}MB " +
                                "(${RAM_SAFETY_MARGIN_BYTES.toMB()}MB safety margin)"
                    )
                }
            }

            // 5. Load the backend plugin if not already loaded
            val plugin = existing ?: run {
                val loaded = loader.load(entry.manifest, entry.installPath).getOrThrow()
                _loadedBackends.value = _loadedBackends.value + (backendId to loaded)
                loaded
            }

            // 6. Load the model
            if (plugin.state.value == BackendState.UNLOADED ||
                plugin.state.value == BackendState.ERROR) {
                plugin.loadModel(model).getOrThrow()
            }

            Log.i(TAG, "Backend '$backendId' ready with model '${model.name}'")
            plugin
        }
    }

    override fun getBackend(id: String): BackendPlugin? = _loadedBackends.value[id]

    override fun isInstalled(backendId: String): Boolean =
        registry.getEntry(backendId)?.status == InstallStatus.INSTALLED

    override suspend fun unloadBackend(id: String) = mutex.withLock {
        val plugin = _loadedBackends.value[id] ?: return@withLock
        plugin.release()
        loader.unload(id)
        _loadedBackends.value = _loadedBackends.value - id
        Log.i(TAG, "Unloaded backend: $id")
    }

    override suspend fun releaseAll() = mutex.withLock {
        _loadedBackends.value.forEach { (id, plugin) ->
            runCatching {
                plugin.release()
                loader.unload(id)
            }.onFailure { Log.w(TAG, "Error releasing $id: ${it.message}") }
        }
        _loadedBackends.value = emptyMap()
        Log.i(TAG, "All backends released")
    }

    override fun registerInstalled(manifest: BackendManifest, installPath: String) {
        registry.register(manifest, installPath)
    }

    override fun unregisterBackend(id: String) {
        registry.unregister(id)
    }

    // --- Internal helpers ---

    private fun findRunningBackend(): BackendPlugin? =
        _loadedBackends.value.values.firstOrNull {
            it.state.value == BackendState.RUNNING
        }

    private fun estimateLoadMemory(model: ModelMetadata, entry: com.dark.backend_manager.registry.InstalledBackendEntry): Long {
        // If backend is already loaded, ask it directly
        val loaded = _loadedBackends.value[entry.manifest.id]
        if (loaded != null) return loaded.estimateMemory(model)

        // Rough estimate from file size: model file + ~20% overhead for KV cache/buffers
        return (model.fileSizeBytes * 1.2).toLong()
    }

    private suspend fun freeIdleBackends(exceptId: String) {
        val idle = _loadedBackends.value.filter { (id, plugin) ->
            id != exceptId && plugin.state.value == BackendState.READY && plugin.activeTask() == null
        }
        for ((id, plugin) in idle) {
            Log.i(TAG, "Freeing idle backend '$id' to reclaim memory")
            plugin.release()
            loader.unload(id)
            _loadedBackends.value = _loadedBackends.value - id
        }
    }

    private fun Long.toMB(): Long = this / (1024 * 1024)

    class CancellationException(message: String) : Exception(message)
}
