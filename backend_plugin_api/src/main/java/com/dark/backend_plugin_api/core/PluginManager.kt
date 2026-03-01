package com.dark.backend_plugin_api.core

import com.dark.backend_plugin_api.callback.ConflictCallback
import com.dark.backend_plugin_api.model.BackendManifest
import com.dark.backend_plugin_api.model.ModelMetadata
import kotlinx.coroutines.flow.StateFlow

/**
 * Central orchestrator for backend plugins.
 * Interface lives in plugin_api; implementation lives in backend_manager.
 *
 * Lifecycle:
 * 1. init() — scan installed backends from local-backend-db
 * 2. requestBackend(model) — load the right backend for a model
 * 3. getBackend(id) — get a loaded backend to call capability methods
 * 4. conflict resolution — if a task is running, ask user via ConflictCallback
 */
interface PluginManager {

    companion object {
        /** The API version this build of the plugin system expects. Exact match required. */
        const val API_VERSION = 1
    }

    /** All installed backends (from local-backend-db), whether loaded or not */
    val installedBackends: StateFlow<List<BackendManifest>>

    /** Currently loaded (in-memory) backends */
    val loadedBackends: StateFlow<Map<String, BackendPlugin>>

    /** Hardware observer instance */
    val hardware: HardwareObserver

    /**
     * Initialize the plugin manager. Scans installed backends directory.
     * @param pluginsDir path to filesDir/plugins/
     * @param conflictCallback how to ask the user about conflicts
     */
    suspend fun init(pluginsDir: String, conflictCallback: ConflictCallback)

    /**
     * Request a backend for a specific model.
     * - Finds the right backend by model.requiredBackend
     * - Checks if it's installed
     * - Checks memory availability
     * - Handles conflicts with running tasks (via ConflictCallback)
     * - Loads the backend if not already loaded
     * - Loads the model
     *
     * @return the loaded BackendPlugin, or failure with reason
     */
    suspend fun requestBackend(model: ModelMetadata): Result<BackendPlugin>

    /** Get a loaded backend by ID, or null if not loaded */
    fun getBackend(id: String): BackendPlugin?

    /** Check if a backend is installed (present on disk, in local-backend-db) */
    fun isInstalled(backendId: String): Boolean

    /** Unload a specific backend and free its resources */
    suspend fun unloadBackend(id: String)

    /** Unload all backends — shutdown */
    suspend fun releaseAll()

    /**
     * Register an installed backend from its manifest.
     * Called by the download manager after extracting a backend zip.
     */
    fun registerInstalled(manifest: BackendManifest, installPath: String)

    /** Unregister a backend (deleted by user) */
    fun unregisterBackend(id: String)
}
