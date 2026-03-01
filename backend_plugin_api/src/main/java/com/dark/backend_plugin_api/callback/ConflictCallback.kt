package com.dark.backend_plugin_api.callback

import com.dark.backend_plugin_api.core.TaskType

/**
 * Called by PluginManager when a new request conflicts with a running task.
 * The app (ToolNeuron) implements this to show the user a dialog like:
 * "Text generation is running. Pause it to start image generation?"
 */
interface ConflictCallback {
    /**
     * A new task wants to run but [runningTask] is active on [runningBackendId].
     * Return true to pause the running task, false to cancel the new request.
     */
    suspend fun onConflict(
        runningBackendId: String,
        runningTask: TaskType,
        requestedBackendId: String,
        requestedTask: TaskType
    ): Boolean
}
