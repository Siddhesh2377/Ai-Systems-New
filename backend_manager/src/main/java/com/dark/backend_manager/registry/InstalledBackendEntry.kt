package com.dark.backend_manager.registry

import com.dark.backend_plugin_api.model.BackendManifest

/**
 * Represents an installed backend on disk.
 * This is what gets stored in the local-backend-db (Room table in ToolNeuron).
 * The backend_manager works with this data class; ToolNeuron maps it to/from Room entities.
 */
data class InstalledBackendEntry(
    val manifest: BackendManifest,
    val installPath: String,
    val installedAt: Long,
    val status: InstallStatus
)

enum class InstallStatus {
    INSTALLED,
    UPDATING,
    CORRUPT
}
