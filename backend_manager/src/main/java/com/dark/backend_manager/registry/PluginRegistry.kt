package com.dark.backend_manager.registry

import android.util.Log
import com.dark.backend_plugin_api.model.BackendManifest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * Tracks installed backends. On init, scans the plugins directory and parses manifests.
 * ToolNeuron can also push entries from its Room DB via [register]/[unregister].
 */
class PluginRegistry {

    companion object {
        private const val TAG = "PluginRegistry"
    }

    private val _entries = MutableStateFlow<Map<String, InstalledBackendEntry>>(emptyMap())
    val entries: StateFlow<Map<String, InstalledBackendEntry>> = _entries.asStateFlow()

    /**
     * Scan the plugins directory for installed backends.
     * Each subdirectory should contain a manifest.json.
     */
    fun scanPluginsDir(pluginsDir: String) {
        val dir = File(pluginsDir)
        if (!dir.exists()) {
            Log.w(TAG, "Plugins directory does not exist: $pluginsDir")
            return
        }

        val found = mutableMapOf<String, InstalledBackendEntry>()

        dir.listFiles()?.filter { it.isDirectory }?.forEach { pluginDir ->
            ManifestParser.parse(pluginDir).onSuccess { manifest ->
                found[manifest.id] = InstalledBackendEntry(
                    manifest = manifest,
                    installPath = pluginDir.absolutePath,
                    installedAt = pluginDir.lastModified(),
                    status = validateInstall(manifest, pluginDir)
                )
                Log.i(TAG, "Found backend: ${manifest.id} v${manifest.version} at ${pluginDir.name}")
            }.onFailure { e ->
                Log.w(TAG, "Skipping ${pluginDir.name}: ${e.message}")
            }
        }

        _entries.value = found
        Log.i(TAG, "Registry scan complete: ${found.size} backends found")
    }

    fun register(manifest: BackendManifest, installPath: String) {
        val entry = InstalledBackendEntry(
            manifest = manifest,
            installPath = installPath,
            installedAt = System.currentTimeMillis(),
            status = InstallStatus.INSTALLED
        )
        _entries.value = _entries.value + (manifest.id to entry)
        Log.i(TAG, "Registered backend: ${manifest.id}")
    }

    fun unregister(backendId: String) {
        _entries.value = _entries.value - backendId
        Log.i(TAG, "Unregistered backend: $backendId")
    }

    fun getEntry(backendId: String): InstalledBackendEntry? = _entries.value[backendId]

    fun getManifests(): List<BackendManifest> = _entries.value.values.map { it.manifest }

    private fun validateInstall(manifest: BackendManifest, pluginDir: File): InstallStatus {
        // Check classes.dex exists
        if (!File(pluginDir, "classes.dex").exists()) return InstallStatus.CORRUPT

        // Check native libs exist for at least one supported ABI
        val hasNativeLibs = manifest.nativeLibs.all { libName ->
            manifest.abi.any { abi ->
                File(pluginDir, "lib/$abi/$libName").exists()
            } || File(pluginDir, "lib/$libName").exists()
        }
        if (!hasNativeLibs) return InstallStatus.CORRUPT

        return InstallStatus.INSTALLED
    }
}
