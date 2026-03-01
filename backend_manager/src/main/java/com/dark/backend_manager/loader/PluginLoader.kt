package com.dark.backend_manager.loader

import android.util.Log
import com.dark.backend_plugin_api.core.BackendPlugin
import com.dark.backend_plugin_api.core.PluginManager
import com.dark.backend_plugin_api.model.BackendManifest
import dalvik.system.DexClassLoader
import java.io.File

/**
 * Loads a backend plugin from its install directory.
 * Handles DexClassLoader for Kotlin/Java code and System.load() for native libs.
 *
 * Expected directory layout:
 * plugins/{backendId}/
 *   ├── manifest.json
 *   ├── classes.dex
 *   └── lib/arm64-v8a/*.so
 */
class PluginLoader(private val cacheDir: File) {

    companion object {
        private const val TAG = "PluginLoader"
    }

    /**
     * Load a backend plugin from disk.
     * @param manifest parsed manifest of the backend
     * @param installPath absolute path to the plugin directory
     * @return instantiated BackendPlugin, or failure
     */
    fun load(manifest: BackendManifest, installPath: String): Result<BackendPlugin> {
        return runCatching {
            // Validate API version — exact match required
            require(manifest.apiVersion == PluginManager.API_VERSION) {
                "API version mismatch: backend=${manifest.apiVersion}, " +
                        "manager=${PluginManager.API_VERSION}. Re-download this backend."
            }

            val pluginDir = File(installPath)
            require(pluginDir.exists()) { "Plugin directory not found: $installPath" }

            // Load native libs first (order matters — dependencies before dependents)
            loadNativeLibs(manifest.nativeLibs, pluginDir)

            // Load DEX
            val dexPath = File(pluginDir, "classes.dex")
            require(dexPath.exists()) { "classes.dex not found in $installPath" }

            val optimizedDir = File(cacheDir, "dex_opt_${manifest.id}")
            optimizedDir.mkdirs()

            val classLoader = DexClassLoader(
                dexPath.absolutePath,
                optimizedDir.absolutePath,
                File(pluginDir, "lib").absolutePath, // native lib search path
                javaClass.classLoader
            )

            // Instantiate the entry class
            val pluginClass = classLoader.loadClass(manifest.entryClass)
            val plugin = pluginClass.getDeclaredConstructor().newInstance()

            require(plugin is BackendPlugin) {
                "${manifest.entryClass} does not implement BackendPlugin"
            }

            Log.i(TAG, "Loaded backend: ${manifest.id} v${manifest.version} " +
                    "[${manifest.capabilities.joinToString()}]")

            plugin
        }
    }

    /**
     * Unload is best-effort on Android — we can't truly unload .so files.
     * The ClassLoader becomes GC-eligible when all references are dropped.
     */
    fun unload(backendId: String) {
        // Clean up optimized DEX cache
        val optimizedDir = File(cacheDir, "dex_opt_$backendId")
        if (optimizedDir.exists()) {
            optimizedDir.deleteRecursively()
        }
        Log.i(TAG, "Cleaned up loader cache for: $backendId")
    }

    private fun loadNativeLibs(libs: List<String>, pluginDir: File) {
        val libDir = File(pluginDir, "lib/${android.os.Build.SUPPORTED_ABIS[0]}")
        if (!libDir.exists()) {
            // Fallback: libs might be directly in lib/
            val fallback = File(pluginDir, "lib")
            for (libName in libs) {
                val libFile = File(fallback, libName)
                if (libFile.exists()) {
                    System.load(libFile.absolutePath)
                    Log.d(TAG, "Loaded native lib: ${libFile.absolutePath}")
                }
            }
            return
        }

        for (libName in libs) {
            val libFile = File(libDir, libName)
            require(libFile.exists()) { "Native lib not found: ${libFile.absolutePath}" }
            System.load(libFile.absolutePath)
            Log.d(TAG, "Loaded native lib: ${libFile.absolutePath}")
        }
    }
}
