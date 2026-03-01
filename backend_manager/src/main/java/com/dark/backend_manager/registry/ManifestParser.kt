package com.dark.backend_manager.registry

import com.dark.backend_plugin_api.core.Capability
import com.dark.backend_plugin_api.model.BackendManifest
import org.json.JSONObject
import java.io.File

/**
 * Parses manifest.json from a backend plugin directory.
 */
object ManifestParser {

    fun parse(pluginDir: File): Result<BackendManifest> = runCatching {
        val manifestFile = File(pluginDir, "manifest.json")
        require(manifestFile.exists()) { "manifest.json not found in ${pluginDir.absolutePath}" }

        val json = JSONObject(manifestFile.readText())

        val capabilitiesArray = json.getJSONArray("capabilities")
        val capabilities = (0 until capabilitiesArray.length())
            .map { Capability.valueOf(capabilitiesArray.getString(it)) }
            .toSet()

        val nativeLibsArray = json.getJSONArray("nativeLibs")
        val nativeLibs = (0 until nativeLibsArray.length())
            .map { nativeLibsArray.getString(it) }

        val abiArray = json.getJSONArray("abi")
        val abi = (0 until abiArray.length())
            .map { abiArray.getString(it) }

        BackendManifest(
            id = json.getString("id"),
            name = json.getString("name"),
            version = json.getString("version"),
            apiVersion = json.getInt("apiVersion"),
            capabilities = capabilities,
            entryClass = json.getString("entryClass"),
            nativeLibs = nativeLibs,
            minSdk = json.getInt("minSdk"),
            abi = abi
        )
    }

    fun toJson(manifest: BackendManifest): String {
        return JSONObject().apply {
            put("id", manifest.id)
            put("name", manifest.name)
            put("version", manifest.version)
            put("apiVersion", manifest.apiVersion)
            put("capabilities", org.json.JSONArray(manifest.capabilities.map { it.name }))
            put("entryClass", manifest.entryClass)
            put("nativeLibs", org.json.JSONArray(manifest.nativeLibs))
            put("minSdk", manifest.minSdk)
            put("abi", org.json.JSONArray(manifest.abi))
        }.toString(2)
    }
}
