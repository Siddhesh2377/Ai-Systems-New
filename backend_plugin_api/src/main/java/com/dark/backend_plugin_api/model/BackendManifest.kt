package com.dark.backend_plugin_api.model

import com.dark.backend_plugin_api.core.Capability

/**
 * Parsed from manifest.json inside a backend plugin zip.
 * Describes everything the plugin manager needs to load and validate a backend.
 */
data class BackendManifest(
    val id: String,
    val name: String,
    val version: String,
    val apiVersion: Int,
    val capabilities: Set<Capability>,
    val entryClass: String,
    val nativeLibs: List<String>,
    val minSdk: Int,
    val abi: List<String>
)
