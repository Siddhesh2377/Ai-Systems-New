package com.dark.backend_plugin_api.model

import com.dark.backend_plugin_api.core.Capability

/**
 * Describes a model that a backend can load.
 * Maps to ToolNeuron's existing Model entity — the [extras] map carries
 * backend-specific loading params (context size, quant type, GPU layers, etc.).
 */
data class ModelMetadata(
    val id: String,
    val name: String,
    val path: String,
    val requiredBackend: String,
    val capabilities: Set<Capability>,
    val fileSizeBytes: Long,
    val extras: Map<String, String> = emptyMap()
)
