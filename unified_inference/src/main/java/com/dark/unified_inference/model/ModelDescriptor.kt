package com.dark.unified_inference.model

data class ModelDescriptor(
    val name: String,
    val format: ModelFormat,
    val source: ModelSource,
    val sizeBytes: Long? = null
)
