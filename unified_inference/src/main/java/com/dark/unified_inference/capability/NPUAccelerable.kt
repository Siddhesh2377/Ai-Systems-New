package com.dark.unified_inference.capability

interface NPUAccelerable {
    fun isNPUAvailable(): Boolean
    fun setAccelerationBackend(backend: AccelerationBackend): Boolean
}
