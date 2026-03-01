package com.dark.backend_plugin_api.core

/**
 * Hardware observation data — provided by backend_manager's C++ layer.
 * Backends can use this to make decisions about model loading and compute.
 */
data class HardwareInfo(
    val totalRamBytes: Long,
    val availableRamBytes: Long,
    val cpuCoreCount: Int,
    val bigCoreCount: Int,
    val bigCoreMaxFreqKhz: Long,
    val littleCoreMaxFreqKhz: Long,
    val thermalState: ThermalState,
    val gpuThermalCelsius: Int,
    val abiList: List<String>
)

enum class ThermalState {
    NOMINAL,
    LIGHT,
    MODERATE,
    SEVERE,
    CRITICAL,
    UNKNOWN
}
