package com.dark.backend_plugin_api.core

/**
 * Interface for querying device hardware state.
 * Implemented by backend_manager (C++ JNI), consumed by plugins and the manager itself.
 */
interface HardwareObserver {

    /** Full hardware snapshot */
    fun getHardwareInfo(): HardwareInfo

    /** Quick RAM check — available bytes right now */
    fun getAvailableRamBytes(): Long

    /** Current thermal state */
    fun getThermalState(): ThermalState

    /** GPU temperature in celsius, -1 if unavailable */
    fun getGpuThermalCelsius(): Int

    /** Recommended thread count for compute (based on big core count and thermal) */
    fun getRecommendedThreadCount(): Int
}
