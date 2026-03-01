package com.dark.backend_manager.hardware

import com.dark.backend_plugin_api.core.HardwareInfo
import com.dark.backend_plugin_api.core.HardwareObserver
import com.dark.backend_plugin_api.core.ThermalState

/**
 * JNI-backed hardware observer. Reads sysfs for CPU/GPU/thermal info,
 * /proc/meminfo for RAM, and CPU topology for big.LITTLE detection.
 */
class NativeHardwareObserver : HardwareObserver {

    companion object {
        init {
            System.loadLibrary("backend_manager")
        }
    }

    override fun getHardwareInfo(): HardwareInfo {
        return HardwareInfo(
            totalRamBytes = nativeGetTotalRam(),
            availableRamBytes = nativeGetAvailableRam(),
            cpuCoreCount = nativeGetCpuCoreCount(),
            bigCoreCount = nativeGetBigCoreCount(),
            bigCoreMaxFreqKhz = nativeGetBigCoreMaxFreqKhz(),
            littleCoreMaxFreqKhz = nativeGetLittleCoreMaxFreqKhz(),
            thermalState = ThermalState.entries.getOrElse(nativeGetThermalState()) { ThermalState.UNKNOWN },
            gpuThermalCelsius = nativeGetGpuThermalCelsius(),
            abiList = nativeGetAbiList().toList()
        )
    }

    override fun getAvailableRamBytes(): Long = nativeGetAvailableRam()

    override fun getThermalState(): ThermalState =
        ThermalState.entries.getOrElse(nativeGetThermalState()) { ThermalState.UNKNOWN }

    override fun getGpuThermalCelsius(): Int = nativeGetGpuThermalCelsius()

    override fun getRecommendedThreadCount(): Int {
        val bigCores = nativeGetBigCoreCount()
        val thermal = nativeGetThermalState()
        return when {
            thermal >= 3 -> maxOf(1, bigCores / 2)  // SEVERE+: halve threads
            thermal >= 2 -> maxOf(2, bigCores - 1)   // MODERATE: back off 1
            else -> bigCores                          // NOMINAL/LIGHT: all big cores
        }
    }

    // --- JNI declarations ---

    private external fun nativeGetTotalRam(): Long
    private external fun nativeGetAvailableRam(): Long
    private external fun nativeGetCpuCoreCount(): Int
    private external fun nativeGetBigCoreCount(): Int
    private external fun nativeGetBigCoreMaxFreqKhz(): Long
    private external fun nativeGetLittleCoreMaxFreqKhz(): Long
    private external fun nativeGetThermalState(): Int
    private external fun nativeGetGpuThermalCelsius(): Int
    private external fun nativeGetAbiList(): Array<String>
}
