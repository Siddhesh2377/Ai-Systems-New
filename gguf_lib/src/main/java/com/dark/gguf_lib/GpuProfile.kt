package com.dark.gguf_lib

/**
 * Coarse GPU/device profile that the [VlmEncoder] scheduler reads to pick
 * a dispatch strategy at runtime. Filled in by [HardwareEngine.probe].
 *
 * The profile is intentionally pessimistic about parallelism — most mobile
 * GPUs have a single graphics queue and serialize submitted compute. This
 * data class describes what the device *can* do, not what we *will* do.
 */
data class GpuProfile(
    /** Coarse vendor bucket. UNKNOWN for non-GPU devices and unrecognised names. */
    val vendor: GpuVendor,
    /** Raw device description from the backend (e.g. "Adreno (TM) 810", "Mali-G610"). */
    val deviceName: String,
    /** Stable backend name (e.g. "Vulkan0", "CPU"). Used for routing decisions. */
    val backendName: String,
    /** ggml device type. */
    val deviceType: GpuDeviceType,
    /** Total device memory in bytes. For UMA GPUs this is the slice the driver reports — not always equal to system RAM. */
    val totalMemoryBytes: Long,
    /** Free memory at probe time (driver estimate). */
    val freeMemoryBytes: Long,
    /** Whether the backend supports async queue submission. */
    val supportsAsync: Boolean,
    /** Whether the backend supports events (cross-queue sync). */
    val supportsEvents: Boolean,
    /**
     * Coarse compute parallelism estimate — how many independent work
     * streams the device can usefully service in parallel. Mobile GPUs
     * (Adreno, Mali) almost always report 1 here; desktop GPUs may report
     * higher. This is an approximation, not a hard guarantee.
     */
    val parallelComputeSlots: Int,
) {
    val isGpu: Boolean get() = deviceType == GpuDeviceType.GPU || deviceType == GpuDeviceType.IGPU

    /** True for unified-memory mobile/integrated GPUs. */
    val isUma: Boolean get() = deviceType == GpuDeviceType.IGPU
}

enum class GpuVendor {
    QUALCOMM_ADRENO,
    ARM_MALI,
    IMAGINATION_POWERVR,
    APPLE,
    INTEL,
    NVIDIA,
    AMD,
    SOFTWARE_RASTERIZER,   // SwiftShader, llvmpipe — still works, but slow
    UNKNOWN,
}

enum class GpuDeviceType { CPU, GPU, IGPU, ACCEL, UNKNOWN }
