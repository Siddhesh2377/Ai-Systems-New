package com.dark.gguf_lib

import org.json.JSONArray
import org.json.JSONObject

/**
 * Reads the available compute backends + devices from ggml's runtime
 * registry and produces a [GpuProfile] list that downstream scheduling
 * (e.g. [VlmEncoder]) can route on.
 *
 * Probe is cheap (single JSON parse from already-cached registry data),
 * so callers may invoke it freely — but in practice profiles don't change
 * after process start, so cache once.
 */
object HardwareEngine {

    /**
     * Snapshot every backend + device the engine knows about, classified into
     * [GpuProfile] entries. CPU is always present; GPU entries appear only
     * when their backend (Vulkan, etc.) was registered at startup.
     */
    fun probe(): List<GpuProfile> {
        val raw = runCatching { GGUFNativeLib.nativeListBackendsJson() }.getOrNull() ?: return emptyList()
        val root = runCatching { JSONObject(raw) }.getOrNull() ?: return emptyList()

        val devicesJson = root.optJSONArray("devices") ?: JSONArray()
        val backends    = root.optJSONArray("backends")?.toNameList() ?: emptyList()

        val profiles = mutableListOf<GpuProfile>()
        for (i in 0 until devicesJson.length()) {
            val d = devicesJson.optJSONObject(i) ?: continue
            val name        = d.optString("name", "?")
            val description = d.optString("description", "")
            val typeStr     = d.optString("type", "?")
            val type = when (typeStr.lowercase()) {
                "cpu"   -> GpuDeviceType.CPU
                "gpu"   -> GpuDeviceType.GPU
                "igpu"  -> GpuDeviceType.IGPU
                "accel" -> GpuDeviceType.ACCEL
                else    -> GpuDeviceType.UNKNOWN
            }
            val combined = (description.takeIf { it.isNotBlank() } ?: name)
            profiles += GpuProfile(
                vendor = vendorFromName(combined),
                deviceName = combined,
                backendName = backends.getOrNull(i) ?: name,
                deviceType = type,
                totalMemoryBytes = d.optLong("memory_total", 0L),
                freeMemoryBytes  = d.optLong("memory_free",  0L),
                supportsAsync    = d.optBoolean("async",     false),
                supportsEvents   = d.optBoolean("events",    false),
                parallelComputeSlots = parallelSlotsFor(type, vendorFromName(combined)),
            )
        }
        return profiles
    }

    /** Convenience: first GPU/IGPU profile if any, else null. */
    fun firstGpu(): GpuProfile? = probe().firstOrNull { it.isGpu }

    /** Convenience: human-readable summary for debug screens. */
    fun summary(): String = buildString {
        val list = probe()
        if (list.isEmpty()) return "no backends registered"
        for (p in list) {
            append("• ${p.backendName} ${p.deviceName}")
            append(" [${p.deviceType.name.lowercase()}, ${p.vendor.name.lowercase()}]")
            if (p.totalMemoryBytes > 0) {
                append(" mem=${p.totalMemoryBytes / (1024 * 1024)} MiB")
            }
            append(" slots=${p.parallelComputeSlots}")
            if (p.supportsAsync) append(" async")
            if (p.supportsEvents) append(" events")
            append('\n')
        }
    }.trim()

    // ── Heuristics ─────────────────────────────────────────────────────────

    private fun JSONArray.toNameList(): List<String> =
        (0 until length()).mapNotNull { optJSONObject(it)?.optString("name", null) }

    /**
     * Vendor is parsed from device description, not from a vendor ID — the
     * ggml registry doesn't expose VkPhysicalDeviceProperties.vendorID
     * through dev_props. Description strings are stable enough in practice.
     */
    private fun vendorFromName(s: String): GpuVendor {
        val n = s.lowercase()
        return when {
            "adreno"     in n -> GpuVendor.QUALCOMM_ADRENO
            "qualcomm"   in n -> GpuVendor.QUALCOMM_ADRENO
            "mali"       in n -> GpuVendor.ARM_MALI
            "powervr"    in n -> GpuVendor.IMAGINATION_POWERVR
            "imagination" in n -> GpuVendor.IMAGINATION_POWERVR
            "apple"      in n -> GpuVendor.APPLE
            "intel"      in n -> GpuVendor.INTEL
            "nvidia"     in n || "geforce" in n || "rtx" in n || "gtx" in n -> GpuVendor.NVIDIA
            "amd"        in n || "radeon"  in n -> GpuVendor.AMD
            "swiftshader" in n || "llvmpipe" in n -> GpuVendor.SOFTWARE_RASTERIZER
            else              -> GpuVendor.UNKNOWN
        }
    }

    /**
     * Coarse parallelism estimate. Mobile GPUs almost always serialize at
     * the queue level — Adreno's vendor docs and Mali's documentation both
     * describe a single primary submission path. Desktop GPUs can have
     * separate compute / transfer / graphics queues, and discrete cards
     * with multiple SMs benefit more from concurrent jobs.
     *
     * This is purposefully conservative — the scheduler treats >1 as a
     * "you can try, but don't expect linear scaling" hint.
     */
    private fun parallelSlotsFor(type: GpuDeviceType, vendor: GpuVendor): Int = when {
        type == GpuDeviceType.CPU -> 1                    // CPU parallelism handled at thread level
        type == GpuDeviceType.IGPU -> 1                   // mobile UMA: single queue
        vendor == GpuVendor.QUALCOMM_ADRENO -> 1
        vendor == GpuVendor.ARM_MALI -> 1
        vendor == GpuVendor.IMAGINATION_POWERVR -> 1
        vendor == GpuVendor.NVIDIA -> 4                   // discrete: optimistic, can saturate with 4 streams
        vendor == GpuVendor.AMD -> 2
        vendor == GpuVendor.INTEL -> 2
        vendor == GpuVendor.APPLE -> 2
        else -> 1
    }
}
