
//
// cpu_helper.h
// Lightweight runtime helpers for Android CPUs
// Supports big.LITTLE core detection for optimal thread pinning
//

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return the number of *physical* cores on the device.
 * (Counts unique core_id entries under /sys/devices/system/cpu/…)
 *
 * Falls back to android_getCpuCount() if /sys topology is absent.
 */
int count_physical_cores(void);

/**
 * Return the number of *performance* (big) cores on a big.LITTLE SoC.
 *
 * Reads cpuinfo_max_freq for each CPU and groups by frequency.
 * Cores whose max frequency >= 75% of the highest max frequency
 * are considered "performance" cores.
 *
 * This avoids scheduling inference work on efficiency (LITTLE) cores
 * which can be 3-5x slower and cause straggler-thread bottlenecks.
 *
 * Falls back to count_physical_cores() if frequency info is unavailable.
 */
int count_performance_cores(void);

/**
 * Return the optimal thread count for inference.
 *
 * Uses performance core count when big.LITTLE is detected,
 * otherwise falls back to physical core count.
 * Ensures at least 2 threads are returned.
 */
int get_optimal_thread_count(void);

#ifdef __cplusplus
}   // extern "C"
#endif
