#pragma once

/**
 * CPU affinity pinning for ARM big.LITTLE architecture.
 *
 * Reads sysfs to identify performance cores by max frequency,
 * then pins current thread group to those cores via sched_setaffinity.
 * Prevents Android scheduler from migrating inference threads to
 * A55 efficiency cores during generation.
 *
 * Ported from gguf_lib (gguf_lib.cpp:210-252).
 */

#include "sd_logger.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sched.h>
#include <thread>
#include <utility>
#include <vector>

namespace sd_cpu {

// Get performance core IDs sorted by max frequency (highest first).
inline std::vector<int> get_perf_core_ids() {
    int n_total = static_cast<int>(std::thread::hardware_concurrency());
    if (n_total <= 0) return {};

    std::vector<std::pair<long, int>> freq_core; // (freq, core_id)
    for (int i = 0; i < n_total; i++) {
        char path[128];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        long freq = 0;
        FILE* f = fopen(path, "r");
        if (f) { fscanf(f, "%ld", &freq); fclose(f); }
        freq_core.push_back({freq, i});
    }

    std::sort(freq_core.begin(), freq_core.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    // Take top half as "performance" cores
    long median = freq_core[n_total / 2].first;
    std::vector<int> perf_ids;
    for (auto& [freq, id] : freq_core) {
        if (freq >= median) perf_ids.push_back(id);
    }
    return perf_ids;
}

// Pin current process threads to performance cores.
inline void pin_to_perf_cores() {
    auto perf_ids = get_perf_core_ids();
    if (perf_ids.empty()) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int id : perf_ids) CPU_SET(id, &set);
    if (sched_setaffinity(0, sizeof(set), &set) == 0) {
        SD_LOG_INFO("[JNI] Pinned to %zu performance cores", perf_ids.size());
    } else {
        SD_LOG_WARN("[JNI] sched_setaffinity failed: %s", strerror(errno));
    }
}

} // namespace sd_cpu
