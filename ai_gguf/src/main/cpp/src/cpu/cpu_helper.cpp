// cpu_helper.cpp
// big.LITTLE aware CPU detection for Android
#include "cpu_helper.h"

#if defined(__ANDROID__)
#include "cpu-features.h"
#include <dirent.h>
#include <cctype>
#include <cstdio>
#include <set>
#include <map>
#include <string>
#include <sys/types.h>
#include <algorithm>

int count_physical_cores(void) {
    std::set<int> coreIds;

    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) {
        return android_getCpuCount();   // old kernels / no topology
    }

    struct dirent *dent;
    while ((dent = readdir(dir)) != nullptr) {
        if (strncmp(dent->d_name, "cpu", 3) != 0 || !std::isdigit(dent->d_name[3]))
            continue;          // ignore non-cpu directories

        std::string path = "/sys/devices/system/cpu/";
        path += dent->d_name;
        path += "/topology/core_id";

        FILE *f = fopen(path.c_str(), "r");
        if (!f) continue;

        int id = -1;
        if (fscanf(f, "%d", &id) == 1 && id >= 0)
            coreIds.insert(id);
        fclose(f);
    }
    closedir(dir);

    return coreIds.empty()
           ? android_getCpuCount()
           : static_cast<int>(coreIds.size());
}

int count_performance_cores(void) {
    // Read max frequency for each online CPU
    // Group by frequency tier, return count of "big" cores
    std::map<int, int> freq_count;  // max_freq_khz -> core_count
    int highest_freq = 0;

    DIR *dir = opendir("/sys/devices/system/cpu");
    if (!dir) {
        return count_physical_cores();
    }

    struct dirent *dent;
    while ((dent = readdir(dir)) != nullptr) {
        if (strncmp(dent->d_name, "cpu", 3) != 0 || !std::isdigit(dent->d_name[3]))
            continue;

        std::string path = "/sys/devices/system/cpu/";
        path += dent->d_name;
        path += "/cpufreq/cpuinfo_max_freq";

        FILE *f = fopen(path.c_str(), "r");
        if (!f) continue;

        int freq = 0;
        if (fscanf(f, "%d", &freq) == 1 && freq > 0) {
            freq_count[freq]++;
            if (freq > highest_freq) highest_freq = freq;
        }
        fclose(f);
    }
    closedir(dir);

    if (highest_freq == 0 || freq_count.empty()) {
        return count_physical_cores();
    }

    // Only one frequency tier → not a big.LITTLE SoC (or all same clock)
    if (freq_count.size() == 1) {
        return freq_count.begin()->second;
    }

    // Performance cores = all cores with max_freq >= 75% of highest
    // This captures prime + big cores while excluding little/efficiency cores.
    //
    // Examples:
    //   SD 8 Gen 3: X4@3.3GHz + A720@3.15GHz + A720@2.96GHz + A520@2.27GHz
    //     threshold = 3300000 * 0.75 = 2475000 → captures 1+3+2 = 6 perf cores
    //   SD 7s Gen 3: A715@2.5GHz + A715@2.4GHz + A510@1.8GHz
    //     threshold = 2500000 * 0.75 = 1875000 → captures 1+3 = 4 perf cores
    int threshold = static_cast<int>(highest_freq * 0.75);
    int perf_cores = 0;

    for (const auto& [freq, count] : freq_count) {
        if (freq >= threshold) {
            perf_cores += count;
        }
    }

    // Sanity: return at least 1
    return (perf_cores > 0) ? perf_cores : count_physical_cores();
}

int get_optimal_thread_count(void) {
    int perf = count_performance_cores();
    int phys = count_physical_cores();

    // Use performance cores if big.LITTLE detected (perf < total physical)
    // Otherwise use all physical cores
    int threads = (perf < phys && perf >= 2) ? perf : phys;

    // Ensure at least 2 threads
    return std::max(threads, 2);
}

#else
// ─────── non-Android fallback ───────
int count_physical_cores(void) { return 4; }
int count_performance_cores(void) { return 4; }
int get_optimal_thread_count(void) { return 4; }
#endif
