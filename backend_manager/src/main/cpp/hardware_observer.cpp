#include <jni.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/sysinfo.h>
#include <android/log.h>

#define TAG "HardwareObserver"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

// JNI class: com.dark.backend_manager.hardware.NativeHardwareObserver
// Note: "backend_manager" → "backend_1manager" in JNI mangling (underscore in package name)
#define JNI_METHOD(ret, name) \
    extern "C" JNIEXPORT ret JNICALL \
    Java_com_dark_backend_1manager_hardware_NativeHardwareObserver_##name

// ============================================================================
// RAM
// ============================================================================

static long read_meminfo_field(const char* field) {
    FILE* fp = fopen("/proc/meminfo", "r");
    if (!fp) return -1;

    char line[256];
    long value = -1;
    size_t field_len = strlen(field);

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, field, field_len) == 0) {
            // Format: "FieldName:    12345 kB"
            char* p = line + field_len;
            while (*p == ' ' || *p == ':') p++;
            value = strtol(p, nullptr, 10) * 1024L; // kB -> bytes
            break;
        }
    }
    fclose(fp);
    return value;
}

JNI_METHOD(jlong, nativeGetTotalRam)(JNIEnv*, jobject) {
    long total = read_meminfo_field("MemTotal");
    if (total > 0) return total;

    // Fallback: sysinfo
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (jlong)info.totalram * info.mem_unit;
    }
    return -1;
}

JNI_METHOD(jlong, nativeGetAvailableRam)(JNIEnv*, jobject) {
    // MemAvailable is the best indicator (includes reclaimable caches)
    long available = read_meminfo_field("MemAvailable");
    if (available > 0) return available;

    // Fallback: MemFree + Buffers + Cached
    long free_mem = read_meminfo_field("MemFree");
    long buffers = read_meminfo_field("Buffers");
    long cached = read_meminfo_field("Cached");
    if (free_mem > 0) {
        return free_mem + std::max(0L, buffers) + std::max(0L, cached);
    }
    return -1;
}

// ============================================================================
// CPU topology
// ============================================================================

static int get_cpu_count() {
    int count = 0;
    char path[128];
    // Count cpu directories: /sys/devices/system/cpu/cpu0, cpu1, ...
    for (int i = 0; i < 16; i++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", i);
        FILE* fp = fopen((std::string(path) + "/online").c_str(), "r");
        if (!fp) {
            // Try just checking directory exists
            DIR* dir = opendir(path);
            if (dir) {
                closedir(dir);
                count++;
            } else {
                break;
            }
        } else {
            fclose(fp);
            count++;
        }
    }
    return count > 0 ? count : 4; // default 4
}

static long read_cpu_max_freq(int cpu_id) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu_id);
    FILE* fp = fopen(path, "r");
    if (!fp) return 0;

    long freq = 0;
    if (fscanf(fp, "%ld", &freq) != 1) freq = 0;
    fclose(fp);
    return freq; // in kHz
}

struct CpuTopology {
    int total_cores;
    int big_cores;
    long big_max_khz;
    long little_max_khz;
};

static CpuTopology get_cpu_topology() {
    CpuTopology topo = {};
    topo.total_cores = get_cpu_count();

    std::vector<long> freqs;
    for (int i = 0; i < topo.total_cores; i++) {
        long f = read_cpu_max_freq(i);
        freqs.push_back(f);
    }

    if (freqs.empty()) {
        topo.big_cores = topo.total_cores;
        return topo;
    }

    long max_freq = *std::max_element(freqs.begin(), freqs.end());
    long min_freq = *std::min_element(freqs.begin(), freqs.end());

    // If all cores have the same frequency, treat all as big
    if (max_freq == min_freq || min_freq == 0) {
        topo.big_cores = topo.total_cores;
        topo.big_max_khz = max_freq;
        topo.little_max_khz = max_freq;
        return topo;
    }

    // Threshold: cores above 70% of max freq are "big"
    long threshold = (long)(max_freq * 0.7);
    int big_count = 0;
    for (long f : freqs) {
        if (f >= threshold) big_count++;
    }

    topo.big_cores = big_count > 0 ? big_count : topo.total_cores;
    topo.big_max_khz = max_freq;
    topo.little_max_khz = min_freq;

    LOGI("CPU topology: %d cores (%d big @ %ld kHz, %d little @ %ld kHz)",
         topo.total_cores, topo.big_cores, topo.big_max_khz,
         topo.total_cores - topo.big_cores, topo.little_max_khz);

    return topo;
}

// Cache topology — compute once
static CpuTopology cached_topology = {};
static bool topology_cached = false;

static const CpuTopology& get_cached_topology() {
    if (!topology_cached) {
        cached_topology = get_cpu_topology();
        topology_cached = true;
    }
    return cached_topology;
}

JNI_METHOD(jint, nativeGetCpuCoreCount)(JNIEnv*, jobject) {
    return get_cached_topology().total_cores;
}

JNI_METHOD(jint, nativeGetBigCoreCount)(JNIEnv*, jobject) {
    return get_cached_topology().big_cores;
}

JNI_METHOD(jlong, nativeGetBigCoreMaxFreqKhz)(JNIEnv*, jobject) {
    return get_cached_topology().big_max_khz;
}

JNI_METHOD(jlong, nativeGetLittleCoreMaxFreqKhz)(JNIEnv*, jobject) {
    return get_cached_topology().little_max_khz;
}

// ============================================================================
// Thermal
// ============================================================================

// Thermal zone patterns for GPU across vendors
static const char* gpu_thermal_patterns[] = {
    "gpu",
    "GPU",
    "adreno",
    "mali",
    "sgpu",
    "pvr",
    nullptr
};

static int read_thermal_zone_temp(const char* type_match) {
    char path[128], type_buf[64], temp_buf[32];

    for (int i = 0; i < 30; i++) {
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/type", i);
        FILE* fp = fopen(path, "r");
        if (!fp) continue;

        if (fgets(type_buf, sizeof(type_buf), fp)) {
            // Strip newline
            char* nl = strchr(type_buf, '\n');
            if (nl) *nl = '\0';

            if (strstr(type_buf, type_match)) {
                fclose(fp);
                snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", i);
                fp = fopen(path, "r");
                if (fp && fgets(temp_buf, sizeof(temp_buf), fp)) {
                    fclose(fp);
                    int raw = atoi(temp_buf);
                    // Some zones report millidegrees, some report degrees
                    return raw > 1000 ? raw / 1000 : raw;
                }
                if (fp) fclose(fp);
                return -1;
            }
        }
        fclose(fp);
    }
    return -1;
}

JNI_METHOD(jint, nativeGetGpuThermalCelsius)(JNIEnv*, jobject) {
    for (int i = 0; gpu_thermal_patterns[i] != nullptr; i++) {
        int temp = read_thermal_zone_temp(gpu_thermal_patterns[i]);
        if (temp >= 0) return temp;
    }
    return -1;
}

JNI_METHOD(jint, nativeGetThermalState)(JNIEnv*, jobject) {
    // Read GPU temp and map to thermal state enum
    // 0=NOMINAL, 1=LIGHT, 2=MODERATE, 3=SEVERE, 4=CRITICAL, 5=UNKNOWN
    int gpu_temp = -1;
    for (int i = 0; gpu_thermal_patterns[i] != nullptr; i++) {
        gpu_temp = read_thermal_zone_temp(gpu_thermal_patterns[i]);
        if (gpu_temp >= 0) break;
    }

    if (gpu_temp < 0) {
        // Try CPU temperature as fallback
        gpu_temp = read_thermal_zone_temp("cpu");
        if (gpu_temp < 0) gpu_temp = read_thermal_zone_temp("soc");
        if (gpu_temp < 0) return 5; // UNKNOWN
    }

    if (gpu_temp < 50) return 0;  // NOMINAL
    if (gpu_temp < 65) return 1;  // LIGHT
    if (gpu_temp < 80) return 2;  // MODERATE
    if (gpu_temp < 95) return 3;  // SEVERE
    return 4;                      // CRITICAL
}

// ============================================================================
// ABI list
// ============================================================================

JNI_METHOD(jobjectArray, nativeGetAbiList)(JNIEnv* env, jobject) {
    // Read from system property
    char value[256] = {};
    __system_property_get("ro.product.cpu.abilist", value);

    std::vector<std::string> abis;
    std::stringstream ss(value);
    std::string abi;
    while (std::getline(ss, abi, ',')) {
        if (!abi.empty()) abis.push_back(abi);
    }

    // Fallback
    if (abis.empty()) {
        abis.push_back("arm64-v8a");
    }

    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray((jint)abis.size(), stringClass, nullptr);
    for (int i = 0; i < (int)abis.size(); i++) {
        env->SetObjectArrayElement(result, i, env->NewStringUTF(abis[i].c_str()));
    }
    return result;
}
