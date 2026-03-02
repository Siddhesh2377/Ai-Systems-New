/**
 * JNI entry points for ai_sd Stable Diffusion module.
 *
 * Pattern follows ai_gguf/src/main/cpp/src/ai_gguf.cpp:
 * - Global state singleton with mutex protection
 * - Atomic stop flag for cancellation
 * - Thread-local JNI callback caching
 *
 * JNI naming: Java_com_dark_ai_1sd_SDNativeLib_<method>
 * (the _1 encodes the underscore in "ai_sd")
 */

#include "state/diffusion_state.h"
#include "upscaler/upscaler.h"
#include "pipeline/pipeline_globals.h"
#include "utils/cpu_affinity.h"
#include "utils/jni_utils.h"
#include "utils/sd_logger.h"

#include <jni.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <mutex>
#include <string>
#include <shared_mutex>
#include <atomic>
#include <cassert>
#include <pthread.h>

// Global state
static DiffusionState g_sd_state;
// Single shared_mutex: load/release take exclusive lock, generate takes shared lock.
// This prevents release-during-generate race (Bug 5 fix).
static std::shared_mutex g_sd_mtx;
static std::atomic<bool> g_sd_stop{false};

// Helper: convert jstring to std::string
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (!chars) return "";
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// RAII guard for JNI byte array elements — prevents leak on exception
struct JniByteArrayGuard {
    JNIEnv* env;
    jbyteArray array;
    jbyte* ptr;
    JniByteArrayGuard(JNIEnv* e, jbyteArray a, jbyte* p) : env(e), array(a), ptr(p) {}
    ~JniByteArrayGuard() { if (ptr) env->ReleaseByteArrayElements(array, ptr, JNI_ABORT); }
    JniByteArrayGuard(const JniByteArrayGuard&) = delete;
    JniByteArrayGuard& operator=(const JniByteArrayGuard&) = delete;
};

extern "C" {

/**
 * Initialize QNN runtime environment.
 * Sets ADSP_LIBRARY_PATH for QNN's internal DSP library resolution.
 * Must be called once before any model loading.
 *
 * @param qnnLibDir Directory containing QNN .so libraries
 * @return true on success
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeInitRuntime(
        JNIEnv* env, jobject /* thiz */, jstring qnnLibDir) {

    std::string libDir = jstring_to_string(env, qnnLibDir);
    if (libDir.empty()) {
        SD_LOG_ERROR("nativeInitRuntime: qnnLibDir is empty");
        return JNI_FALSE;
    }

    SD_LOG_INFO("Initializing QNN runtime with lib dir: %s", libDir.c_str());

    // Set ADSP_LIBRARY_PATH for QNN's internal DSP skel library resolution
    // Same approach as llama.cpp ggml-qnn backend (set_qnn_lib_search_path)
    std::string adsp_path = libDir +
        ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;"
        "/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
    setenv("ADSP_LIBRARY_PATH", adsp_path.c_str(), 1);

    SD_LOG_INFO("ADSP_LIBRARY_PATH set to: %s", adsp_path.c_str());
    return JNI_TRUE;
}

/**
 * Load Stable Diffusion model components.
 *
 * @param clipPath Path to CLIP model (.bin or .mnn)
 * @param unetPath Path to UNet model (.bin or .mnn)
 * @param vaeDecoderPath Path to VAE decoder model
 * @param vaeEncoderPath Path to VAE encoder model (nullable for txt2img only)
 * @param tokenizerPath Path to tokenizer.json
 * @param safetyCheckerPath Path to safety checker .mnn (nullable)
 * @param patchPath Path to resolution patch file (nullable)
 * @param modelDir Model directory root
 * @param qnnBackendPath Path to libQnnHtp.so
 * @param qnnSystemLibPath Path to libQnnSystem.so
 * @param textEmbeddingSize CLIP embedding dimension (768 or 1024)
 * @param runOnCpu Use CPU-only inference (MNN)
 * @param useCpuClip Use MNN for CLIP even with QNN UNet
 * @param isPony Pony v5.5 model (v_prediction)
 * @param useSafetyChecker Enable NSFW filter
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadModel(
        JNIEnv* env, jobject /* thiz */,
        jstring clipPath, jstring unetPath, jstring vaeDecoderPath,
        jstring vaeEncoderPath, jstring tokenizerPath,
        jstring safetyCheckerPath, jstring patchPath, jstring modelDir,
        jstring qnnBackendPath, jstring qnnSystemLibPath,
        jint textEmbeddingSize, jboolean runOnCpu, jboolean useCpuClip,
        jboolean isPony, jboolean useSafetyChecker) {

    std::unique_lock lock(g_sd_mtx);

    SDModelConfig config;
    config.clipPath = jstring_to_string(env, clipPath);
    config.unetPath = jstring_to_string(env, unetPath);
    config.vaeDecoderPath = jstring_to_string(env, vaeDecoderPath);
    config.vaeEncoderPath = jstring_to_string(env, vaeEncoderPath);
    config.tokenizerPath = jstring_to_string(env, tokenizerPath);
    config.safetyCheckerPath = jstring_to_string(env, safetyCheckerPath);
    config.patchPath = jstring_to_string(env, patchPath);
    config.modelDir = jstring_to_string(env, modelDir);
    config.qnnBackendPath = jstring_to_string(env, qnnBackendPath);
    config.qnnSystemLibPath = jstring_to_string(env, qnnSystemLibPath);
    config.textEmbeddingSize = textEmbeddingSize;
    config.runOnCpu = runOnCpu;
    config.useCpuClip = useCpuClip;
    config.isPony = isPony;
    config.useSafetyChecker = useSafetyChecker;

    SD_LOG_INFO("Loading model: clip=%s, unet=%s, cpu=%d",
                config.clipPath.c_str(), config.unetPath.c_str(), (int)runOnCpu);

    return g_sd_state.load_models(config) ? JNI_TRUE : JNI_FALSE;
}

/**
 * Run image generation.
 *
 * @param prompt Positive prompt text
 * @param negativePrompt Negative prompt text
 * @param steps Number of diffusion steps
 * @param cfgScale Classifier-free guidance scale
 * @param seed Random seed (0 for random)
 * @param width Output image width
 * @param height Output image height
 * @param scheduler Scheduler type ("dpm" or "euler_a")
 * @param useOpenCL Use OpenCL for MNN backend
 * @param inputImage Base64 RGB input image for img2img (nullable)
 * @param mask Base64 mask for inpainting (nullable)
 * @param denoiseStrength Denoising strength for img2img
 * @param showProcess Show intermediate diffusion images
 * @param showStride Show every N-th intermediate image
 * @param callback SDCallback instance for progress/completion
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeGenerate(
        JNIEnv* env, jobject /* thiz */,
        jstring prompt, jstring negativePrompt,
        jint steps, jfloat cfgScale, jlong seed,
        jint width, jint height, jstring scheduler,
        jboolean useOpenCL, jbyteArray inputImage, jbyteArray mask,
        jfloat denoiseStrength, jboolean showProcess, jint showStride,
        jobject callback) {

    std::shared_lock lock(g_sd_mtx);

    // Pin inference threads to performance cores (A78, not A55)
    sd_cpu::pin_to_perf_cores();

    if (!g_sd_state.is_ready()) {
        sd_jni::on_error(env, callback, "Model not loaded");
        return JNI_FALSE;
    }

    g_sd_stop.store(false);

    SDGenerateParams params;
    params.prompt = jstring_to_string(env, prompt);
    params.negativePrompt = jstring_to_string(env, negativePrompt);
    params.steps = steps;
    params.cfgScale = cfgScale;
    params.seed = static_cast<unsigned int>(seed);
    params.width = width;
    params.height = height;
    params.scheduler = jstring_to_string(env, scheduler);
    params.useOpenCL = useOpenCL;
    params.denoiseStrength = denoiseStrength;
    params.showDiffusionProcess = showProcess;
    params.showDiffusionStride = showStride;

    // Handle img2img input
    if (inputImage) {
        jsize imgLen = env->GetArrayLength(inputImage);
        if (imgLen > 0) {
            params.isImg2Img = true;
            jbyte* imgBytes = env->GetByteArrayElements(inputImage, nullptr);
            JniByteArrayGuard imgGuard(env, inputImage, imgBytes);
            // Convert from uint8 RGB to float NCHW [-1, 1]
            int pixelCount = width * height;
            params.inputImage.resize(3 * pixelCount);
            for (int i = 0; i < pixelCount; i++) {
                params.inputImage[0 * pixelCount + i] = (static_cast<uint8_t>(imgBytes[i * 3])     / 127.5f) - 1.0f;
                params.inputImage[1 * pixelCount + i] = (static_cast<uint8_t>(imgBytes[i * 3 + 1]) / 127.5f) - 1.0f;
                params.inputImage[2 * pixelCount + i] = (static_cast<uint8_t>(imgBytes[i * 3 + 2]) / 127.5f) - 1.0f;
            }
        }
    }

    // Handle mask input
    if (mask) {
        jsize maskLen = env->GetArrayLength(mask);
        if (maskLen > 0) {
            params.hasMask = true;
            // Mask processing is handled inside the pipeline
            // (same as xororz's server handler)
            jbyte* maskBytes = env->GetByteArrayElements(mask, nullptr);
            JniByteArrayGuard maskGuard(env, mask, maskBytes);
            int sampleW = width / 8;
            int sampleH = height / 8;

            // Latent-space mask (4 channels)
            params.mask.resize(4 * sampleW * sampleH);
            for (int y = 0; y < sampleH; y++) {
                for (int x = 0; x < sampleW; x++) {
                    // Average RGB to grayscale, normalize to [0,1]
                    int srcIdx = (y * sampleW + x) * 3;
                    float val = (static_cast<uint8_t>(maskBytes[srcIdx]) +
                                 static_cast<uint8_t>(maskBytes[srcIdx + 1]) +
                                 static_cast<uint8_t>(maskBytes[srcIdx + 2])) / (3.0f * 255.0f);
                    for (int c = 0; c < 4; c++) {
                        params.mask[c * sampleW * sampleH + y * sampleW + x] = val;
                    }
                }
            }

            // Full-resolution mask (3 channels)
            params.maskFull.resize(3 * width * height);
        }
    }

    SD_LOG_INFO("Starting generation: prompt='%s', steps=%d, size=%dx%d, seed=%u",
                params.prompt.c_str(), params.steps, params.width, params.height, params.seed);

    // Create JNI-safe progress callback
    // The callback captures env and callback jobject — must be called on same thread
    auto callerThread = pthread_self();
    auto progressCb = [env, callback, callerThread](int step, int totalSteps,
                                       const uint8_t* imageData, int imageDataLen,
                                       int imgWidth, int imgHeight) {
        assert(pthread_equal(pthread_self(), callerThread) &&
               "JNI env used on wrong thread — callback must run on JNI caller thread");
        if (imageData && imageDataLen > 0) {
            sd_jni::on_image_progress(env, callback, step, totalSteps,
                                      imageData, imageDataLen, imgWidth, imgHeight);
        } else {
            sd_jni::on_progress(env, callback, step, totalSteps);
        }
    };

    try {
        SDGenerationResult result = g_sd_state.generate(params, progressCb, g_sd_stop);

        if (g_sd_stop.load()) {
            sd_jni::on_error(env, callback, "Generation cancelled");
            return JNI_FALSE;
        }

        if (result.imageData.empty()) {
            sd_jni::on_error(env, callback, "Generation produced no image data");
            return JNI_FALSE;
        }

        sd_jni::on_complete(env, callback, result.imageData.data(),
                            static_cast<int>(result.imageData.size()),
                            result.width, result.height,
                            static_cast<long>(result.seed),
                            result.generationTimeMs);
        return JNI_TRUE;

    } catch (const std::exception& e) {
        SD_LOG_ERROR("Generation failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

/**
 * Stop ongoing generation.
 * Sets atomic flag that is checked each diffusion step.
 */
JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeStopGeneration(
        JNIEnv* /* env */, jobject /* thiz */) {
    g_sd_stop.store(true);
    SD_LOG_INFO("Stop generation requested");
}

/**
 * Release all model resources.
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeRelease(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::unique_lock lock(g_sd_mtx);
    g_sd_state.release();
    sd_jni::reset_cache();
    SD_LOG_INFO("Models released");
    return JNI_TRUE;
}

/**
 * Get model info as JSON string.
 */
JNIEXPORT jstring JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeGetModelInfo(
        JNIEnv* env, jobject /* thiz */) {
    std::string info = g_sd_state.get_model_info();
    return env->NewStringUTF(info.c_str());
}

/**
 * Apply a LoRA to the current model.
 *
 * @param loraPath Path to LoRA .safetensors file
 * @param weight LoRA strength multiplier
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeApplyLora(
        JNIEnv* env, jobject /* thiz */, jstring loraPath, jfloat weight) {
    std::unique_lock lock(g_sd_mtx);
    std::string path = jstring_to_string(env, loraPath);
    return g_sd_state.apply_lora(path, weight) ? JNI_TRUE : JNI_FALSE;
}

/**
 * Clear all applied LoRA weights.
 */
JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeClearLora(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::unique_lock lock(g_sd_mtx);
    g_sd_state.clear_lora();
}

// =========================================================================
// Upscaler (Phase 5.1)
// =========================================================================

// Upscaler state — separate from diffusion pipeline
static std::mutex g_upscaler_mtx;
static bool g_upscaler_loaded = false;
static bool g_upscaler_use_mnn = false;
static std::string g_upscaler_model_path;

/**
 * Load a 4x upscaler model.
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadUpscaler(
        JNIEnv* env, jobject /* thiz */,
        jstring modelPath, jboolean useMnn, jboolean useOpenCL) {
    std::lock_guard<std::mutex> lock(g_upscaler_mtx);

    g_upscaler_model_path = jstring_to_string(env, modelPath);
    g_upscaler_use_mnn = useMnn;

    if (g_upscaler_model_path.empty()) {
        SD_LOG_ERROR("[UPSCALER] Model path is empty");
        return JNI_FALSE;
    }

    if (!useMnn) {
        // QNN path: upscalerApp is a global managed by model_loader
        // It should be loaded during nativeLoadModel if upscaler_path was provided
        // For standalone upscaler loading, we'd need to create QnnModel here
        // For now, check if it's already loaded via the pipeline
        SD_LOG_INFO("[UPSCALER] QNN upscaler path set: %s", g_upscaler_model_path.c_str());
    } else {
        // MNN path: model is loaded on-demand in upscaleImageWithMNN()
        SD_LOG_INFO("[UPSCALER] MNN upscaler path set: %s (opencl=%d)",
                    g_upscaler_model_path.c_str(), (int)useOpenCL);
    }

    g_upscaler_loaded = true;
    return JNI_TRUE;
}

/**
 * Upscale an image 4x.
 */
JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeUpscaleImage(
        JNIEnv* env, jobject /* thiz */,
        jbyteArray inputRgb, jint width, jint height, jobject callback) {
    std::lock_guard<std::mutex> lock(g_upscaler_mtx);

    if (!g_upscaler_loaded) {
        sd_jni::on_error(env, callback, "Upscaler not loaded");
        return JNI_FALSE;
    }

    if (!inputRgb) {
        sd_jni::on_error(env, callback, "Input image is null");
        return JNI_FALSE;
    }

    jsize inputLen = env->GetArrayLength(inputRgb);
    if (inputLen != width * height * 3) {
        sd_jni::on_error(env, callback, "Input size mismatch (expected width*height*3)");
        return JNI_FALSE;
    }

    // Copy input bytes
    jbyte* inputBytes = env->GetByteArrayElements(inputRgb, nullptr);
    JniByteArrayGuard inputGuard(env, inputRgb, inputBytes);

    std::vector<uint8_t> input_image(inputLen);
    memcpy(input_image.data(), inputBytes, inputLen);

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        xt::xarray<uint8_t> result;
        if (g_upscaler_use_mnn) {
            result = upscaleImageWithMNN(input_image, width, height,
                                          g_upscaler_model_path, false);
        } else {
            // QNN path uses the global upscalerApp
            result = upscaleImageWithModel(input_image, width, height, upscalerApp);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        // Convert xt::xarray output to contiguous RGB bytes
        int out_width = width * 4;
        int out_height = height * 4;
        int out_size = out_width * out_height * 3;

        sd_jni::on_complete(env, callback, result.data(), out_size,
                            out_width, out_height, 0, timeMs);
        return JNI_TRUE;

    } catch (const std::exception& e) {
        SD_LOG_ERROR("[UPSCALER] Upscale failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

/**
 * Release upscaler model resources.
 */
JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeReleaseUpscaler(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_upscaler_mtx);
    g_upscaler_loaded = false;
    g_upscaler_model_path.clear();
    SD_LOG_INFO("[UPSCALER] Released");
}

// =========================================================================
// Hardware Info (SoC / NPU detection at C++ level)
// =========================================================================

static std::string read_sysfs(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return "";
    char buf[256] = {};
    if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
    fclose(f);
    size_t len = strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = 0;
    return buf;
}

/**
 * Get SoC hardware info from sysfs + detect QNN HTP at native level.
 *
 * Returns JSON string:
 * {
 *   "soc_id": "636",
 *   "machine": "VOLCANO",
 *   "family": "Snapdragon",
 *   "revision": "1.0",
 *   "htp_version": 73,
 *   "has_qnn_htp": true
 * }
 */
JNIEXPORT jstring JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeGetSocInfo(
        JNIEnv* env, jobject /* thiz */) {

    std::string socId   = read_sysfs("/sys/devices/soc0/soc_id");
    std::string machine = read_sysfs("/sys/devices/soc0/machine");
    std::string family  = read_sysfs("/sys/devices/soc0/family");
    std::string rev     = read_sysfs("/sys/devices/soc0/revision");

    // Detect HTP version from vendor stub libraries
    int htpVersion = 0;
    const char* vendorDirs[] = {"/vendor/lib64", "/vendor/lib", "/system/vendor/lib64"};
    for (const char* dir : vendorDirs) {
        DIR* d = opendir(dir);
        if (!d) continue;
        struct dirent* entry;
        while ((entry = readdir(d)) != nullptr) {
            const char* name = entry->d_name;
            const char* prefix = "libQnnHtpV";
            if (strncmp(name, prefix, 10) == 0) {
                int v = atoi(name + 10);
                if (v > htpVersion) htpVersion = v;
            }
        }
        closedir(d);
        if (htpVersion > 0) break;
    }

    // Check QNN HTP library exists
    bool hasQnnHtp = false;
    const char* qnnPaths[] = {
        "/vendor/lib64/libQnnHtp.so",
        "/vendor/lib/libQnnHtp.so",
        "/system/vendor/lib64/libQnnHtp.so"
    };
    for (const char* p : qnnPaths) {
        FILE* f = fopen(p, "r");
        if (f) { fclose(f); hasQnnHtp = true; break; }
    }

    char json[512];
    snprintf(json, sizeof(json),
        "{\"soc_id\":\"%s\",\"machine\":\"%s\",\"family\":\"%s\","
        "\"revision\":\"%s\",\"htp_version\":%d,\"has_qnn_htp\":%s}",
        socId.c_str(), machine.c_str(), family.c_str(), rev.c_str(),
        htpVersion, hasQnnHtp ? "true" : "false");

    SD_LOG_INFO("SoC Info: %s", json);
    return env->NewStringUTF(json);
}

} // extern "C"
