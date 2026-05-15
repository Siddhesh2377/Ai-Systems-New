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

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "state/diffusion_state.h"
#include "upscaler/upscaler.h"
#include "segmentation/segmenter.h"
#include "inpainting/lama_inpainter.h"
#include "depth/depth_estimator.h"
#include "style/style_transfer.h"
#include "pipeline/pipeline_globals.h"
#include "model/qnn_model.h"  // full type needed for unique_ptr<QnnModel>::reset()
#include "loader/model_loader.h"  // ensureQnnSystemReady + loadStandaloneQnnUpscaler
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

// QNN runtime lib dir captured by nativeInitRuntime — used by the standalone
// QNN upscaler load to compute libQnnHtp.so / libQnnSystem.so paths.
static std::string g_qnn_lib_dir;

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
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_INIT,
               "nativeInitRuntime: qnnLibDir is empty");
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

    // Cache the lib dir so standalone helpers (e.g. QNN upscaler load)
    // can resolve libQnnHtp.so / libQnnSystem.so paths without a separate
    // diffusion-model load. Best-effort populate the QNN system funcs now;
    // failures here don't block runtime init since callers may still use
    // CPU/MNN paths.
    g_qnn_lib_dir = libDir;
    std::string qnnSystemLibPath = libDir + "/libQnnSystem.so";
    std::string qnnBackendPath = libDir + "/libQnnHtp.so";
    if (!sd_pipeline::ensureQnnSystemReady(qnnSystemLibPath, qnnBackendPath)) {
        SD_LOG_WARN("ensureQnnSystemReady failed — QNN-only paths will fail "
                    "until a diffusion model load succeeds");
    }
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
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_INIT,
               "nativeGenerate: model not loaded");
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

    // Handle mask input.
    // Two valid sizes: latent-space (sampleW*sampleH*3) or full-resolution
    // (width*height*3). Anything else means the caller passed the wrong shape
    // and would silently produce a wrong inpaint without this check.
    if (mask) {
        jsize maskLen = env->GetArrayLength(mask);
        if (maskLen > 0) {
            int sampleW = width / 8;
            int sampleH = height / 8;
            jsize expectedLatent = sampleW * sampleH * 3;
            jsize expectedFull   = width * height * 3;
            if (maskLen != expectedLatent && maskLen != expectedFull) {
                SD_LOG_ERROR(
                    "Mask size mismatch: got %d bytes; expected %d (latent %dx%dx3) "
                    "or %d (full %dx%dx3). Skipping mask.",
                    (int)maskLen, expectedLatent, sampleW, sampleH,
                    expectedFull, width, height);
            } else {
                params.hasMask = true;
                jbyte* maskBytes = env->GetByteArrayElements(mask, nullptr);
                JniByteArrayGuard maskGuard(env, mask, maskBytes);

                // Latent-space mask (4 channels). Downsample if caller passed
                // full-resolution bytes; otherwise read directly.
                params.mask.resize(4 * sampleW * sampleH);
                bool isFullRes = (maskLen == expectedFull);
                for (int y = 0; y < sampleH; y++) {
                    for (int x = 0; x < sampleW; x++) {
                        int srcIdx = isFullRes
                            ? ((y * 8) * width + (x * 8)) * 3
                            : (y * sampleW + x) * 3;
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
            TN_CANCEL("user requested stop");
            sd_jni::on_error(env, callback, "Generation cancelled");
            return JNI_FALSE;
        }

        if (result.imageData.empty()) {
            TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_VAE,
                   "Generation produced no image data");
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
        // Stage is intentionally generic at the JNI layer — the structured
        // error from the failing stage (CLIP / UNET / VAE) was already emitted
        // by the orchestrator / loader / runner that originated the throw.
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_UNSPECIFIED,
               "Generation failed: %s", e.what());
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
// Defined at the bottom of this TU — needs the full unique_ptr<T> types of
// the module globals which appear later in the file.
static void cleanup_secondary_modules();

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeRelease(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::unique_lock lock(g_sd_mtx);
    g_sd_state.release();
    sd_jni::reset_cache();
    // Tear down upscaler / segmenter / lama / depth / style globals too. Host
    // apps call nativeRelease expecting a clean slate before model swap or
    // shutdown; those secondary modules used to leak across.
    cleanup_secondary_modules();
    SD_LOG_INFO("Models released (full)");
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
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_LOAD,
               "[UPSCALER] Model path is empty");
        return JNI_FALSE;
    }

    if (!useMnn) {
        // Standalone QNN upscaler load: createQnnModel + initializeQnnApp,
        // mirroring LocalDream's per-request /upscale handler.
        // ensureQnnSystemReady was best-effort'd inside nativeInitRuntime;
        // re-attempt here if it didn't take (e.g. caller skipped init).
        if (!g_qnn_lib_dir.empty()) {
            std::string sys = g_qnn_lib_dir + "/libQnnSystem.so";
            std::string bk = g_qnn_lib_dir + "/libQnnHtp.so";
            (void)sd_pipeline::ensureQnnSystemReady(sys, bk);
        }
        if (!sd_pipeline::loadStandaloneQnnUpscaler(g_upscaler_model_path)) {
            TN_ERR_FIX(TN_CODE_MODEL_LOAD_FAIL, TN_STAGE_LOAD,
                       "Check that the upscaler .bin matches this SoC's HTP version, "
                       "or pass useMnn=true to use the CPU/GPU fallback.",
                       "[UPSCALER] Standalone QNN load failed for %s",
                       g_upscaler_model_path.c_str());
            g_upscaler_loaded = false;
            return JNI_FALSE;
        }
        SD_LOG_INFO("[UPSCALER] QNN upscaler loaded: %s",
                    g_upscaler_model_path.c_str());
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
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_UPSCALE, "Upscaler not loaded");
        sd_jni::on_error(env, callback, "Upscaler not loaded");
        return JNI_FALSE;
    }

    if (!inputRgb) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_UPSCALE,
               "Input image is null");
        sd_jni::on_error(env, callback, "Input image is null");
        return JNI_FALSE;
    }

    jsize inputLen = env->GetArrayLength(inputRgb);
    if (inputLen != width * height * 3) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_UPSCALE,
               "Input size mismatch: got %d bytes, expected %d (%dx%d*3)",
               (int)inputLen, width * height * 3, width, height);
        sd_jni::on_error(env, callback, "Input size mismatch (expected width*height*3)");
        return JNI_FALSE;
    }

    // Cap input at 2048×2048: 4× output = 8192×8192 → ~768 MB heap. Larger
    // inputs OOM-crash on most Android devices.
    constexpr int UPSCALER_MAX_DIM = 2048;
    if (width > UPSCALER_MAX_DIM || height > UPSCALER_MAX_DIM) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "Upscaler input too large (%dx%d). Max %d on each side; downscale first.",
                 width, height, UPSCALER_MAX_DIM);
        TN_ERR_FIX(TN_CODE_RESOURCE_EXHAUSTED, TN_STAGE_SD_UPSCALE,
                   "Downscale the image before requesting 4x upscale.",
                   "%s", buf);
        sd_jni::on_error(env, callback, buf);
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
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_UPSCALE,
               "[UPSCALER] Upscale failed: %s", e.what());
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
    // Free the QNN upscaler global so memory is reclaimed without a full
    // nativeRelease. MNN path uses a function-local interpreter; nothing to do.
    if (upscalerApp) {
        upscalerApp.reset();
        SD_LOG_INFO("[UPSCALER] Released QNN upscalerApp");
    }
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

// Map known Qualcomm soc_id -> {chipset, marketing generation, sdxl support,
// recommended QNN model variant}. The soc_id is the canonical sysfs identifier
// (kernel exposes it as a decimal int). When soc_id is not in the table we
// fall back to inferring from `machine` if possible. Unknown -> conservative
// "min" variant which the xororz bundles target as their broadest fallback.
struct SocEntry {
    int soc_id;
    const char* chipset;       // SM-number
    const char* marketing;     // "8 Gen 3" etc.
    bool supports_sdxl;
    const char* recommended_variant; // "8gen1", "8gen2", "8gen3", "min"
};
static const SocEntry kSocTable[] = {
    // 7-series
    { 530, "SM7325", "778G",          false, "min"   },
    { 636, "SM7475", "7s Gen 3",      false, "min"   },
    // 8-series
    { 415, "SM8350", "8 Gen 1",       false, "8gen1" },
    { 475, "SM8450", "8 Gen 1+",      false, "8gen1" },
    { 502, "SM8475", "8+ Gen 1",      false, "8gen1" },
    { 519, "SM8550", "8 Gen 2",       true,  "8gen2" },
    { 557, "SM8650", "8 Gen 3",       true,  "8gen3" },
    { 614, "SM8750", "8 Elite",       true,  "8gen3" },
    // 8 Gen 5 / sm8845 — added in xororz Apr 2026 commit. Real id verified at
    // device first-boot; placeholder until confirmed.
    { 678, "SM8845", "8 Gen 5",       true,  "8gen3" },
};
static const SocEntry* find_soc(int id) {
    for (const auto& e : kSocTable) if (e.soc_id == id) return &e;
    return nullptr;
}

/**
 * Get SoC hardware info from sysfs + detect QNN HTP at native level.
 *
 * Returns JSON string with raw sysfs data plus a derived capability section:
 * {
 *   "soc_id": "636",
 *   "machine": "VOLCANO",
 *   "family": "Snapdragon",
 *   "revision": "1.0",
 *   "htp_version": 73,
 *   "has_qnn_htp": true,
 *   "chipset": "SM7475",
 *   "marketing": "7s Gen 3",
 *   "supports_sdxl": false,
 *   "recommended_variant": "min"
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

    // Capability lookup. Unknown soc_id -> conservative defaults.
    int soc_id_int = atoi(socId.c_str());
    const SocEntry* soc = find_soc(soc_id_int);
    const char* chipset = soc ? soc->chipset : "unknown";
    const char* marketing = soc ? soc->marketing : "unknown";
    bool supports_sdxl = soc ? soc->supports_sdxl : false;
    const char* recommended_variant = soc ? soc->recommended_variant : "min";

    // HTP version cross-check: SDXL needs at least HTP V73 (8 Gen 2 era).
    // If SoC table says yes but HTP libs are older, downgrade.
    if (supports_sdxl && htpVersion > 0 && htpVersion < 73) {
        supports_sdxl = false;
    }

    char json[1024];
    snprintf(json, sizeof(json),
        "{\"soc_id\":\"%s\",\"machine\":\"%s\",\"family\":\"%s\","
        "\"revision\":\"%s\",\"htp_version\":%d,\"has_qnn_htp\":%s,"
        "\"chipset\":\"%s\",\"marketing\":\"%s\","
        "\"supports_sdxl\":%s,\"recommended_variant\":\"%s\"}",
        socId.c_str(), machine.c_str(), family.c_str(), rev.c_str(),
        htpVersion, hasQnnHtp ? "true" : "false",
        chipset, marketing,
        supports_sdxl ? "true" : "false", recommended_variant);

    SD_LOG_INFO("SoC Info: %s", json);
    return env->NewStringUTF(json);
}

// =========================================================================
// Segmenter (Phase 5.3 — MobileSAM)
// =========================================================================

static std::mutex g_segmenter_mtx;
static std::unique_ptr<Segmenter> g_segmenter;

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadSegmenter(
        JNIEnv* env, jobject /* thiz */,
        jstring encoderPath, jstring decoderPath, jboolean useOpenCL) {
    std::lock_guard<std::mutex> lock(g_segmenter_mtx);

    std::string encoder = jstring_to_string(env, encoderPath);
    std::string decoder = jstring_to_string(env, decoderPath);

    if (encoder.empty() || decoder.empty()) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_SEGMENT,
               "[SEGMENTER] Encoder or decoder path is empty");
        return JNI_FALSE;
    }

    g_segmenter = std::make_unique<Segmenter>();
    bool ok = g_segmenter->loadModel(encoder, decoder, useOpenCL);
    if (!ok) {
        g_segmenter.reset();
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_SEGMENT,
               "[SEGMENTER] Failed to load models (encoder=%s decoder=%s)",
               encoder.c_str(), decoder.c_str());
    } else {
        SD_LOG_INFO("[SEGMENTER] Loaded encoder=%s decoder=%s", encoder.c_str(), decoder.c_str());
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeSegmenterEncodeImage(
        JNIEnv* env, jobject /* thiz */,
        jbyteArray rgbBytes, jint width, jint height) {
    std::lock_guard<std::mutex> lock(g_segmenter_mtx);

    if (!g_segmenter || !g_segmenter->isLoaded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_SEGMENT, "[SEGMENTER] Not loaded");
        return JNI_FALSE;
    }

    jbyte* bytes = env->GetByteArrayElements(rgbBytes, nullptr);
    JniByteArrayGuard guard(env, rgbBytes, bytes);

    bool ok = g_segmenter->encodeImage(reinterpret_cast<const uint8_t*>(bytes), width, height);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeSegmentAtPoint(
        JNIEnv* env, jobject /* thiz */,
        jfloat x, jfloat y, jobject callback) {
    std::lock_guard<std::mutex> lock(g_segmenter_mtx);

    if (!g_segmenter || !g_segmenter->isEncoded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_SEGMENT,
               "Segmenter not ready (load model + encode image first)");
        sd_jni::on_error(env, callback, "Segmenter not ready (load model + encode image first)");
        return JNI_FALSE;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        float score = 0.0f;
        auto mask = g_segmenter->segmentAtPoint(x, y, score);
        auto end = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Return mask as single-channel data through the callback
        // We pack score into the seed field (reinterpreted as long)
        long scoreAsLong = static_cast<long>(score * 10000);  // score * 10000 for precision
        int maskW = static_cast<int>(std::sqrt(mask.size()));  // mask is square (encoder output size)
        int maskH = maskW;
        sd_jni::on_complete(env, callback, mask.data(), static_cast<int>(mask.size()),
                            maskW, maskH, scoreAsLong, timeMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_SEGMENT,
               "[SEGMENTER] segmentAtPoint failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeSegmentWithBox(
        JNIEnv* env, jobject /* thiz */,
        jfloat x1, jfloat y1, jfloat x2, jfloat y2, jobject callback) {
    std::lock_guard<std::mutex> lock(g_segmenter_mtx);

    if (!g_segmenter || !g_segmenter->isEncoded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_SEGMENT,
               "Segmenter not ready (load model + encode image first)");
        sd_jni::on_error(env, callback, "Segmenter not ready (load model + encode image first)");
        return JNI_FALSE;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        float score = 0.0f;
        auto mask = g_segmenter->segmentWithBox(x1, y1, x2, y2, score);
        auto end = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        long scoreAsLong = static_cast<long>(score * 10000);
        int maskW = static_cast<int>(std::sqrt(mask.size()));
        int maskH = maskW;
        sd_jni::on_complete(env, callback, mask.data(), static_cast<int>(mask.size()),
                            maskW, maskH, scoreAsLong, timeMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_SEGMENT,
               "[SEGMENTER] segmentWithBox failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeReleaseSegmenter(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_segmenter_mtx);
    if (g_segmenter) {
        g_segmenter->release();
        g_segmenter.reset();
    }
    SD_LOG_INFO("[SEGMENTER] Released");
}

// =========================================================================
// LaMa Inpainter (Phase 5.4)
// =========================================================================

static std::mutex g_lama_mtx;
static std::unique_ptr<LamaInpainter> g_lama;

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadLamaInpainter(
        JNIEnv* env, jobject /* thiz */,
        jstring modelPath, jboolean useOpenCL) {
    std::lock_guard<std::mutex> lock(g_lama_mtx);

    std::string path = jstring_to_string(env, modelPath);
    if (path.empty()) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_INPAINT,
               "[LAMA] Model path is empty");
        return JNI_FALSE;
    }

    g_lama = std::make_unique<LamaInpainter>();
    bool ok = g_lama->loadModel(path, useOpenCL);
    if (!ok) {
        g_lama.reset();
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_INPAINT,
               "[LAMA] Failed to load model: %s", path.c_str());
    } else {
        SD_LOG_INFO("[LAMA] Loaded: %s", path.c_str());
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLamaInpaint(
        JNIEnv* env, jobject /* thiz */,
        jbyteArray rgbBytes, jbyteArray maskBytes, jint width, jint height,
        jobject callback) {
    std::lock_guard<std::mutex> lock(g_lama_mtx);

    if (!g_lama || !g_lama->isLoaded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_INPAINT,
               "LaMa inpainter not loaded");
        sd_jni::on_error(env, callback, "LaMa inpainter not loaded");
        return JNI_FALSE;
    }

    jbyte* rgb = env->GetByteArrayElements(rgbBytes, nullptr);
    JniByteArrayGuard rgbGuard(env, rgbBytes, rgb);
    jbyte* mask = env->GetByteArrayElements(maskBytes, nullptr);
    JniByteArrayGuard maskGuard(env, maskBytes, mask);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = g_lama->inpaint(
            reinterpret_cast<const uint8_t*>(rgb),
            reinterpret_cast<const uint8_t*>(mask),
            width, height);
        auto end = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        sd_jni::on_complete(env, callback, result.data(), static_cast<int>(result.size()),
                            width, height, 0, timeMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_INPAINT,
               "[LAMA] Inpaint failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeReleaseLamaInpainter(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_lama_mtx);
    if (g_lama) {
        g_lama->release();
        g_lama.reset();
    }
    SD_LOG_INFO("[LAMA] Released");
}

// =========================================================================
// Depth Estimator (Phase 5.5)
// =========================================================================

static std::mutex g_depth_mtx;
static std::unique_ptr<DepthEstimator> g_depth;

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadDepthEstimator(
        JNIEnv* env, jobject /* thiz */,
        jstring modelPath, jboolean useOpenCL) {
    std::lock_guard<std::mutex> lock(g_depth_mtx);

    std::string path = jstring_to_string(env, modelPath);
    if (path.empty()) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_DEPTH,
               "[DEPTH] Model path is empty");
        return JNI_FALSE;
    }

    g_depth = std::make_unique<DepthEstimator>();
    bool ok = g_depth->loadModel(path, useOpenCL);
    if (!ok) {
        g_depth.reset();
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_DEPTH,
               "[DEPTH] Failed to load model: %s", path.c_str());
    } else {
        SD_LOG_INFO("[DEPTH] Loaded: %s", path.c_str());
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeEstimateDepthColorized(
        JNIEnv* env, jobject /* thiz */,
        jbyteArray rgbBytes, jint width, jint height, jobject callback) {
    std::lock_guard<std::mutex> lock(g_depth_mtx);

    if (!g_depth || !g_depth->isLoaded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_DEPTH,
               "Depth estimator not loaded");
        sd_jni::on_error(env, callback, "Depth estimator not loaded");
        return JNI_FALSE;
    }

    jbyte* bytes = env->GetByteArrayElements(rgbBytes, nullptr);
    JniByteArrayGuard guard(env, rgbBytes, bytes);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = g_depth->estimateDepthColorized(
            reinterpret_cast<const uint8_t*>(bytes), width, height);
        auto end = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        sd_jni::on_complete(env, callback, result.data(), static_cast<int>(result.size()),
                            width, height, 0, timeMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_DEPTH,
               "[DEPTH] Estimation failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeReleaseDepthEstimator(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_depth_mtx);
    if (g_depth) {
        g_depth->release();
        g_depth.reset();
    }
    SD_LOG_INFO("[DEPTH] Released");
}

// =========================================================================
// Style Transfer (Phase 5.6)
// =========================================================================

static std::mutex g_style_mtx;
static std::unique_ptr<StyleTransfer> g_style;

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeLoadStyleTransfer(
        JNIEnv* env, jobject /* thiz */,
        jstring modelPath, jboolean useOpenCL) {
    std::lock_guard<std::mutex> lock(g_style_mtx);

    std::string path = jstring_to_string(env, modelPath);
    if (path.empty()) {
        TN_ERR(TN_CODE_INVALID_PARAM, TN_STAGE_SD_STYLE,
               "[STYLE] Model path is empty");
        return JNI_FALSE;
    }

    g_style = std::make_unique<StyleTransfer>();
    bool ok = g_style->loadModel(path, useOpenCL);
    if (!ok) {
        g_style.reset();
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_STYLE,
               "[STYLE] Failed to load model: %s", path.c_str());
    } else {
        SD_LOG_INFO("[STYLE] Loaded: %s", path.c_str());
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeStylize(
        JNIEnv* env, jobject /* thiz */,
        jbyteArray contentRgb, jint contentW, jint contentH,
        jbyteArray styleRgb, jint styleW, jint styleH,
        jfloat strength, jobject callback) {
    std::lock_guard<std::mutex> lock(g_style_mtx);

    if (!g_style || !g_style->isLoaded()) {
        TN_ERR(TN_CODE_NOT_READY, TN_STAGE_SD_STYLE,
               "Style transfer not loaded");
        sd_jni::on_error(env, callback, "Style transfer not loaded");
        return JNI_FALSE;
    }

    jbyte* content = env->GetByteArrayElements(contentRgb, nullptr);
    JniByteArrayGuard contentGuard(env, contentRgb, content);
    jbyte* style = env->GetByteArrayElements(styleRgb, nullptr);
    JniByteArrayGuard styleGuard(env, styleRgb, style);

    try {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = g_style->stylize(
            reinterpret_cast<const uint8_t*>(content), contentW, contentH,
            reinterpret_cast<const uint8_t*>(style), styleW, styleH,
            strength);
        auto end = std::chrono::high_resolution_clock::now();
        int timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        sd_jni::on_complete(env, callback, result.data(), static_cast<int>(result.size()),
                            contentW, contentH, 0, timeMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_SD_STYLE,
               "[STYLE] Stylize failed: %s", e.what());
        sd_jni::on_error(env, callback, e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeReleaseStyleTransfer(
        JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_style_mtx);
    if (g_style) {
        g_style->release();
        g_style.reset();
    }
    SD_LOG_INFO("[STYLE] Released");
}

// Returns a flat int[] of supported (w, h) pairs the loaded UNet binary
// can run at. Filesystem-only — no model load required, no QNN init.
// Caller passes the model directory and the base resolution baked into
// the .bin (xororz/Mr-J-369 packs put this in the dir name, e.g.
// `output_512` → 512×512). Used by the consuming app to populate the
// resolution selector, which is the cleanest fix for the silent
// noise-on-1024² problem we hit when the patch file is missing.
JNIEXPORT jintArray JNICALL
Java_com_dark_ai_1sd_SDNativeLib_nativeGetSupportedResolutions(
        JNIEnv* env, jobject /* thiz */,
        jstring modelDir, jint baseWidth, jint baseHeight) {
    std::string dir = jstring_to_string(env, modelDir);
    auto resolutions = sd_pipeline::get_supported_resolutions(
        dir, static_cast<int>(baseWidth), static_cast<int>(baseHeight));

    jsize n = static_cast<jsize>(resolutions.size() * 2);
    jintArray result = env->NewIntArray(n);
    if (!result) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        return nullptr;
    }
    if (n == 0) return result;

    std::vector<jint> flat;
    flat.reserve(static_cast<size_t>(n));
    for (const auto& r : resolutions) {
        flat.push_back(r.width);
        flat.push_back(r.height);
    }
    env->SetIntArrayRegion(result, 0, n, flat.data());
    return result;
}

} // extern "C"

// Defined here so the unique_ptr<T> globals' full types are in scope for
// reset(). Each module owns its own mutex; we lock then drop the pointer so
// the destructor runs on each.
static void cleanup_secondary_modules() {
    {
        std::lock_guard<std::mutex> u(g_upscaler_mtx);
        g_upscaler_loaded = false;
        g_upscaler_model_path.clear();
        if (upscalerApp) upscalerApp.reset();
    }
    { std::lock_guard<std::mutex> l(g_segmenter_mtx); g_segmenter.reset(); }
    { std::lock_guard<std::mutex> l(g_lama_mtx);      g_lama.reset();      }
    { std::lock_guard<std::mutex> l(g_depth_mtx);     g_depth.reset();     }
    { std::lock_guard<std::mutex> l(g_style_mtx);     g_style.reset();     }
    SD_LOG_INFO("Secondary modules released");
}
