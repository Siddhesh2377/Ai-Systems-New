/**
 * Model Loader — extracted from diffusion_pipeline.cpp Phase 1.2.
 *
 * Contains:
 *   - initialize_models(): Load tokenizer, CLIP, UNet, VAE, safety checker
 *   - cleanup(): Release all models and sessions
 *   - createQnnModel(): QNN model factory
 *   - ZSTD patch functions: applyZstdPatch, applyZstdPatchToBuffer
 */

#include "model_loader.h"
#include "../pipeline/pipeline_globals.h"
#include "../core/edit_cache.h"
#include "../state/diffusion_state.h"
#include "../utils/sd_logger.h"
#include "../utils/config.h"
#include "../utils/float_conversion.h"
#include "../utils/sd_utils.h"
#include "../model/qnn_model.h"
#include "../pipeline/prompt_processor.h"

// QNN Headers
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnSampleAppUtils.hpp"

// External Libraries
#include "tokenizers_cpp.h"

// MNN
#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>

// ZSTD
#include "zstd.h"

#include <cstdio>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// Local state — only model_loader needs these
// ============================================================================

struct PatchedModelBuffer {
    std::shared_ptr<uint8_t> buffer;
    uint64_t size;

    PatchedModelBuffer() : buffer(nullptr), size(0) {}
    PatchedModelBuffer(uint8_t *buf, uint64_t sz)
        : buffer(buf, std::default_delete<uint8_t[]>()), size(sz) {}
    void reset() { buffer.reset(); size = 0; }
};

static std::unique_ptr<PatchedModelBuffer> g_unetPatchedBuffer;
static qnn::tools::sample_app::QnnFunctionPointers g_qnnSystemFuncs;
static std::string g_backendPathCmd;

// ============================================================================
// QNN Model Factory
// ============================================================================

static std::unique_ptr<QnnModel> createQnnModel(const std::string &modelPath,
                                                 const std::string &modelName) {
    using namespace qnn::tools;
    sample_app::QnnFunctionPointers funcs = g_qnnSystemFuncs;
    void *backendHandle = nullptr;
    void *modelHandle = nullptr;
    dynamicloadutil::StatusCode drvStatus =
        dynamicloadutil::getQnnFunctionPointers(g_backendPathCmd, modelPath,
                                                &funcs, &backendHandle, false,
                                                &modelHandle);
    if (drvStatus != dynamicloadutil::StatusCode::SUCCESS) {
        QNN_ERROR("Failed get QNN func ptrs for %s.", modelName.c_str());
        return nullptr;
    }
    std::string inputListPaths, opPackagePaths, outputPath, saveBinaryName;
    bool debug = false;
    bool dumpOutputs = false;
    iotensor::OutputDataType outputDataType =
        iotensor::OutputDataType::FLOAT_ONLY;
    iotensor::InputDataType inputDataType = iotensor::InputDataType::FLOAT;
    sample_app::ProfilingLevel profilingLevel = ProfilingLevel::OFF;
    return std::make_unique<QnnModel>(
        funcs, inputListPaths, opPackagePaths, backendHandle, outputPath, debug,
        outputDataType, inputDataType, profilingLevel, dumpOutputs, modelPath,
        saveBinaryName);
}

// ============================================================================
// ZSTD Patch Functions
// ============================================================================

namespace qnn {
namespace tools {
namespace sample_app {

static std::vector<char> readFileForPatch(const std::string &filePath) {
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (size > 0) {
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Failed to read file: " + filePath);
        }
    }
    return buffer;
}

static std::unique_ptr<PatchedModelBuffer> applyZstdPatchToBuffer(
    const std::string &oldFilePath, const std::string &patchFilePath) {
    try {
        std::vector<char> oldFileBuffer = readFileForPatch(oldFilePath);
        QNN_INFO("Read old file (%s): %zu bytes.", oldFilePath.c_str(),
                 oldFileBuffer.size());

        std::vector<char> patchFileBuffer = readFileForPatch(patchFilePath);
        QNN_INFO("Read patch file (%s): %zu bytes.", patchFilePath.c_str(),
                 patchFileBuffer.size());

        if (patchFileBuffer.empty()) {
            throw std::runtime_error("Patch file (" + patchFilePath +
                                     ") is empty or could not be read.");
        }

        unsigned long long const decompressedSize = ZSTD_getFrameContentSize(
            patchFileBuffer.data(), patchFileBuffer.size());

        if (decompressedSize == ZSTD_CONTENTSIZE_ERROR) {
            throw std::runtime_error("Patch file (" + patchFilePath +
                                     ") is not a valid zstd frame.");
        }
        if (decompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error(
                "Decompressed size is unknown. Cannot proceed.");
        }

        if (decompressedSize == 0) {
            QNN_ERROR("Patch resulted in empty buffer.");
            return nullptr;
        }

        auto newBuffer = std::make_unique<uint8_t[]>(decompressedSize);

        std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx(
            ZSTD_createDCtx(), &ZSTD_freeDCtx);
        if (!dctx) {
            throw std::runtime_error("ZSTD_createDCtx() failed!");
        }

        size_t const actualDecompressedSize = ZSTD_decompress_usingDict(
            dctx.get(), newBuffer.get(), decompressedSize, patchFileBuffer.data(),
            patchFileBuffer.size(), oldFileBuffer.data(), oldFileBuffer.size());

        if (ZSTD_isError(actualDecompressedSize)) {
            throw std::runtime_error(
                "ZSTD_decompress_usingDict() failed: " +
                std::string(ZSTD_getErrorName(actualDecompressedSize)));
        }

        QNN_INFO("Successfully applied patch to buffer. Decompressed %zu bytes.",
                 actualDecompressedSize);

        return std::make_unique<PatchedModelBuffer>(newBuffer.release(),
                                                    actualDecompressedSize);

    } catch (const std::exception &e) {
        QNN_ERROR("Error applying patch to buffer: %s", e.what());
        return nullptr;
    }
}

// QnnModel Initialization
template <typename AppType>
static int initializeQnnApp(const std::string &modelName,
                            std::unique_ptr<AppType> &app,
                            const uint8_t *buffer = nullptr,
                            uint64_t bufferSize = 0) {
    if (!app) return EXIT_FAILURE;

    if (buffer && bufferSize > 0) {
        QNN_INFO("Initializing QNN App from Buffer: %s (size: %llu bytes)",
                 modelName.c_str(), bufferSize);
    } else {
        QNN_INFO("Initializing QNN App from Cache: %s", modelName.c_str());
    }

    if (StatusCode::SUCCESS != app->initialize())
        return app->reportError(modelName + " Init failure");
    if (StatusCode::SUCCESS != app->initializeBackend())
        return app->reportError(modelName + " Backend Init failure");
    auto devPropStat = app->isDevicePropertySupported();
    if (StatusCode::FAILURE != devPropStat) {
        if (StatusCode::SUCCESS != app->createDevice())
            return app->reportError(modelName + " Device Creation failure");
    }
    if (StatusCode::SUCCESS != app->initializeProfiling())
        return app->reportError(modelName + " Profiling Init failure");
    if (StatusCode::SUCCESS != app->registerOpPackages())
        return app->reportError(modelName + " Register Op Packages failure");

    if (buffer && bufferSize > 0) {
        if (StatusCode::SUCCESS != app->createFromBuffer(buffer, bufferSize))
            return app->reportError(modelName + " Create From Buffer failure");
    } else {
        if (StatusCode::SUCCESS != app->createFromBinary())
            return app->reportError(modelName + " Create From Binary failure");
    }

    // BURST mode for the UNet denoising loop only — the inner loop runs
    // 10-30+ short calls back to back, so we want peak HTP frequency for
    // the first few steps. CLIP and VAE are single-shot per generation,
    // so PERFORMANCE_MODE (sustained) is appropriate there.
    bool useBurstMode = (modelName == "UNET");
    if (StatusCode::SUCCESS != app->enablePerformaceMode(useBurstMode))
        return app->reportError(modelName + " Enable Performance Mode failure");

    if (buffer && bufferSize > 0) {
        QNN_INFO("QNN App Initialized from Buffer: %s", modelName.c_str());
    } else {
        QNN_INFO("QNN App Initialized from Cache: %s", modelName.c_str());
    }
    return EXIT_SUCCESS;
}

}  // namespace sample_app
}  // namespace tools
}  // namespace qnn

// ============================================================================
// Public API
// ============================================================================

namespace sd_pipeline {

bool initialize_models(const SDModelConfig& config) {
    // Loading a new model invalidates any previously-cached CLIP embeddings.
    clear_clip_cache();

    using namespace qnn::tools;

    if (!qnn::log::initializeLogging()) {
        QNN_ERROR("Failed to initialize QNN logging");
        return false;
    }

    // Set globals from config (replacing CLI argument parsing)
    clipPath = config.clipPath;
    unetPath = config.unetPath;
    vaeDecoderPath = config.vaeDecoderPath;
    vaeEncoderPath = config.vaeEncoderPath;
    tokenizerPath = config.tokenizerPath;
    safetyCheckerPath = config.safetyCheckerPath;
    ponyv55 = config.isPony;
    use_mnn = config.runOnCpu;
    use_mnn_clip = config.useCpuClip;
    use_safety_checker = config.useSafetyChecker;
    nsfw_threshold = config.nsfwThreshold;
    text_embedding_size = config.textEmbeddingSize;
    modelDir = config.modelDir;

    // Check for clip_v2 variant. Match by exact filename (not raw substr) so a
    // custom name like `myclip.mnn` doesn't accidentally trigger the v2 sibling
    // lookup.
    std::filesystem::path clipPathObj(clipPath);
    if (clipPathObj.filename() == "clip.mnn") {
        std::filesystem::path parentDir = clipPathObj.parent_path();
        std::filesystem::path v2Path = parentDir / "clip_v2.mnn";

        if (std::filesystem::exists(v2Path)) {
            QNN_INFO("Found clip_v2.mnn, upgrading to v2 CLIP");
            clipPath = v2Path.string();
            use_clip_v2 = true;

            std::filesystem::path posEmbPath = parentDir / "pos_emb.bin";
            std::filesystem::path tokenEmbPath = parentDir / "token_emb.bin";

            if (!std::filesystem::exists(posEmbPath)) {
                QNN_ERROR("pos_emb.bin not found: %s", posEmbPath.string().c_str());
                return false;
            }
            if (!std::filesystem::exists(tokenEmbPath)) {
                QNN_ERROR("token_emb.bin not found: %s", tokenEmbPath.string().c_str());
                return false;
            }

            std::ifstream posFile(posEmbPath, std::ios::binary);
            posFile.seekg(0, std::ios::end);
            size_t posSize = posFile.tellg() / sizeof(float);
            posFile.seekg(0, std::ios::beg);
            pos_emb.resize(posSize);
            posFile.read(reinterpret_cast<char*>(pos_emb.data()), posSize * sizeof(float));
            posFile.close();

            std::ifstream tokenFile(tokenEmbPath, std::ios::binary);
            tokenFile.seekg(0, std::ios::end);
            size_t fileSize = tokenFile.tellg();
            tokenFile.seekg(0, std::ios::beg);

            const size_t SIZE_THRESHOLD = 100 * 1024 * 1024;
            if (fileSize > SIZE_THRESHOLD) {
                size_t tokenSize = fileSize / sizeof(float);
                std::vector<float> tempBuffer(tokenSize);
                tokenFile.read(reinterpret_cast<char*>(tempBuffer.data()), fileSize);
                token_emb.resize(tokenSize);
                for (size_t i = 0; i < tokenSize; i++) {
                    token_emb[i] = fp32_to_fp16(tempBuffer[i]);
                }
            } else {
                size_t tokenSize = fileSize / sizeof(uint16_t);
                token_emb.resize(tokenSize);
                tokenFile.read(reinterpret_cast<char*>(token_emb.data()), fileSize);
            }
            tokenFile.close();
        }
    }

    // Load tokenizer
    try {
        auto blob = LoadBytesFromFile(tokenizerPath);
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
        if (!tokenizer) throw std::runtime_error("Tokenizer creation failed.");
    } catch (const std::exception& e) {
        QNN_ERROR("Failed to load tokenizer: %s", e.what());
        return false;
    }

    // Load embeddings
    if (!modelDir.empty()) {
        std::filesystem::path modelPath(modelDir);
        std::filesystem::path embeddingsPath =
            modelPath.parent_path().parent_path() / "embeddings";
        if (std::filesystem::exists(embeddingsPath)) {
            try {
                // Pass the model's text embedding dim so TI files that don't
                // match (e.g. SDXL 1280/2048 in an SD1.5 769 pipeline) are
                // skipped instead of crashing the memcpy in text_encoder.cpp.
                promptProcessor.loadEmbeddings(embeddingsPath.string(),
                                               config.textEmbeddingSize);
                QNN_INFO("Loaded %zu embeddings (dim=%d filter) from %s",
                         promptProcessor.getEmbeddingCount(),
                         config.textEmbeddingSize,
                         embeddingsPath.string().c_str());
            } catch (const std::exception& e) {
                QNN_WARN("Failed to load embeddings: %s", e.what());
            }
        }
    }

    // Setup MNN sessions
    MNN::ScheduleConfig cfg_common;
    cfg_common.type = MNN_FORWARD_CPU;
    cfg_common.numThread = 1;
    MNN::BackendConfig bkCfg_common;
    bkCfg_common.memory = MNN::BackendConfig::Memory_Low;
    bkCfg_common.power = MNN::BackendConfig::Power_High;
    cfg_common.backendConfig = &bkCfg_common;
    MNN::ScheduleConfig cfg_mnn_clip = cfg_common;
    cfg_mnn_clip.numThread = 4;

    // Safety checker
    if (use_safety_checker && !safetyCheckerPath.empty()) {
        safetyCheckerInterpreter =
            MNN::Interpreter::createFromFile(safetyCheckerPath.c_str());
        if (!safetyCheckerInterpreter) {
            QNN_ERROR("Failed to load safety checker: %s", safetyCheckerPath.c_str());
            return false;
        }
        safetyCheckerSession = safetyCheckerInterpreter->createSession(cfg_common);
        if (safetyCheckerSession) {
            auto input = safetyCheckerInterpreter->getSessionInput(safetyCheckerSession, nullptr);
            safetyCheckerInterpreter->resizeTensor(input, {1, 224, 224, 3});
            safetyCheckerInterpreter->resizeSession(safetyCheckerSession);
            safetyCheckerInterpreter->releaseModel();
        }
    }

    // MNN CLIP (for CPU or hybrid mode)
    if (use_mnn_clip) {
        clipInterpreter = MNN::Interpreter::createFromFile(clipPath.c_str());
        if (!clipInterpreter) {
            QNN_ERROR("Failed to load MNN CLIP: %s", clipPath.c_str());
            return false;
        }
        clipSession = clipInterpreter->createSession(cfg_mnn_clip);
        if (clipSession) {
            if (use_clip_v2) {
                auto input = clipInterpreter->getSessionInput(clipSession, "input_embedding");
                clipInterpreter->resizeTensor(input, {1, 77, 768});
            } else {
                auto input = clipInterpreter->getSessionInput(clipSession, "input_ids");
                clipInterpreter->resizeTensor(input, {1, 77});
            }
            clipInterpreter->resizeSession(clipSession);
            clipInterpreter->releaseModel();
        }
    }

    // MNN CPU mode — validate that required MNN model files exist
    if (use_mnn) {
        // UNET is loaded lazily in UNetRunner::initIfNeeded(), but we validate now
        // to fail fast with a clear message instead of crashing during generation
        std::ifstream unetTest(unetPath);
        if (!unetTest.good()) {
            QNN_ERROR("CPU mode requires unet.mnn but file not found: %s", unetPath.c_str());
            return false;
        }
        QNN_INFO("MNN CPU mode: validated unet exists at %s", unetPath.c_str());
    }

    // QNN models — log SoC info before attempting HTP init
    if (!use_mnn) {
        // Read SoC info from sysfs for diagnostics
        auto readSysfs = [](const char* path) -> std::string {
            FILE* f = fopen(path, "r");
            if (!f) return "unknown";
            char buf[128] = {};
            if (!fgets(buf, sizeof(buf), f)) buf[0] = 0;
            fclose(f);
            size_t len = strlen(buf);
            while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = 0;
            return buf;
        };
        std::string socId   = readSysfs("/sys/devices/soc0/soc_id");
        std::string machine = readSysfs("/sys/devices/soc0/machine");
        std::string family  = readSysfs("/sys/devices/soc0/family");

        // Detect HTP version
        int htpVer = 0;
        DIR* d = opendir("/vendor/lib64");
        if (d) {
            struct dirent* entry;
            while ((entry = readdir(d)) != nullptr) {
                if (strncmp(entry->d_name, "libQnnHtpV", 10) == 0) {
                    int v = atoi(entry->d_name + 10);
                    if (v > htpVer) htpVer = v;
                }
            }
            closedir(d);
        }

        QNN_INFO("=== SoC Hardware Info ===");
        QNN_INFO("  SoC ID  : %s", socId.c_str());
        QNN_INFO("  Machine : %s", machine.c_str());
        QNN_INFO("  Family  : %s", family.c_str());
        QNN_INFO("  HTP Ver : V%d", htpVer);
        QNN_INFO("=========================");
        if (config.qnnSystemLibPath.empty() || config.qnnBackendPath.empty()) {
            QNN_ERROR("QNN system library and backend paths required for GPU mode");
            return false;
        }

        g_backendPathCmd = config.qnnBackendPath;
        dynamicloadutil::StatusCode sysStatus =
            dynamicloadutil::getQnnSystemFunctionPointers(config.qnnSystemLibPath,
                                                          &g_qnnSystemFuncs);
        if (sysStatus != dynamicloadutil::StatusCode::SUCCESS) {
            QNN_ERROR("Failed to get QNN system function pointers");
            return false;
        }

        // Apply patch to unet if needed
        if (!config.patchPath.empty()) {
            QNN_INFO("Applying patch to unet model in memory...");
            g_unetPatchedBuffer = qnn::tools::sample_app::applyZstdPatchToBuffer(
                unetPath, config.patchPath);
            if (!g_unetPatchedBuffer) {
                QNN_ERROR("Failed to apply patch to unet model buffer");
                return false;
            }
        }

        // Create QNN models
        if (!use_mnn_clip) {
            clipApp = createQnnModel(clipPath, "clip");
            if (!clipApp) { QNN_ERROR("Failed to create QNN CLIP model"); return false; }
        }

        unetApp = createQnnModel(unetPath, "unet");
        if (!unetApp) { QNN_ERROR("Failed to create QNN UNET model"); return false; }

        vaeDecoderApp = createQnnModel(vaeDecoderPath, "vae_decoder");
        if (!vaeDecoderApp) { QNN_ERROR("Failed to create QNN VAE Decoder"); return false; }

        if (!vaeEncoderPath.empty()) {
            vaeEncoderApp = createQnnModel(vaeEncoderPath, "vae_encoder");
            if (!vaeEncoderApp) QNN_WARN("Failed to create QNN VAE Encoder");
        }

        // Initialize QNN apps
        using namespace qnn::tools::sample_app;
        int status = EXIT_SUCCESS;
        if (!use_mnn_clip && clipApp) {
            status = initializeQnnApp("CLIP", clipApp);
            if (status != EXIT_SUCCESS) return false;
        }
        if (unetApp) {
            if (g_unetPatchedBuffer && g_unetPatchedBuffer->buffer) {
                status = initializeQnnApp(
                    "UNET", unetApp, g_unetPatchedBuffer->buffer.get(),
                    g_unetPatchedBuffer->size);
            } else {
                status = initializeQnnApp("UNET", unetApp);
            }
            if (status != EXIT_SUCCESS) return false;
            if (g_unetPatchedBuffer) {
                QNN_INFO("Releasing unet patch buffer to free memory");
                g_unetPatchedBuffer.reset();
            }
        }
        if (vaeDecoderApp) {
            status = initializeQnnApp("VAEDecoder", vaeDecoderApp);
            if (status != EXIT_SUCCESS) return false;
        }
        if (vaeEncoderApp) {
            status = initializeQnnApp("VAEEncoder", vaeEncoderApp);
            if (status != EXIT_SUCCESS) return false;
        }
    }

    QNN_INFO("All models initialized successfully");
    return true;
}

void recreateClipSession() {
    // Tear down existing CLIP session
    if (clipSession && clipInterpreter) {
        clipInterpreter->releaseSession(clipSession);
        clipSession = nullptr;
    }
    delete clipInterpreter;
    clipInterpreter = nullptr;

    if (!use_mnn_clip) {
        SD_LOG_WARN("[LORA] CLIP session recreation only supported in MNN/CPU mode");
        return;
    }

    // Recreate from (potentially LoRA-modified) file
    clipInterpreter = MNN::Interpreter::createFromFile(clipPath.c_str());
    if (!clipInterpreter) {
        SD_LOG_ERROR("[LORA] Failed to recreate MNN CLIP interpreter");
        return;
    }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;
    MNN::BackendConfig bkCfg;
    bkCfg.memory = MNN::BackendConfig::Memory_Low;
    bkCfg.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &bkCfg;

    clipSession = clipInterpreter->createSession(cfg);
    if (clipSession) {
        if (use_clip_v2) {
            auto input = clipInterpreter->getSessionInput(clipSession, "input_embedding");
            clipInterpreter->resizeTensor(input, {1, 77, 768});
        } else {
            auto input = clipInterpreter->getSessionInput(clipSession, "input_ids");
            clipInterpreter->resizeTensor(input, {1, 77});
        }
        clipInterpreter->resizeSession(clipSession);
        clipInterpreter->releaseModel();
    }

    // Reload pos_emb and token_emb if clip_v2 (LoRA regeneration rewrites these)
    if (use_clip_v2) {
        std::string posEmbPath = std::string(modelDir) + "/pos_emb.bin";
        std::string tokenEmbPath = std::string(modelDir) + "/token_emb.bin";

        std::ifstream posFile(posEmbPath, std::ios::binary);
        if (posFile.good()) {
            posFile.seekg(0, std::ios::end);
            size_t posSize = posFile.tellg() / sizeof(float);
            posFile.seekg(0, std::ios::beg);
            pos_emb.resize(posSize);
            posFile.read(reinterpret_cast<char*>(pos_emb.data()), posSize * sizeof(float));
        }

        std::ifstream tokenFile(tokenEmbPath, std::ios::binary);
        if (tokenFile.good()) {
            tokenFile.seekg(0, std::ios::end);
            size_t fileSize = tokenFile.tellg();
            tokenFile.seekg(0, std::ios::beg);
            size_t tokenSize = fileSize / sizeof(uint16_t);
            token_emb.resize(tokenSize);
            tokenFile.read(reinterpret_cast<char*>(token_emb.data()), fileSize);
        }
    }

    SD_LOG_INFO("[LORA] CLIP session recreated");
}

void recreateUNetSession() {
    // Just cleanup the persistent UNet runner — it will auto-recreate
    // on the next generation call via initIfNeeded()
    cleanup_persistent_sessions();
    SD_LOG_INFO("[LORA] UNet session invalidated (will recreate on next generation)");
}

void cleanup() {
    QNN_INFO("Cleaning up pipeline resources");

    // Perf 7: release persistent UNet runner + reset VAE dimension tracking
    cleanup_persistent_sessions();
    // Drop CLIP embedding cache so a new model never sees stale embeddings
    // from a different tokenizer / dimension.
    clear_clip_cache();
    // Drop the DiffEdit last-gen cache. Latents + embeds are model-specific
    // (different tokenizer + UNet weights → meaningless across loads).
    invalidate_edit_cache();

    if (clipSession && clipInterpreter) {
        clipInterpreter->releaseSession(clipSession);
        clipSession = nullptr;
    }
    if (unetSession && unetInterpreter) {
        unetInterpreter->releaseSession(unetSession);
        unetSession = nullptr;
    }
    if (safetyCheckerSession && safetyCheckerInterpreter) {
        safetyCheckerInterpreter->releaseSession(safetyCheckerSession);
        safetyCheckerSession = nullptr;
    }
    // Perf 7: Release persistent VAE sessions
    if (vaeDecoderSession && vaeDecoderInterpreter) {
        vaeDecoderInterpreter->releaseSession(vaeDecoderSession);
        vaeDecoderSession = nullptr;
    }
    if (vaeEncoderSession && vaeEncoderInterpreter) {
        vaeEncoderInterpreter->releaseSession(vaeEncoderSession);
        vaeEncoderSession = nullptr;
    }

    delete clipInterpreter;     clipInterpreter = nullptr;
    delete unetInterpreter;     unetInterpreter = nullptr;
    delete vaeDecoderInterpreter; vaeDecoderInterpreter = nullptr;
    delete vaeEncoderInterpreter; vaeEncoderInterpreter = nullptr;
    delete safetyCheckerInterpreter; safetyCheckerInterpreter = nullptr;

    clipApp.reset();
    unetApp.reset();
    vaeDecoderApp.reset();
    vaeEncoderApp.reset();
    upscalerApp.reset();

    tokenizer.reset();
    g_unetPatchedBuffer.reset();
    pos_emb.clear();
    token_emb.clear();

    QNN_INFO("Pipeline resources cleaned up");
}

// ============================================================================
// Standalone QNN upscaler load — mirrors LocalDream's per-request /upscale
// handler so the upscaler can be loaded without first loading a diffusion
// model.
// ============================================================================

bool ensureQnnSystemReady(const std::string& qnnSystemLibPath,
                           const std::string& qnnBackendPath) {
    using namespace qnn::tools;
    if (qnnSystemLibPath.empty() || qnnBackendPath.empty()) {
        QNN_ERROR("ensureQnnSystemReady: empty paths (system='%s' backend='%s')",
                  qnnSystemLibPath.c_str(), qnnBackendPath.c_str());
        return false;
    }
    if (!g_backendPathCmd.empty()) {
        // Already initialized by initialize_models() or a prior call.
        return true;
    }
    g_backendPathCmd = qnnBackendPath;
    dynamicloadutil::StatusCode sysStatus =
        dynamicloadutil::getQnnSystemFunctionPointers(qnnSystemLibPath,
                                                       &g_qnnSystemFuncs);
    if (sysStatus != dynamicloadutil::StatusCode::SUCCESS) {
        QNN_ERROR("ensureQnnSystemReady: getQnnSystemFunctionPointers failed");
        g_backendPathCmd.clear();
        return false;
    }
    QNN_INFO("ensureQnnSystemReady: QNN system funcs loaded from %s",
             qnnSystemLibPath.c_str());
    return true;
}

bool loadStandaloneQnnUpscaler(const std::string& modelPath) {
    if (g_backendPathCmd.empty()) {
        QNN_ERROR("loadStandaloneQnnUpscaler: QNN system not initialized; "
                  "call ensureQnnSystemReady() first");
        return false;
    }
    if (modelPath.empty()) {
        QNN_ERROR("loadStandaloneQnnUpscaler: empty modelPath");
        return false;
    }

    upscalerApp = createQnnModel(modelPath, "upscaler");
    if (!upscalerApp) {
        QNN_ERROR("loadStandaloneQnnUpscaler: createQnnModel failed for %s",
                  modelPath.c_str());
        return false;
    }

    int status = initializeQnnApp("Upscaler", upscalerApp,
                                  /*buffer=*/nullptr, /*bufferSize=*/0);
    if (status != EXIT_SUCCESS) {
        QNN_ERROR("loadStandaloneQnnUpscaler: initializeQnnApp failed "
                  "(status=%d)", status);
        upscalerApp.reset();
        return false;
    }

    QNN_INFO("loadStandaloneQnnUpscaler: upscaler ready at %s",
             modelPath.c_str());
    return true;
}

std::vector<Resolution> get_supported_resolutions(const std::string& modelDir,
                                                   int baseW, int baseH) {
    std::vector<Resolution> result;
    std::vector<std::pair<int,int>> seen;
    auto already_seen = [&](int w, int h) {
        for (auto& p : seen) if (p.first == w && p.second == h) return true;
        return false;
    };

    if (baseW > 0 && baseH > 0) {
        result.push_back({baseW, baseH});
        seen.push_back({baseW, baseH});
    }

    if (modelDir.empty()) return result;

    std::error_code ec;
    std::filesystem::path dir(modelDir);
    if (!std::filesystem::is_directory(dir, ec)) {
        QNN_WARN("get_supported_resolutions: not a directory: %s",
                 modelDir.c_str());
        return result;
    }

    auto parse_uint = [](const std::string& s) -> int {
        if (s.empty()) return -1;
        for (char c : s) if (c < '0' || c > '9') return -1;
        try { return std::stoi(s); } catch (...) { return -1; }
    };

    for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        std::string name = entry.path().filename().string();
        // Must end with ".patch"
        const std::string suffix = ".patch";
        if (name.size() <= suffix.size()) continue;
        if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        std::string stem = name.substr(0, name.size() - suffix.size());

        int w = -1, h = -1;
        auto x_pos = stem.find('x');
        if (x_pos == std::string::npos) {
            // Square: <N>.patch
            int n = parse_uint(stem);
            if (n > 0) { w = n; h = n; }
        } else {
            // Rectangular: <W>x<H>.patch
            int parsed_w = parse_uint(stem.substr(0, x_pos));
            int parsed_h = parse_uint(stem.substr(x_pos + 1));
            if (parsed_w > 0 && parsed_h > 0) { w = parsed_w; h = parsed_h; }
        }
        if (w < 0 || h < 0) continue;

        if (!already_seen(w, h)) {
            result.push_back({w, h});
            seen.push_back({w, h});
        }
    }

    std::sort(result.begin(), result.end(),
              [](const Resolution& a, const Resolution& b) {
        long la = static_cast<long>(a.width) * a.height;
        long lb = static_cast<long>(b.width) * b.height;
        if (la != lb) return la < lb;
        return a.width < b.width;
    });

    return result;
}

void releaseStandaloneQnnUpscaler() {
    if (upscalerApp) {
        upscalerApp.reset();
        QNN_INFO("releaseStandaloneQnnUpscaler: released");
    }
}

}  // namespace sd_pipeline
