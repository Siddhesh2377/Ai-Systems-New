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

    if (StatusCode::SUCCESS != app->enablePerformaceMode())
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

    // Check for clip_v2 variant
    if (clipPath.length() >= 8 &&
        clipPath.substr(clipPath.length() - 8) == "clip.mnn") {
        std::filesystem::path clipPathObj(clipPath);
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
                promptProcessor.loadEmbeddings(embeddingsPath.string());
                QNN_INFO("Loaded %zu embeddings from %s",
                         promptProcessor.getEmbeddingCount(),
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

void cleanup() {
    QNN_INFO("Cleaning up pipeline resources");

    // Perf 7: release persistent UNet runner + reset VAE dimension tracking
    cleanup_persistent_sessions();

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

}  // namespace sd_pipeline
