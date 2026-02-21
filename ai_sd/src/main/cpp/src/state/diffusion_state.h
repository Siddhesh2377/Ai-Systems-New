#pragma once

/**
 * Global state manager for Stable Diffusion inference pipeline.
 *
 * Holds all model resources (QNN models, MNN interpreters, tokenizer)
 * and provides load/release/generate operations.
 *
 * Thread safety: callers must hold appropriate mutexes (g_sd_init_mtx
 * for load/release, g_sd_generate_mtx for generate). See ai_sd_jni.cpp.
 */

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>

// Forward declarations - actual QNN/MNN types included in .cpp
namespace MNN {
    class Interpreter;
    class Session;
}

/**
 * Configuration passed to load_models()
 */
struct SDModelConfig {
    // Model file paths
    std::string clipPath;
    std::string unetPath;
    std::string vaeDecoderPath;
    std::string vaeEncoderPath;
    std::string tokenizerPath;
    std::string safetyCheckerPath;
    std::string upscalerPath;
    std::string patchPath;
    std::string modelDir;

    // QNN backend paths
    std::string qnnBackendPath;       // libQnnHtp.so
    std::string qnnSystemLibPath;     // libQnnSystem.so

    // Flags
    int textEmbeddingSize = 768;
    bool runOnCpu = false;
    bool useCpuClip = false;
    bool isPony = false;
    bool useSafetyChecker = false;
    float nsfwThreshold = 0.5f;
};

/**
 * Parameters for a single generation call
 */
struct SDGenerateParams {
    std::string prompt;
    std::string negativePrompt;
    int steps = 28;
    float cfgScale = 7.0f;
    unsigned int seed = 0;
    int width = 512;
    int height = 512;
    std::string scheduler = "dpm";
    bool useOpenCL = false;

    // Img2Img
    std::vector<float> inputImage;       // Preprocessed NCHW float data
    std::vector<float> mask;             // Latent-space mask
    std::vector<float> maskFull;         // Full-resolution mask
    float denoiseStrength = 0.6f;
    bool isImg2Img = false;
    bool hasMask = false;

    // Process visualization
    bool showDiffusionProcess = false;
    int showDiffusionStride = 1;
};

/**
 * Result from a generation call
 */
struct SDGenerationResult {
    std::vector<uint8_t> imageData;  // RGB byte data
    int width = 0;
    int height = 0;
    int channels = 3;
    int generationTimeMs = 0;
    int firstStepTimeMs = 0;
    unsigned int seed = 0;           // Actual seed used
};

/**
 * Callback for progress updates during generation
 */
using SDProgressCallback = std::function<void(int step, int totalSteps,
                                              const uint8_t* imageData, int imageDataLen,
                                              int width, int height)>;

class DiffusionState {
public:
    DiffusionState();
    ~DiffusionState();

    // Non-copyable
    DiffusionState(const DiffusionState&) = delete;
    DiffusionState& operator=(const DiffusionState&) = delete;

    /**
     * Load all model components from config paths
     * @return true on success
     */
    bool load_models(const SDModelConfig& config);

    /**
     * Run image generation with the loaded models
     * @param params Generation parameters
     * @param progressCb Callback for step progress and intermediate images
     * @param stopFlag Atomic flag checked each step to allow cancellation
     * @return Generation result with RGB image data
     */
    SDGenerationResult generate(const SDGenerateParams& params,
                                SDProgressCallback progressCb,
                                std::atomic<bool>& stopFlag);

    /**
     * Apply a LoRA to the current model
     * @param loraPath Path to the LoRA safetensors file
     * @param weight LoRA weight multiplier
     * @return true on success
     */
    bool apply_lora(const std::string& loraPath, float weight);

    /**
     * Clear any applied LoRA weights
     */
    void clear_lora();

    /**
     * Release all resources
     */
    void release();

    /**
     * Check if models are loaded and ready
     */
    bool is_ready() const { return m_ready; }

    /**
     * Get model info as JSON string
     */
    std::string get_model_info() const;

private:
    bool m_ready = false;
    SDModelConfig m_config;

    // Resource handles are opaque here - implemented in diffusion_state.cpp
    // with full QNN/MNN includes
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
