/**
 * DiffusionState implementation
 *
 * This is a thin wrapper that delegates to the full inference pipeline
 * (ported from xororz's main.cpp). The actual inference code lives in
 * pipeline/diffusion_pipeline.cpp.
 *
 * This file handles resource lifecycle (load/release) and translates
 * between the clean DiffusionState API and the global-state-based
 * inference pipeline.
 */

#include "diffusion_state.h"
#include "../utils/logger.h"

#include <chrono>
#include <sstream>

// Forward-declare pipeline functions (defined in diffusion_pipeline.cpp)
namespace sd_pipeline {
    bool initialize_models(const SDModelConfig& config);
    SDGenerationResult run_generation(const SDGenerateParams& params,
                                      SDProgressCallback progressCb,
                                      std::atomic<bool>& stopFlag);
    bool apply_lora(const std::string& path, float weight);
    void clear_lora();
    void cleanup();
    std::string get_info();
}

struct DiffusionState::Impl {
    // Currently empty - all state is in the pipeline globals
    // (matching xororz's architecture). Refactoring to instance-based
    // state is a future improvement.
};

DiffusionState::DiffusionState() = default;

DiffusionState::~DiffusionState() {
    release();
}

bool DiffusionState::load_models(const SDModelConfig& config) {
    if (m_ready) {
        SD_LOG_WARN("Models already loaded, releasing first");
        release();
    }

    m_config = config;
    m_impl = std::make_unique<Impl>();

    SD_LOG_INFO("Loading models from: %s", config.modelDir.c_str());

    if (!sd_pipeline::initialize_models(config)) {
        SD_LOG_ERROR("Failed to initialize models");
        m_impl.reset();
        return false;
    }

    m_ready = true;
    SD_LOG_INFO("Models loaded successfully");
    return true;
}

SDGenerationResult DiffusionState::generate(const SDGenerateParams& params,
                                            SDProgressCallback progressCb,
                                            std::atomic<bool>& stopFlag) {
    if (!m_ready) {
        SD_LOG_ERROR("Cannot generate: models not loaded");
        return {};
    }

    return sd_pipeline::run_generation(params, std::move(progressCb), stopFlag);
}

bool DiffusionState::apply_lora(const std::string& loraPath, float weight) {
    if (!m_ready) {
        SD_LOG_ERROR("Cannot apply LoRA: models not loaded");
        return false;
    }
    return sd_pipeline::apply_lora(loraPath, weight);
}

void DiffusionState::clear_lora() {
    if (m_ready) {
        sd_pipeline::clear_lora();
    }
}

void DiffusionState::release() {
    if (m_ready) {
        SD_LOG_INFO("Releasing diffusion models");
        sd_pipeline::cleanup();
        m_ready = false;
    }
    m_impl.reset();
}

std::string DiffusionState::get_model_info() const {
    if (!m_ready) return "{}";
    return sd_pipeline::get_info();
}
