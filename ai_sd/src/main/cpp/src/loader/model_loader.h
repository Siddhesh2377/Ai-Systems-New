#pragma once

/**
 * Model Loader — loads and releases all SD pipeline models.
 *
 * Extracted from diffusion_pipeline.cpp Phase 1.2.
 * Functions write to file-scope globals during migration.
 * When migration completes, they will write to PipelineContext& instead.
 */

#include <string>
#include <memory>

// Forward declarations
struct SDModelConfig;
class QnnModel;

namespace sd_pipeline {

/// Load all models (tokenizer, CLIP, UNet, VAE, safety checker, QNN/MNN)
/// based on config. Returns true on success.
bool initialize_models(const SDModelConfig& config);

/// Release all loaded models and sessions.
void cleanup();

}  // namespace sd_pipeline
