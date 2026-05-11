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
#include <vector>

// Forward declarations
struct SDModelConfig;
class QnnModel;

namespace sd_pipeline {

/// Load all models (tokenizer, CLIP, UNet, VAE, safety checker, QNN/MNN)
/// based on config. Returns true on success.
bool initialize_models(const SDModelConfig& config);

/// Release all loaded models and sessions.
void cleanup();

/// Populate the QNN system function pointers + backend path from the
/// provided absolute paths. Idempotent: returns true if already set.
/// Must be called before loadStandaloneQnnUpscaler() if no diffusion
/// model has been loaded yet.
bool ensureQnnSystemReady(const std::string& qnnSystemLibPath,
                           const std::string& qnnBackendPath);

/// Build + initialize a standalone QNN upscaler from a .bin file, writing
/// the result into the global `upscalerApp` so subsequent calls to
/// `upscaleImageWithModel(..., upscalerApp)` succeed. Mirrors the
/// per-request load that LocalDream's `/upscale` HTTP handler performs.
/// Caller must have invoked ensureQnnSystemReady() first (either directly
/// or via initialize_models()).
bool loadStandaloneQnnUpscaler(const std::string& modelPath);

/// Release the standalone upscaler if loaded.
void releaseStandaloneQnnUpscaler();

/// (width, height) pair returned by get_supported_resolutions.
struct Resolution {
    int width;
    int height;
};

/// Scan `modelDir` for `.patch` files and combine with the base
/// resolution to enumerate every (width, height) the loaded UNet binary
/// can run at. Patch files follow xororz / Mr-J-369 naming:
///   `<N>.patch`         — square resolution N×N
///   `<W>x<H>.patch`     — rectangular WxH
/// `baseW`/`baseH` are always included (whatever the unet.bin was
/// compiled for, typically 512×512 for SD-1.5 `min` variants). Result
/// is sorted by total pixel count ascending and de-duplicated.
/// Independent of model load state — pure filesystem scan, safe to call
/// before `initialize_models`.
std::vector<Resolution> get_supported_resolutions(const std::string& modelDir,
                                                   int baseW, int baseH);

}  // namespace sd_pipeline
