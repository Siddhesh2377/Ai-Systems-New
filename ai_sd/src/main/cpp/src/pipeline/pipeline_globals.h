#pragma once

/**
 * Pipeline globals — extern declarations for the 50+ file-scope variables
 * in diffusion_pipeline.cpp.
 *
 * MIGRATION ARTIFACT: This header exists only during the Phase 1 migration.
 * As modules are extracted and switch to reading PipelineContext&, the globals
 * they use will be removed. When all globals are gone, this file is deleted.
 *
 * Definitions live in diffusion_pipeline.cpp (unchanged).
 */

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
class QnnModel;
class PromptProcessor;
namespace MNN {
    class Interpreter;
    class Session;
}
namespace tokenizers {
    class Tokenizer;
}
// ============================================================================
// Model config flags
// ============================================================================
extern bool ponyv55;
extern bool use_mnn;
extern bool use_safety_checker;
extern bool use_mnn_clip;
extern bool use_clip_v2;
extern bool upscaler_mode;
extern float nsfw_threshold;

// ============================================================================
// Model paths
// ============================================================================
extern std::string clipPath, unetPath, vaeDecoderPath, vaeEncoderPath,
    safetyCheckerPath, tokenizerPath, patchPath, modelDir, upscalerPath;

// ============================================================================
// Model objects
// ============================================================================
extern std::vector<float> pos_emb;
extern std::vector<uint16_t> token_emb;
extern std::shared_ptr<tokenizers::Tokenizer> tokenizer;
extern PromptProcessor promptProcessor;
extern std::unique_ptr<QnnModel> clipApp;
extern std::unique_ptr<QnnModel> unetApp;
extern std::unique_ptr<QnnModel> vaeDecoderApp;
extern std::unique_ptr<QnnModel> vaeEncoderApp;
extern std::unique_ptr<QnnModel> upscalerApp;

// MNN Interpreter + Session pairs
extern MNN::Interpreter *clipInterpreter;
extern MNN::Interpreter *unetInterpreter;
extern MNN::Interpreter *vaeDecoderInterpreter;
extern MNN::Interpreter *vaeEncoderInterpreter;
extern MNN::Interpreter *safetyCheckerInterpreter;
extern MNN::Session *clipSession;
extern MNN::Session *unetSession;
extern MNN::Session *vaeDecoderSession;
extern MNN::Session *vaeEncoderSession;
extern MNN::Session *safetyCheckerSession;

// ============================================================================
// Generation params (set per run_generation call)
// ============================================================================
extern std::string prompt;
extern std::string negative_prompt;
extern int steps;
extern float cfg;
extern unsigned seed;
extern std::string scheduler_type;
extern std::vector<float> img_data;
extern std::vector<float> mask_data;
extern std::vector<float> mask_data_full;
extern float denoise_strength;
extern bool request_img2img;
extern bool request_has_mask;
extern bool use_opencl;

// ============================================================================
// Misc state
// ============================================================================
extern bool cvt_model;
extern bool show_diffusion_process;
extern int show_diffusion_stride;

// PatchedModelBuffer, g_unetPatchedBuffer, g_qnnSystemFuncs, g_backendPathCmd
// are local to model_loader.cpp (only used by initialize_models/cleanup)
extern std::string model_dir;
extern bool clip_skip_2;

// Note: text_embedding_size, sample_width/height, output_width/height
// are inline globals in config.h — no extern needed.

// Perf 7: cleanup persistent MNN sessions (UNet runner, VAE dimension tracking)
// Defined in pipeline_orchestrator.cpp, called by model_loader cleanup()
namespace sd_pipeline {
    void cleanup_persistent_sessions();
    void recreateClipSession();
    void recreateUNetSession();
    /// Clear cached CLIP text embeddings. Call on model load/release/swap so a
    /// new model never sees a stale embedding from a different tokenizer.
    void clear_clip_cache();
}
