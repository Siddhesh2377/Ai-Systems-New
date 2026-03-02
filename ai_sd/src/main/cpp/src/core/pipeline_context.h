#pragma once

/**
 * PipelineContext — Owning struct for all Stable Diffusion pipeline state.
 *
 * Replaces the 50+ file-scope globals in diffusion_pipeline.cpp.
 * Organized into sections: Models, Config, Buffers, GenParams.
 *
 * Migration strategy:
 *   1. Context created here, instantiated in DiffusionState::Impl
 *   2. initialize_models() populates Models + Config sections
 *   3. run_generation() populates GenParams before calling generateImage()
 *   4. As modules are extracted (Phase 1 steps 3-13), each module takes
 *      PipelineContext& instead of reading globals
 *   5. Once all modules migrated, file-scope globals are deleted
 *
 * Note: QnnModel, PromptProcessor, MNN types are raw pointers because
 * the file-scope globals in diffusion_pipeline.cpp still own these objects
 * during migration. They become unique_ptr when ownership transfers fully.
 */

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations — full headers only in .cpp files
class QnnModel;
class PromptProcessor;
namespace MNN {
    class Interpreter;
    class Session;
}
namespace tokenizers {
    class Tokenizer;
}

struct PipelineContext {

    // =========================================================================
    // Models — set by model_loader, read by pipeline modules
    // =========================================================================
    struct Models {
        // QNN models (NPU path) — raw pointers, globals own during migration
        QnnModel* clip = nullptr;
        QnnModel* unet = nullptr;
        QnnModel* vae_decoder = nullptr;
        QnnModel* vae_encoder = nullptr;
        QnnModel* upscaler = nullptr;

        // MNN models (CPU/GPU path) — Interpreter owns model, Session owns compute graph
        MNN::Interpreter* mnn_clip = nullptr;
        MNN::Interpreter* mnn_unet = nullptr;
        MNN::Interpreter* mnn_vae_dec = nullptr;
        MNN::Interpreter* mnn_vae_enc = nullptr;
        MNN::Interpreter* mnn_safety = nullptr;

        MNN::Session* mnn_clip_session = nullptr;
        MNN::Session* mnn_unet_session = nullptr;
        MNN::Session* mnn_vae_dec_session = nullptr;
        MNN::Session* mnn_vae_enc_session = nullptr;
        MNN::Session* mnn_safety_session = nullptr;

        // Tokenizer + prompt processing — raw pointers, globals own during migration
        tokenizers::Tokenizer* tokenizer = nullptr;
        PromptProcessor* prompt_processor = nullptr;
        std::vector<float> pos_emb;
        std::vector<uint16_t> token_emb;  // FP16 to save memory

        // QNN infrastructure
        std::string qnn_backend_path;

        // Patched model buffer (for unet zstd patching)
        struct PatchedBuffer {
            std::shared_ptr<uint8_t> data;
            uint64_t size = 0;
            void reset() { data.reset(); size = 0; }
        };
        PatchedBuffer unet_patched;
    } models;

    // =========================================================================
    // Config — set once at load, immutable during generation
    // =========================================================================
    struct Config {
        // Backend selection
        bool use_mnn = false;            // true = MNN (CPU/GPU), false = QNN (NPU)
        bool use_mnn_clip = false;       // true = use MNN for CLIP even in QNN mode
        bool use_clip_v2 = false;        // CLIP v2 variant detected
        bool pony = false;               // Pony v5.5 model
        bool use_safety_checker = false;
        float nsfw_threshold = 0.5f;

        // Model dimensions
        int text_embedding_size = 768;

        // Paths (stored for get_info and potential reload)
        std::string model_dir;
        std::string clip_path;
        std::string unet_path;
        std::string vae_dec_path;
        std::string vae_enc_path;
        std::string safety_path;
        std::string tokenizer_path;
        std::string patch_path;
        std::string upscaler_path;
    } config;

    // =========================================================================
    // Buffers — created once, reused across generations (Phase 3 optimization)
    // =========================================================================
    struct Buffers {
        // Pre-allocated for UNet loop (eliminates per-step heap alloc)
        std::vector<float> latents_in;     // 2 * single_latent_size
        std::vector<float> unet_out;       // 2 * single_latent_size
        std::vector<float> noise_pred;     // single_latent_size
        std::vector<float> cfg_result;     // single_latent_size

        // Output
        std::vector<uint8_t> rgb_output;   // width * height * 3

        // QNN reuse buffers
        std::vector<float> qnn_float_buf;  // for convertToFloat reuse

        // Quantized CLIP cache (QNN mode — avoid re-quantizing every step)
        std::vector<uint16_t> clip_quant_cache;
        bool clip_quant_cached = false;

        // Resize buffers for a given latent/output size
        void resize_for_generation(int sample_w, int sample_h, int out_w, int out_h) {
            int latent_size = 4 * sample_w * sample_h;
            latents_in.resize(2 * latent_size);
            unet_out.resize(2 * latent_size);
            noise_pred.resize(latent_size);
            cfg_result.resize(latent_size);
            rgb_output.resize(out_w * out_h * 3);
            clip_quant_cached = false;
        }

        void clear() {
            latents_in.clear(); latents_in.shrink_to_fit();
            unet_out.clear(); unet_out.shrink_to_fit();
            noise_pred.clear(); noise_pred.shrink_to_fit();
            cfg_result.clear(); cfg_result.shrink_to_fit();
            rgb_output.clear(); rgb_output.shrink_to_fit();
            qnn_float_buf.clear(); qnn_float_buf.shrink_to_fit();
            clip_quant_cache.clear(); clip_quant_cache.shrink_to_fit();
            clip_quant_cached = false;
        }
    } buffers;

    // =========================================================================
    // GenParams — set per generation call, read by pipeline
    // =========================================================================
    struct GenParams {
        std::string prompt;
        std::string negative_prompt;
        int steps = 28;
        float cfg = 7.0f;
        unsigned int seed = 0;
        int width = 512;
        int height = 512;
        int sample_w = 64;    // width / 8
        int sample_h = 64;    // height / 8
        std::string scheduler_type = "dpm";
        bool use_opencl = false;

        // Img2Img
        bool img2img = false;
        bool has_mask = false;
        float denoise_strength = 0.6f;
        std::vector<float> img_data;
        std::vector<float> mask_data;
        std::vector<float> mask_data_full;

        // Process visualization
        bool show_process = false;
        int show_stride = 1;

        void clear() {
            img_data.clear(); img_data.shrink_to_fit();
            mask_data.clear(); mask_data.shrink_to_fit();
            mask_data_full.clear(); mask_data_full.shrink_to_fit();
        }
    } gen;

    // =========================================================================
    // Lifecycle
    // =========================================================================
    bool is_loaded() const { return models.tokenizer != nullptr; }

    // Defined in diffusion_pipeline.cpp (needs full MNN types)
    void release();
};
