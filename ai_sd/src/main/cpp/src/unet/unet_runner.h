#pragma once

/**
 * UNet Runner — encapsulates single-step UNet inference for both QNN and MNN.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.5).
 *
 * Usage:
 *   UNetRunner runner;
 *   runner.init(use_mnn, use_opencl, unetPath, modelDir, batch_size,
 *               sample_h, sample_w, text_emb_size);
 *   for each step:
 *     runner.step(latents_in, timestep, text_embeddings, unet_out);
 *   runner.cleanup();
 */

#include <memory>
#include <string>
#include <vector>

// Forward declarations
class QnnModel;
namespace MNN {
    class Interpreter;
    class Session;
}

class UNetRunner {
public:
    UNetRunner() = default;
    ~UNetRunner();

    // Non-copyable
    UNetRunner(const UNetRunner&) = delete;
    UNetRunner& operator=(const UNetRunner&) = delete;

    /**
     * Initialize UNet runner. For MNN mode, creates interpreter + session.
     * For QNN mode, validates that the global unetApp exists.
     *
     * @param use_mnn       Use MNN backend (true) or QNN backend (false)
     * @param use_opencl    Use OpenCL acceleration for MNN
     * @param unet_path     Path to UNet model file (MNN only)
     * @param model_dir     Model directory for cache files
     * @param batch_size    Batch size (typically 2 for CFG)
     * @param sample_h      Latent height
     * @param sample_w      Latent width
     * @param text_emb_size CLIP text embedding dimension
     */
    void init(bool use_mnn, bool use_opencl,
              const std::string& unet_path, const std::string& model_dir,
              int batch_size, int sample_h, int sample_w, int text_emb_size);

    /**
     * Perf 7: Reuse existing session if params match. Returns true if
     * session was already valid, false if re-initialized.
     */
    bool initIfNeeded(bool use_mnn, bool use_opencl,
                      const std::string& unet_path, const std::string& model_dir,
                      int batch_size, int sample_h, int sample_w, int text_emb_size);

    /**
     * Run one UNet denoising step.
     *
     * @param latents_in        Input latents [batch_size * 4 * sample_h * sample_w]
     * @param timestep          Current timestep value
     * @param text_embeddings   Text embeddings [batch_size * 77 * text_emb_size]
     * @param unet_out          Output noise prediction [batch_size * 4 * sample_h * sample_w]
     * @param cfg_scale         CFG scale; if 1.0f the QNN runner skips the uncond
     *                          pass and mirrors the cond output into the uncond
     *                          half so the CPU-side CFG combiner stays correct
     *                          (uc + 1.0*(tx-uc) == tx). MNN path is unaffected
     *                          (batch=2 is fixed at session creation).
     */
    void step(const float* latents_in, int timestep,
              const float* text_embeddings, float* unet_out,
              float cfg_scale = 7.0f);

    /// Release MNN session/interpreter. Safe to call multiple times.
    void cleanup();

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
    bool use_mnn_ = false;
    int batch_size_ = 2;
    int sample_h_ = 0;
    int sample_w_ = 0;
    int text_emb_size_ = 0;
    int single_latent_size_ = 0;

    // MNN state (only used when use_mnn_ == true)
    MNN::Interpreter* mnn_interpreter_ = nullptr;
    MNN::Session* mnn_session_ = nullptr;

    // Perf 7: track init params for session reuse
    bool use_opencl_ = false;
    std::string unet_path_;
};
