#pragma once

/**
 * LoRA Engine — runtime LoRA application for MNN-based Stable Diffusion.
 *
 * Manages active LoRA stack, weight file backups, and model regeneration.
 * Uses the existing SafeTensor->MNN infrastructure (safetensor_to_mnn.h)
 * to regenerate .mnn.weight files with LoRA deltas applied.
 *
 * MNN-only: QNN uses pre-compiled binary contexts where weights are baked in.
 *
 * Thread safety: callers must hold the pipeline mutex (g_sd_mtx).
 */

#include <string>
#include <vector>

namespace sd_pipeline {

class LoRAEngine {
public:
    struct ActiveLoRA {
        std::string path;   // Absolute path to .safetensors LoRA file
        float weight;       // LoRA strength multiplier (typically 0.5-1.5)
    };

    /**
     * Apply a LoRA to the current model weights.
     *
     * On first call, backs up original .mnn.weight files.
     * Regenerates CLIP + UNet weights from base .safetensors + all active LoRAs.
     *
     * @param lora_path  Absolute path to LoRA .safetensors file
     * @param weight     LoRA strength multiplier
     * @param model_dir  Model directory containing .mnn + .safetensors files
     * @param use_clip_v2  Whether CLIP v2 is in use
     * @return true on success
     */
    bool apply(const std::string& lora_path, float weight,
               const std::string& model_dir, bool use_clip_v2);

    /**
     * Remove all active LoRAs and restore original weights.
     *
     * @param model_dir  Model directory
     * @param use_clip_v2  Whether CLIP v2 is in use
     * @return true on success
     */
    bool clear(const std::string& model_dir, bool use_clip_v2);

    /** Check if any LoRAs are currently applied. */
    bool has_active() const { return !active_loras_.empty(); }

    /** Get the list of currently active LoRAs. */
    const std::vector<ActiveLoRA>& active_loras() const { return active_loras_; }

    /** Reset state (called on model release). */
    void reset();

private:
    std::vector<ActiveLoRA> active_loras_;
    bool originals_backed_up_ = false;

    /** Back up original weight files before first LoRA application. */
    bool backup_originals(const std::string& model_dir, bool use_clip_v2);

    /** Restore original weight files from backups. */
    bool restore_originals(const std::string& model_dir, bool use_clip_v2);

    /**
     * Regenerate CLIP + UNet weights with all active LoRAs applied.
     * Delegates to generateClipModel() / generateModel() / patchModel()
     * from safetensor_to_mnn.h.
     */
    bool regenerate_weights(const std::string& model_dir, bool use_clip_v2);

    /** Find the base .safetensors file in model_dir. */
    static std::string find_safetensor(const std::string& model_dir);

    /** Validate that a LoRA file is a readable SafeTensor with LoRA keys. */
    static bool validate_lora_file(const std::string& path);
};

} // namespace sd_pipeline
