/**
 * LoRA Engine implementation.
 *
 * Orchestrates runtime LoRA application by:
 * 1. Backing up original .mnn.weight files
 * 2. Regenerating weights via generateClipModel() / generateModel() / patchModel()
 * 3. Restoring backups on clear
 *
 * All weight math is delegated to the existing applyLoRA() in safetensor_to_mnn.h.
 */

#include "lora_engine.h"
#include "../utils/safetensor_to_mnn.h"
#include "../utils/sd_logger.h"
#include "../model/sd_structure.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// ============================================================================
// Public API
// ============================================================================

bool LoRAEngine::apply(const std::string& lora_path, float weight,
                       const std::string& model_dir, bool use_clip_v2) {
    // Validate LoRA file
    if (!validate_lora_file(lora_path)) {
        SD_LOG_ERROR("[LORA] Invalid LoRA file: %s", lora_path.c_str());
        return false;
    }

    // Find base safetensors
    std::string safetensor = find_safetensor(model_dir);
    if (safetensor.empty()) {
        SD_LOG_ERROR("[LORA] No .safetensors base model found in: %s", model_dir.c_str());
        return false;
    }

    // Backup originals on first apply
    if (!originals_backed_up_) {
        if (!backup_originals(model_dir, use_clip_v2)) {
            SD_LOG_ERROR("[LORA] Failed to backup original weight files");
            return false;
        }
    }

    // Add to active stack
    active_loras_.push_back({lora_path, weight});

    // Regenerate all weights with combined LoRA stack
    if (!regenerate_weights(model_dir, use_clip_v2)) {
        SD_LOG_ERROR("[LORA] Weight regeneration failed, restoring originals");
        active_loras_.pop_back();
        if (active_loras_.empty()) {
            restore_originals(model_dir, use_clip_v2);
        } else {
            // Re-regenerate with previous LoRA stack
            regenerate_weights(model_dir, use_clip_v2);
        }
        return false;
    }

    SD_LOG_INFO("[LORA] Applied: %s (weight=%.2f), %zu active LoRA(s)",
                lora_path.c_str(), weight, active_loras_.size());
    return true;
}

bool LoRAEngine::clear(const std::string& model_dir, bool use_clip_v2) {
    if (active_loras_.empty()) return true;

    active_loras_.clear();

    if (originals_backed_up_) {
        if (!restore_originals(model_dir, use_clip_v2)) {
            SD_LOG_ERROR("[LORA] Failed to restore original weights");
            return false;
        }
        originals_backed_up_ = false;
    }

    SD_LOG_INFO("[LORA] All LoRAs cleared, original weights restored");
    return true;
}

void LoRAEngine::reset() {
    active_loras_.clear();
    originals_backed_up_ = false;
}

// ============================================================================
// Private helpers
// ============================================================================

bool LoRAEngine::backup_originals(const std::string& model_dir, bool use_clip_v2) {
    try {
        std::string clip_name = use_clip_v2 ? "clip_v2" : "clip";

        // Weight files to backup
        std::vector<std::string> files = {
            clip_name + ".mnn.weight",
            "unet.mnn.weight",
            "unet.mnn",               // patchModel() modifies the structure file too
        };

        // Also backup pos_emb.bin and token_emb.bin (generateClipModel rewrites these)
        if (use_clip_v2) {
            files.push_back("pos_emb.bin");
            files.push_back("token_emb.bin");
        }

        for (const auto& file : files) {
            fs::path src = fs::path(model_dir) / file;
            fs::path dst = fs::path(model_dir) / (file + ".orig");
            if (fs::exists(src)) {
                fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
                SD_LOG_DEBUG("[LORA] Backed up: %s", file.c_str());
            }
        }

        originals_backed_up_ = true;
        SD_LOG_INFO("[LORA] Original weight files backed up");
        return true;
    } catch (const std::exception& e) {
        SD_LOG_ERROR("[LORA] Backup failed: %s", e.what());
        return false;
    }
}

bool LoRAEngine::restore_originals(const std::string& model_dir, bool use_clip_v2) {
    try {
        std::string clip_name = use_clip_v2 ? "clip_v2" : "clip";

        std::vector<std::string> files = {
            clip_name + ".mnn.weight",
            "unet.mnn.weight",
            "unet.mnn",
        };

        if (use_clip_v2) {
            files.push_back("pos_emb.bin");
            files.push_back("token_emb.bin");
        }

        for (const auto& file : files) {
            fs::path src = fs::path(model_dir) / (file + ".orig");
            fs::path dst = fs::path(model_dir) / file;
            if (fs::exists(src)) {
                fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
                SD_LOG_DEBUG("[LORA] Restored: %s", file.c_str());
            }
        }

        SD_LOG_INFO("[LORA] Original weight files restored");
        return true;
    } catch (const std::exception& e) {
        SD_LOG_ERROR("[LORA] Restore failed: %s", e.what());
        return false;
    }
}

bool LoRAEngine::regenerate_weights(const std::string& model_dir, bool use_clip_v2) {
    std::string safetensor = find_safetensor(model_dir);
    if (safetensor.empty()) return false;

    // Build LoRA file list + weight vector
    std::vector<std::string> lora_files;
    std::vector<float> lora_weights;
    for (const auto& lora : active_loras_) {
        lora_files.push_back(lora.path);
        lora_weights.push_back(lora.weight);
    }

    // The safetensor filename relative to model_dir
    std::string safetensor_filename = fs::path(safetensor).filename().string();

    try {
        SD_LOG_INFO("[LORA] Regenerating CLIP weights with %zu LoRA(s)...",
                    active_loras_.size());
        generateClipModel(model_dir, safetensor_filename, use_clip_v2,
                          lora_files, lora_weights);

        SD_LOG_INFO("[LORA] Regenerating UNet weights with %zu LoRA(s)...",
                    active_loras_.size());
        generateModel(model_dir, safetensor_filename, "unet", unet_structure,
                      lora_files, lora_weights);
        patchModel(model_dir, safetensor_filename, "unet", unet_small_weights);

        SD_LOG_INFO("[LORA] Weight regeneration complete");
        return true;
    } catch (const std::exception& e) {
        SD_LOG_ERROR("[LORA] Weight regeneration failed: %s", e.what());
        return false;
    }
}

std::string LoRAEngine::find_safetensor(const std::string& model_dir) {
    try {
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (entry.path().extension() == ".safetensors") {
                return entry.path().string();
            }
        }
    } catch (const std::exception& e) {
        SD_LOG_ERROR("[LORA] Error scanning model dir: %s", e.what());
    }
    return "";
}

bool LoRAEngine::validate_lora_file(const std::string& path) {
    if (!fs::exists(path)) {
        SD_LOG_ERROR("[LORA] File not found: %s", path.c_str());
        return false;
    }

    // Check file extension
    if (fs::path(path).extension() != ".safetensors") {
        SD_LOG_ERROR("[LORA] Not a .safetensors file: %s", path.c_str());
        return false;
    }

    // Validate it's a readable SafeTensor with LoRA keys
    try {
        SafeTensorReader reader(path);
        auto names = reader.get_tensor_names();

        bool has_lora_keys = false;
        for (const auto& name : names) {
            if (name.find("lora_down") != std::string::npos ||
                name.find("lora_up") != std::string::npos) {
                has_lora_keys = true;
                break;
            }
        }

        if (!has_lora_keys) {
            SD_LOG_ERROR("[LORA] No LoRA keys found in: %s", path.c_str());
            return false;
        }

        SD_LOG_INFO("[LORA] Validated: %s (%d tensors)",
                    path.c_str(), reader.get_tensor_count());
        return true;
    } catch (const std::exception& e) {
        SD_LOG_ERROR("[LORA] Validation failed for %s: %s", path.c_str(), e.what());
        return false;
    }
}
