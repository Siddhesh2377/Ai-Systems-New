#pragma once

/**
 * Diffusion Pipeline - Core inference code ported from xororz/local-dream
 *
 * This module contains the full Stable Diffusion inference pipeline:
 * - CLIP text encoding (QNN or MNN)
 * - UNet diffusion loop with classifier-free guidance
 * - VAE encoding/decoding (+ tiling for large images)
 * - Scheduler integration (DPM-Solver, Euler Ancestral)
 * - LoRA weight merging
 * - Safety checker
 * - Upscaling
 *
 * HTTP server code has been stripped. Progress is reported via callbacks
 * instead of SSE streams.
 */

#include "../state/diffusion_state.h"
#include <string>
#include <atomic>

namespace sd_pipeline {

/**
 * Initialize all model components from the given config.
 * Sets up QNN function pointers, loads tokenizer, creates MNN sessions,
 * and initializes QNN models.
 */
bool initialize_models(const SDModelConfig& config);

/**
 * Run a complete image generation (txt2img or img2img).
 * Progress is reported via the callback.
 * The stopFlag is checked each diffusion step for cancellation.
 */
SDGenerationResult run_generation(const SDGenerateParams& params,
                                  SDProgressCallback progressCb,
                                  std::atomic<bool>& stopFlag);

/**
 * Apply a LoRA to the currently loaded model.
 */
bool apply_lora(const std::string& path, float weight);

/**
 * Clear all applied LoRA weights.
 */
void clear_lora();

/**
 * Release all model resources and reset state.
 */
void cleanup();

/**
 * Get model information as a JSON string.
 */
std::string get_info();

} // namespace sd_pipeline
