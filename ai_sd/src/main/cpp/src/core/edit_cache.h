#pragma once

/**
 * Last-Generation Cache for the DiffEdit text-driven edit feature.
 *
 * After every successful generation, the orchestrator copies the final
 * (clean, pre-VAE-scale) latent and the cond/uncond CLIP embeddings into
 * this cache. A subsequent edit call can then re-noise the cached latent
 * at an intermediate timestep and run the two-pass mask-detection +
 * inpaint loop without paying the cost of regenerating from scratch.
 *
 * Memory cost (SD 1.5 at 512²): ~64 KB latent + 2 × 240 KB embeddings ≈
 * 540 KB. Negligible.
 *
 * Lifecycle:
 *   - Written: at the end of generateImage(), just before VAE scaling.
 *   - Read:    by run_diff_edit() before the mask-detection passes.
 *   - Invalidated: on cleanup() / model swap / explicit clear.
 *
 * Note: this is a file-scope singleton during the migration to
 * PipelineContext. Once PC fully owns generation state it should move
 * onto the context as a `LastGen` substruct.
 */

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace sd_pipeline {

struct LastGenCache {
    // Final clean latent in NCHW layout: 1 × 4 × sample_h × sample_w (fp32).
    std::vector<float> latent;

    // CLIP-encoded conditioning. Layout matches text_embedding_float in
    // pipeline_orchestrator: [uncond | cond], total 2 × 77 × text_emb_dim
    // (fp32). Stored as a single buffer; uncond is the first half, cond
    // the second half — matches what the QNN/MNN UNet runner expects.
    std::vector<float> text_embedding;

    // Geometry needed to re-allocate buffers on the edit path. sample_w/h
    // are the latent dims (output_w/h ÷ 8). text_embedding_size is the
    // CLIP dim (768 for SD 1.5, 1024 for SD 2.x, 2048 for SDXL pooled).
    int sample_w = 0;
    int sample_h = 0;
    int output_w = 0;
    int output_h = 0;
    int text_embedding_size = 0;

    // Generation params worth remembering for an edit (so the user can
    // hold the same scheduler / cfg or override).
    std::string scheduler_type;
    float cfg = 7.0f;
    int steps = 28;
    uint64_t seed = 0;

    // Original prompt text — useful so the mask-detection pass can
    // re-encode it if the cached embedding is for any reason discarded
    // (e.g. CLIP cache eviction).
    std::string original_prompt;
    std::string original_negative_prompt;

    bool valid = false;
};

/// Process-global cache instance. Guarded by edit_cache_mutex().
LastGenCache& edit_cache();

/// Mutex protecting edit_cache(). Both writers (orchestrator post-gen)
/// and readers (DiffEdit entry point) must hold this around access.
std::mutex& edit_cache_mutex();

/// Drop the cache. Called from cleanup() / on model swap.
void invalidate_edit_cache();

/// True if the cache is populated AND the requested geometry matches.
/// Used by run_diff_edit to fail fast with a clear error rather than
/// silently producing noise on a dimension mismatch.
bool edit_cache_matches(int sample_w, int sample_h);

}  // namespace sd_pipeline
