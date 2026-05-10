#include "edit_cache.h"

#include "../utils/sd_logger.h"

namespace sd_pipeline {

LastGenCache& edit_cache() {
    static LastGenCache instance;
    return instance;
}

std::mutex& edit_cache_mutex() {
    static std::mutex m;
    return m;
}

void invalidate_edit_cache() {
    std::lock_guard<std::mutex> lock(edit_cache_mutex());
    auto& c = edit_cache();
    if (c.valid) {
        SD_LOG_INFO("[EDIT] Invalidating last-gen cache (was %dx%d, %zu B latent)",
                    c.output_w, c.output_h, c.latent.size() * sizeof(float));
    }
    c.latent.clear(); c.latent.shrink_to_fit();
    c.text_embedding.clear(); c.text_embedding.shrink_to_fit();
    c.original_prompt.clear();
    c.original_negative_prompt.clear();
    c.scheduler_type.clear();
    c.sample_w = c.sample_h = 0;
    c.output_w = c.output_h = 0;
    c.text_embedding_size = 0;
    c.cfg = 0.0f;
    c.steps = 0;
    c.seed = 0;
    c.valid = false;
}

bool edit_cache_matches(int sample_w, int sample_h) {
    std::lock_guard<std::mutex> lock(edit_cache_mutex());
    const auto& c = edit_cache();
    return c.valid && c.sample_w == sample_w && c.sample_h == sample_h;
}

}  // namespace sd_pipeline
