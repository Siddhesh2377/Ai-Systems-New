#pragma once

/**
 * Safety checker — NSFW detection using MNN classifier.
 *
 * Extracted from sd_utils.h (Phase 1.8).
 * Takes RGB image data, runs VGG-based classifier, returns NSFW score.
 */

#include <cstdint>
#include <vector>

namespace MNN {
class Interpreter;
class Session;
}

/// Run NSFW safety check on image data.
/// Returns true on success (score written to nsfw_score), false on error.
bool safety_check(const std::vector<uint8_t>& image_data, int width, int height,
                  float& nsfw_score, MNN::Interpreter* interpreter,
                  MNN::Session* session);
