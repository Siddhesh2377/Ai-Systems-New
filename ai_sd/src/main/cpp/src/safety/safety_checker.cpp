/**
 * Safety checker implementation — NSFW detection using MNN classifier.
 *
 * Extracted from sd_utils.h (Phase 1.8).
 * Resize to 256x256 -> center crop 224x224 -> VGG preprocess -> MNN inference.
 *
 * Perf 9: Removed JPEG encode+decode roundtrip — resize directly to uint8,
 * then crop and preprocess. Eliminates stb JPEG codec overhead.
 */

#include "safety_checker.h"
#include "../utils/sd_logger.h"

#include <MNN/Interpreter.hpp>

#include "stb_image_resize2.h"

#include <cstring>
#include <stdexcept>

bool safety_check(const std::vector<uint8_t>& image_data, int width, int height,
                  float& nsfw_score, MNN::Interpreter* interpreter,
                  MNN::Session* session) {
  try {
    // Resize to 256x256
    std::vector<uint8_t> resized_256(256 * 256 * 3);
    if (!stbir_resize_uint8_linear(image_data.data(), width, height, 0,
                                   resized_256.data(), 256, 256, 0,
                                   STBIR_RGB)) {
      throw std::runtime_error("Resize failed");
    }

    // Center crop 224x224 + VGG preprocess (subtract mean)
    std::vector<float> processed_data(224 * 224 * 3);
    int crop_x = (256 - 224) / 2;
    int crop_y = (256 - 224) / 2;
    float vgg_mean[] = {104.0f, 117.0f, 123.0f};
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        for (int c = 0; c < 3; c++) {
          int src_idx = ((y + crop_y) * 256 + (x + crop_x)) * 3 + c;
          int dst_idx = (y * 224 + x) * 3 + c;
          processed_data[dst_idx] =
              static_cast<float>(resized_256[src_idx]) - vgg_mean[c];
        }
      }
    }

    auto input_tensor = interpreter->getSessionInput(session, nullptr);
    auto inputHost = input_tensor->host<float>();
    memcpy(inputHost, processed_data.data(), 224 * 224 * 3 * sizeof(float));
    interpreter->runSession(session);
    auto output_tensor = interpreter->getSessionOutput(session, nullptr);
    auto outputHost = output_tensor->host<float>();
    nsfw_score = outputHost[1];
    SD_LOG_DEBUG("[SAFETY] NSFW Score: %.4f", nsfw_score);
    return true;
  } catch (const std::exception& e) {
    SD_LOG_ERROR("[SAFETY] Safety check error: %s", e.what());
    return false;
  }
}
