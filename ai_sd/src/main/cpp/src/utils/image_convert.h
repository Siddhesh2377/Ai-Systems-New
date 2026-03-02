#pragma once

/**
 * Image format conversion utilities.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.9).
 * Pure math — no dependencies beyond standard library.
 */

#include <cstdint>
#include <vector>

/// Convert NCHW float [-1,1] to interleaved RGB uint8 [0,255].
/// Explicit loop avoids xtensor transpose iterator layout ambiguity.
inline std::vector<uint8_t> nchw_to_rgb_bytes(const float* nchw_data,
                                               int channels, int height,
                                               int width) {
  const int pixel_count = height * width;
  std::vector<uint8_t> rgb(pixel_count * channels);
  for (int c = 0; c < channels; c++) {
    const float* ch_ptr = nchw_data + c * pixel_count;
    for (int i = 0; i < pixel_count; i++) {
      float val = ((ch_ptr[i] + 1.0f) * 0.5f) * 255.0f;
      if (val < 0.0f) val = 0.0f;
      if (val > 255.0f) val = 255.0f;
      rgb[i * channels + c] = static_cast<uint8_t>(val);
    }
  }
  return rgb;
}
