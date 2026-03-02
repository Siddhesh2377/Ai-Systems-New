#pragma once

/**
 * Upscaler — 4x super-resolution with tiled inference.
 *
 * Extracted from diffusion_pipeline.cpp (Phase 1.7).
 * Supports QNN (executeUpscalerGraphs) and MNN backends.
 */

#include <memory>
#include <string>
#include <vector>

#include <xtensor/xarray.hpp>

// Forward declarations
class QnnModel;

/// Upscale image 4x using QNN model with tiled inference and overlap blending.
xt::xarray<uint8_t> upscaleImageWithModel(
    const std::vector<uint8_t>& input_image, int width, int height,
    std::unique_ptr<QnnModel>& upscaler);

/// Upscale image 4x using MNN model with tiled inference and overlap blending.
xt::xarray<uint8_t> upscaleImageWithMNN(
    const std::vector<uint8_t>& input_image, int width, int height,
    const std::string& model_path, bool use_opencl);
