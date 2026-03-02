#pragma once

/**
 * DepthEstimator — Monocular depth estimation.
 *
 * Phase 5.5: Single-image depth map for bokeh, 3D parallax, fog effects.
 *
 * Model options:
 *   MiDaS v2.1 small (~25MB, 15-30ms GPU)
 *   Depth Anything V2 small (~50MB, 30-60ms)
 *   Depth Anything V2 base (~200MB, 60-120ms)
 *
 * Output: float array normalized [0, 1] (near=0, far=1)
 */

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
    class Interpreter;
    class Session;
}

class DepthEstimator {
public:
    DepthEstimator() = default;
    ~DepthEstimator();

    DepthEstimator(const DepthEstimator&) = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;

    /// Load depth model.
    bool loadModel(const std::string& modelPath, bool useOpenCL = false);

    /// Estimate depth. Returns float array (width x height), normalized [0, 1].
    std::vector<float> estimateDepth(const uint8_t* rgbData, int width, int height);

    /// Estimate depth and return colorized RGB heatmap for visualization.
    std::vector<uint8_t> estimateDepthColorized(const uint8_t* rgbData,
                                                 int width, int height);

    /// Release all resources.
    void release();

    bool isLoaded() const { return loaded_; }

private:
    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    bool loaded_ = false;
    int modelInputSize_ = 256;  // 256 for MiDaS, 384 for DepthAnything
};
