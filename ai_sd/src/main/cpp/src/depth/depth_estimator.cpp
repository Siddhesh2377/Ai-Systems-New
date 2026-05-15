/**
 * DepthEstimator — Monocular depth estimation (Phase 5.5).
 *
 * Supports MiDaS v2.1 small and Depth Anything V2 models.
 * Input: 1x3xHxW RGB normalized
 * Output: 1x1xHxW depth map (inverse depth, higher = closer)
 */

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "depth_estimator.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

DepthEstimator::~DepthEstimator() {
    release();
}

bool DepthEstimator::loadModel(const std::string& modelPath, bool useOpenCL) {
    release();

    interpreter_ = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!interpreter_) {
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_DEPTH,
               "[DEPTH] Failed to load model: %s", modelPath.c_str());
        SD_LOG_ERROR("[DEPTH] Failed to load model: %s", modelPath.c_str());
        return false;
    }

    MNN::ScheduleConfig cfg;
    MNN::BackendConfig back;
    if (useOpenCL) {
        auto cacheFile = modelPath + ".mnnc";
        interpreter_->setCacheFile(cacheFile.c_str());
        cfg.type = MNN_FORWARD_OPENCL;
        cfg.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
        back.precision = MNN::BackendConfig::Precision_Low;
    } else {
        cfg.type = MNN_FORWARD_CPU;
        cfg.numThread = 4;
        back.memory = MNN::BackendConfig::Memory_Low;
    }
    back.power = MNN::BackendConfig::Power_High;
    cfg.backendConfig = &back;

    session_ = interpreter_->createSession(cfg);
    if (!session_) {
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_DEPTH,
               "[DEPTH] Failed to create session");
        SD_LOG_ERROR("[DEPTH] Failed to create session");
        release();
        return false;
    }

    // Detect model input size from first input tensor
    auto input = interpreter_->getSessionInput(session_, nullptr);
    if (input) {
        auto shape = input->shape();
        if (shape.size() >= 4) {
            modelInputSize_ = shape[2];  // HxW (square models)
            SD_LOG_INFO("[DEPTH] Detected model input size: %d", modelInputSize_);
        }
    }

    if (useOpenCL) {
        interpreter_->updateCacheFile(session_);
    }
    interpreter_->releaseModel();

    loaded_ = true;
    SD_LOG_INFO("[DEPTH] Model loaded: %s (input=%d)", modelPath.c_str(), modelInputSize_);
    return true;
}

std::vector<float> DepthEstimator::estimateDepth(const uint8_t* rgbData, int width, int height) {
    if (!loaded_) return {};

    auto input = interpreter_->getSessionInput(session_, nullptr);
    int inSize = modelInputSize_;

    // Resize session for model input
    interpreter_->resizeTensor(input, {1, 3, inSize, inSize});
    interpreter_->resizeSession(session_);

    // Normalize: MiDaS uses ImageNet normalization
    // mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std_val[3] = {0.229f, 0.224f, 0.225f};

    std::unique_ptr<MNN::Tensor> inputHost(
        new MNN::Tensor(input, MNN::Tensor::CAFFE));
    float* dst = inputHost->host<float>();

    float scaleX = static_cast<float>(width) / inSize;
    float scaleY = static_cast<float>(height) / inSize;

    // Bilinear resize + normalize into NCHW buffer
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < inSize; ++y) {
            float srcY = y * scaleY;
            int y0 = static_cast<int>(srcY);
            int y1 = std::min(y0 + 1, height - 1);
            float fy = srcY - y0;

            for (int x = 0; x < inSize; ++x) {
                float srcX = x * scaleX;
                int x0 = static_cast<int>(srcX);
                int x1 = std::min(x0 + 1, width - 1);
                float fx = srcX - x0;

                float v = rgbData[(y0 * width + x0) * 3 + c] * (1-fx) * (1-fy)
                        + rgbData[(y0 * width + x1) * 3 + c] * fx * (1-fy)
                        + rgbData[(y1 * width + x0) * 3 + c] * (1-fx) * fy
                        + rgbData[(y1 * width + x1) * 3 + c] * fx * fy;
                dst[c * inSize * inSize + y * inSize + x] =
                    (v / 255.0f - mean[c]) / std_val[c];
            }
        }
    }

    input->copyFromHostTensor(inputHost.get());

    SD_LOG_INFO("[DEPTH] Running depth estimation at %dx%d", inSize, inSize);
    interpreter_->runSession(session_);

    // Get output: typically 1x1xHxW or 1xHxW
    auto output = interpreter_->getSessionOutput(session_, nullptr);
    std::unique_ptr<MNN::Tensor> outputHost(
        new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputHost.get());

    float* depthRaw = outputHost->host<float>();
    int outSize = inSize;  // output matches input for most depth models

    // Find min/max for normalization
    float minVal = depthRaw[0], maxVal = depthRaw[0];
    for (int i = 1; i < outSize * outSize; ++i) {
        minVal = std::min(minVal, depthRaw[i]);
        maxVal = std::max(maxVal, depthRaw[i]);
    }
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;

    // Resize to original dimensions and normalize [0, 1]
    std::vector<float> result(width * height);
    for (int y = 0; y < height; ++y) {
        float srcY = y * static_cast<float>(outSize) / height;
        int y0 = static_cast<int>(srcY);
        int y1 = std::min(y0 + 1, outSize - 1);
        float fy = srcY - y0;

        for (int x = 0; x < width; ++x) {
            float srcX = x * static_cast<float>(outSize) / width;
            int x0 = static_cast<int>(srcX);
            int x1 = std::min(x0 + 1, outSize - 1);
            float fx = srcX - x0;

            float v = depthRaw[y0 * outSize + x0] * (1-fx) * (1-fy)
                    + depthRaw[y0 * outSize + x1] * fx * (1-fy)
                    + depthRaw[y1 * outSize + x0] * (1-fx) * fy
                    + depthRaw[y1 * outSize + x1] * fx * fy;
            result[y * width + x] = (v - minVal) / range;
        }
    }

    SD_LOG_INFO("[DEPTH] Estimation complete: %dx%d, range=[%.3f, %.3f]", width, height, minVal, maxVal);
    return result;
}

std::vector<uint8_t> DepthEstimator::estimateDepthColorized(const uint8_t* rgbData,
                                                              int width, int height) {
    auto depth = estimateDepth(rgbData, width, height);
    if (depth.empty()) return {};

    // Inferno-like colormap: near (0) = warm/yellow, far (1) = cool/purple
    auto colormap = [](float t, uint8_t& r, uint8_t& g, uint8_t& b) {
        // Simplified inferno: dark purple -> red -> yellow -> white
        if (t < 0.25f) {
            float s = t / 0.25f;
            r = static_cast<uint8_t>(s * 120);
            g = 0;
            b = static_cast<uint8_t>(40 + s * 80);
        } else if (t < 0.5f) {
            float s = (t - 0.25f) / 0.25f;
            r = static_cast<uint8_t>(120 + s * 135);
            g = static_cast<uint8_t>(s * 50);
            b = static_cast<uint8_t>(120 - s * 80);
        } else if (t < 0.75f) {
            float s = (t - 0.5f) / 0.25f;
            r = 255;
            g = static_cast<uint8_t>(50 + s * 150);
            b = static_cast<uint8_t>(40 - s * 40);
        } else {
            float s = (t - 0.75f) / 0.25f;
            r = 255;
            g = static_cast<uint8_t>(200 + s * 55);
            b = static_cast<uint8_t>(s * 200);
        }
    };

    std::vector<uint8_t> result(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        // Invert: near objects should be bright (high t)
        float t = 1.0f - depth[i];
        colormap(t, result[i * 3], result[i * 3 + 1], result[i * 3 + 2]);
    }

    return result;
}

void DepthEstimator::release() {
    if (session_ && interpreter_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    delete interpreter_;
    interpreter_ = nullptr;
    loaded_ = false;
}
