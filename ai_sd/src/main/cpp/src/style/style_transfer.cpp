/**
 * StyleTransfer — Neural style transfer (Phase 5.6).
 *
 * Supports fast arbitrary style transfer models that take:
 *   Content image (1x3xHxW) + Style image (1x3x256x256) → Stylized (1x3xHxW)
 *
 * Some models use a two-step approach:
 *   1. Style encoder: style image → style embedding vector
 *   2. Transfer network: content + style embedding → stylized output
 *
 * This implementation uses a single combined model (AdaIN-style) for simplicity.
 */

#include "style_transfer.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

static constexpr int STYLE_SIZE = 256;  // Style image input size

StyleTransfer::~StyleTransfer() {
    release();
}

bool StyleTransfer::loadModel(const std::string& modelPath, bool useOpenCL) {
    release();

    interpreter_ = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!interpreter_) {
        SD_LOG_ERROR("[STYLE] Failed to load model: %s", modelPath.c_str());
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
        SD_LOG_ERROR("[STYLE] Failed to create session");
        release();
        return false;
    }

    if (useOpenCL) {
        interpreter_->updateCacheFile(session_);
    }
    interpreter_->releaseModel();

    loaded_ = true;
    SD_LOG_INFO("[STYLE] Model loaded: %s", modelPath.c_str());
    return true;
}

std::vector<uint8_t> StyleTransfer::stylize(const uint8_t* contentRgb, int contentW, int contentH,
                                             const uint8_t* styleRgb, int styleW, int styleH,
                                             float strength) {
    if (!loaded_) return {};

    // Round content size to multiple of 8
    int procW = (contentW + 7) & ~7;
    int procH = (contentH + 7) & ~7;
    // Cap at reasonable size for mobile
    procW = std::min(procW, 512);
    procH = std::min(procH, 512);

    // Resize session for content dimensions
    auto contentInput = interpreter_->getSessionInput(session_, "content");
    auto styleInput = interpreter_->getSessionInput(session_, "style");

    // Fallback: if named inputs don't exist, use first and second inputs
    if (!contentInput) {
        contentInput = interpreter_->getSessionInput(session_, nullptr);
    }

    if (contentInput) {
        interpreter_->resizeTensor(contentInput, {1, 3, procH, procW});
    }
    if (styleInput) {
        interpreter_->resizeTensor(styleInput, {1, 3, STYLE_SIZE, STYLE_SIZE});
    }
    interpreter_->resizeSession(session_);

    // Prepare content input: resize and normalize [0, 1]
    if (contentInput) {
        std::unique_ptr<MNN::Tensor> contentHost(
            new MNN::Tensor(contentInput, MNN::Tensor::CAFFE));
        float* dst = contentHost->host<float>();

        float scaleX = static_cast<float>(contentW) / procW;
        float scaleY = static_cast<float>(contentH) / procH;

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < procH; ++y) {
                float srcY = y * scaleY;
                int y0 = static_cast<int>(srcY);
                int y1 = std::min(y0 + 1, contentH - 1);
                float fy = srcY - y0;

                for (int x = 0; x < procW; ++x) {
                    float srcX = x * scaleX;
                    int x0 = static_cast<int>(srcX);
                    int x1 = std::min(x0 + 1, contentW - 1);
                    float fx = srcX - x0;

                    float v = contentRgb[(y0 * contentW + x0) * 3 + c] * (1-fx) * (1-fy)
                            + contentRgb[(y0 * contentW + x1) * 3 + c] * fx * (1-fy)
                            + contentRgb[(y1 * contentW + x0) * 3 + c] * (1-fx) * fy
                            + contentRgb[(y1 * contentW + x1) * 3 + c] * fx * fy;
                    dst[c * procH * procW + y * procW + x] = v / 255.0f;
                }
            }
        }
        contentInput->copyFromHostTensor(contentHost.get());
    }

    // Prepare style input: resize to STYLE_SIZE x STYLE_SIZE and normalize [0, 1]
    if (styleInput) {
        std::unique_ptr<MNN::Tensor> styleHost(
            new MNN::Tensor(styleInput, MNN::Tensor::CAFFE));
        float* dst = styleHost->host<float>();

        float scaleX = static_cast<float>(styleW) / STYLE_SIZE;
        float scaleY = static_cast<float>(styleH) / STYLE_SIZE;

        for (int c = 0; c < 3; ++c) {
            for (int y = 0; y < STYLE_SIZE; ++y) {
                float srcY = y * scaleY;
                int y0 = static_cast<int>(srcY);
                int y1 = std::min(y0 + 1, styleH - 1);
                float fy = srcY - y0;

                for (int x = 0; x < STYLE_SIZE; ++x) {
                    float srcX = x * scaleX;
                    int x0 = static_cast<int>(srcX);
                    int x1 = std::min(x0 + 1, styleW - 1);
                    float fx = srcX - x0;

                    float v = styleRgb[(y0 * styleW + x0) * 3 + c] * (1-fx) * (1-fy)
                            + styleRgb[(y0 * styleW + x1) * 3 + c] * fx * (1-fy)
                            + styleRgb[(y1 * styleW + x0) * 3 + c] * (1-fx) * fy
                            + styleRgb[(y1 * styleW + x1) * 3 + c] * fx * fy;
                    dst[c * STYLE_SIZE * STYLE_SIZE + y * STYLE_SIZE + x] = v / 255.0f;
                }
            }
        }
        styleInput->copyFromHostTensor(styleHost.get());
    }

    SD_LOG_INFO("[STYLE] Running style transfer: content=%dx%d, style=%dx%d, strength=%.2f",
                procW, procH, STYLE_SIZE, STYLE_SIZE, strength);
    interpreter_->runSession(session_);

    // Get output
    auto output = interpreter_->getSessionOutput(session_, nullptr);
    std::unique_ptr<MNN::Tensor> outputHost(
        new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputHost.get());

    float* outData = outputHost->host<float>();

    // Resize back to original content dimensions and blend with original
    std::vector<uint8_t> result(contentW * contentH * 3);
    float invScaleX = static_cast<float>(procW) / contentW;
    float invScaleY = static_cast<float>(procH) / contentH;

    for (int y = 0; y < contentH; ++y) {
        float srcY = y * invScaleY;
        int y0 = static_cast<int>(srcY);
        int y1 = std::min(y0 + 1, procH - 1);
        float fy = srcY - y0;

        for (int x = 0; x < contentW; ++x) {
            float srcX = x * invScaleX;
            int x0 = static_cast<int>(srcX);
            int x1 = std::min(x0 + 1, procW - 1);
            float fx = srcX - x0;

            for (int c = 0; c < 3; ++c) {
                float stylized = outData[c * procH * procW + y0 * procW + x0] * (1-fx) * (1-fy)
                               + outData[c * procH * procW + y0 * procW + x1] * fx * (1-fy)
                               + outData[c * procH * procW + y1 * procW + x0] * (1-fx) * fy
                               + outData[c * procH * procW + y1 * procW + x1] * fx * fy;

                // Blend with strength parameter
                float original = contentRgb[(y * contentW + x) * 3 + c] / 255.0f;
                float blended = original * (1.0f - strength) + stylized * strength;

                int pixel = static_cast<int>(blended * 255.0f + 0.5f);
                result[(y * contentW + x) * 3 + c] =
                    static_cast<uint8_t>(std::max(0, std::min(255, pixel)));
            }
        }
    }

    SD_LOG_INFO("[STYLE] Transfer complete: %dx%d", contentW, contentH);
    return result;
}

void StyleTransfer::release() {
    if (session_ && interpreter_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    delete interpreter_;
    interpreter_ = nullptr;
    loaded_ = false;
}
