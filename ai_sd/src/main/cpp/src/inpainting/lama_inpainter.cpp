/**
 * LamaInpainter — LaMa-based fast inpainting (Phase 5.4).
 *
 * LaMa (Large Mask Inpainting) uses fast Fourier convolutions for
 * high-quality inpainting even with large masks.
 *
 * Input: 1x4xHxW (3 RGB channels + 1 mask channel), normalized [0,1]
 * Output: 1x3xHxW RGB, normalized [0,1]
 * Typical model input size: 512x512 (will resize if larger)
 */

#include "lama_inpainter.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

static constexpr int LAMA_MAX_SIZE = 512;  // LaMa trained at 512x512

LamaInpainter::~LamaInpainter() {
    release();
}

bool LamaInpainter::loadModel(const std::string& modelPath, bool useOpenCL) {
    release();

    interpreter_ = MNN::Interpreter::createFromFile(modelPath.c_str());
    if (!interpreter_) {
        SD_LOG_ERROR("[LAMA] Failed to load model: %s", modelPath.c_str());
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
        SD_LOG_ERROR("[LAMA] Failed to create session");
        release();
        return false;
    }

    if (useOpenCL) {
        interpreter_->updateCacheFile(session_);
    }
    interpreter_->releaseModel();

    loaded_ = true;
    SD_LOG_INFO("[LAMA] Model loaded: %s", modelPath.c_str());
    return true;
}

std::vector<uint8_t> LamaInpainter::inpaint(const uint8_t* rgbData, const uint8_t* mask,
                                              int width, int height) {
    if (!loaded_) return {};

    // Determine processing size (pad to model size if needed)
    int procW = std::min(width, LAMA_MAX_SIZE);
    int procH = std::min(height, LAMA_MAX_SIZE);
    // Round to nearest multiple of 8 for model compatibility
    procW = (procW + 7) & ~7;
    procH = (procH + 7) & ~7;

    float scaleX = static_cast<float>(width) / procW;
    float scaleY = static_cast<float>(height) / procH;

    // Resize session to match input dims
    auto input = interpreter_->getSessionInput(session_, nullptr);
    interpreter_->resizeTensor(input, {1, 4, procH, procW});
    interpreter_->resizeSession(session_);

    // Prepare input: 3 RGB channels + 1 mask channel, all normalized [0,1]
    std::unique_ptr<MNN::Tensor> inputHost(
        new MNN::Tensor(input, MNN::Tensor::CAFFE));
    float* dst = inputHost->host<float>();

    // Bilinear resize RGB channels
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < procH; ++y) {
            float srcY = y * scaleY;
            int y0 = static_cast<int>(srcY);
            int y1 = std::min(y0 + 1, height - 1);
            float fy = srcY - y0;

            for (int x = 0; x < procW; ++x) {
                float srcX = x * scaleX;
                int x0 = static_cast<int>(srcX);
                int x1 = std::min(x0 + 1, width - 1);
                float fx = srcX - x0;

                float v = rgbData[(y0 * width + x0) * 3 + c] * (1-fx) * (1-fy)
                        + rgbData[(y0 * width + x1) * 3 + c] * fx * (1-fy)
                        + rgbData[(y1 * width + x0) * 3 + c] * (1-fx) * fy
                        + rgbData[(y1 * width + x1) * 3 + c] * fx * fy;
                dst[c * procH * procW + y * procW + x] = v / 255.0f;
            }
        }
    }

    // Mask channel (nearest-neighbor resize, binary)
    for (int y = 0; y < procH; ++y) {
        int srcY = static_cast<int>(y * scaleY);
        srcY = std::min(srcY, height - 1);
        for (int x = 0; x < procW; ++x) {
            int srcX = static_cast<int>(x * scaleX);
            srcX = std::min(srcX, width - 1);
            dst[3 * procH * procW + y * procW + x] =
                (mask[srcY * width + srcX] > 127) ? 1.0f : 0.0f;
        }
    }

    input->copyFromHostTensor(inputHost.get());

    SD_LOG_INFO("[LAMA] Running inpainting at %dx%d", procW, procH);
    interpreter_->runSession(session_);

    // Get output
    auto output = interpreter_->getSessionOutput(session_, nullptr);
    std::unique_ptr<MNN::Tensor> outputHost(
        new MNN::Tensor(output, MNN::Tensor::CAFFE));
    output->copyToHostTensor(outputHost.get());

    float* outData = outputHost->host<float>();

    // Resize output back to original dimensions and convert to uint8 RGB
    std::vector<uint8_t> result(width * height * 3);
    for (int y = 0; y < height; ++y) {
        float srcY2 = y / scaleY;
        int y0 = static_cast<int>(srcY2);
        int y1 = std::min(y0 + 1, procH - 1);
        float fy = srcY2 - y0;

        for (int x = 0; x < width; ++x) {
            float srcX2 = x / scaleX;
            int x0 = static_cast<int>(srcX2);
            int x1 = std::min(x0 + 1, procW - 1);
            float fx = srcX2 - x0;

            for (int c = 0; c < 3; ++c) {
                float v = outData[c * procH * procW + y0 * procW + x0] * (1-fx) * (1-fy)
                        + outData[c * procH * procW + y0 * procW + x1] * fx * (1-fy)
                        + outData[c * procH * procW + y1 * procW + x0] * (1-fx) * fy
                        + outData[c * procH * procW + y1 * procW + x1] * fx * fy;
                int pixel = static_cast<int>(v * 255.0f + 0.5f);
                result[(y * width + x) * 3 + c] =
                    static_cast<uint8_t>(std::max(0, std::min(255, pixel)));
            }

            // Blend: only replace inpainted region, keep original elsewhere
            if (mask[y * width + x] <= 127) {
                result[(y * width + x) * 3 + 0] = rgbData[(y * width + x) * 3 + 0];
                result[(y * width + x) * 3 + 1] = rgbData[(y * width + x) * 3 + 1];
                result[(y * width + x) * 3 + 2] = rgbData[(y * width + x) * 3 + 2];
            }
        }
    }

    SD_LOG_INFO("[LAMA] Inpainting complete: %dx%d", width, height);
    return result;
}

void LamaInpainter::release() {
    if (session_ && interpreter_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }
    delete interpreter_;
    interpreter_ = nullptr;
    loaded_ = false;
}
