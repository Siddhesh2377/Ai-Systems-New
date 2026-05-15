/**
 * Segmenter — MobileSAM-based image segmentation (Phase 5.3).
 *
 * Two-session architecture:
 *   1. Image encoder (TinyViT): runs once per image, produces 256-dim embedding
 *   2. Mask decoder: runs per query (point/box), produces binary mask + IoU score
 *
 * MobileSAM input sizes:
 *   Encoder: 1x3x1024x1024 (image resized to 1024)
 *   Decoder: embedding(1x256x64x64) + point_coords(1xNx2) + point_labels(1xN)
 *            + mask_input(1x1x256x256) + has_mask_input(1)
 */

#define TN_MODULE TN_MODULE_AI_SD
#define TN_TAG    "ai_sd"
#include <tn_security/tn_security_macros.h>

#include "segmenter.h"
#include "../utils/sd_logger.h"

#include <MNN/MNNDefine.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

static constexpr int SAM_INPUT_SIZE = 1024;
static constexpr int SAM_EMBED_SIZE = 64;   // embedding spatial dim
static constexpr int SAM_MASK_SIZE = 256;   // mask input/output spatial dim

Segmenter::~Segmenter() {
    release();
}

bool Segmenter::loadModel(const std::string& encoderPath, const std::string& decoderPath,
                          bool useOpenCL) {
    release();

    // --- Encoder ---
    encoder_ = MNN::Interpreter::createFromFile(encoderPath.c_str());
    if (!encoder_) {
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_SEGMENT,
               "[SEG] Failed to load encoder: %s", encoderPath.c_str());
        SD_LOG_ERROR("[SEG] Failed to load encoder: %s", encoderPath.c_str());
        return false;
    }

    MNN::ScheduleConfig encCfg;
    MNN::BackendConfig encBack;
    if (useOpenCL) {
        encCfg.type = MNN_FORWARD_OPENCL;
        encCfg.mode = MNN_GPU_MEMORY_BUFFER | MNN_GPU_TUNING_FAST;
        encBack.precision = MNN::BackendConfig::Precision_Low;
    } else {
        encCfg.type = MNN_FORWARD_CPU;
        encCfg.numThread = 4;
        encBack.memory = MNN::BackendConfig::Memory_Low;
    }
    encBack.power = MNN::BackendConfig::Power_High;
    encCfg.backendConfig = &encBack;

    encoderSession_ = encoder_->createSession(encCfg);
    if (!encoderSession_) {
        SD_LOG_ERROR("[SEG] Failed to create encoder session");
        release();
        return false;
    }

    auto encInput = encoder_->getSessionInput(encoderSession_, nullptr);
    encoder_->resizeTensor(encInput, {1, 3, SAM_INPUT_SIZE, SAM_INPUT_SIZE});
    encoder_->resizeSession(encoderSession_);
    encoder_->releaseModel();

    // --- Decoder ---
    decoder_ = MNN::Interpreter::createFromFile(decoderPath.c_str());
    if (!decoder_) {
        TN_ERR(TN_CODE_MNN_INIT_FAIL, TN_STAGE_SD_SEGMENT,
               "[SEG] Failed to load decoder: %s", decoderPath.c_str());
        SD_LOG_ERROR("[SEG] Failed to load decoder: %s", decoderPath.c_str());
        release();
        return false;
    }

    MNN::ScheduleConfig decCfg;
    MNN::BackendConfig decBack;
    decCfg.type = MNN_FORWARD_CPU;  // decoder is lightweight, CPU is fine
    decCfg.numThread = 4;
    decBack.memory = MNN::BackendConfig::Memory_Low;
    decBack.power = MNN::BackendConfig::Power_High;
    decCfg.backendConfig = &decBack;

    decoderSession_ = decoder_->createSession(decCfg);
    if (!decoderSession_) {
        SD_LOG_ERROR("[SEG] Failed to create decoder session");
        release();
        return false;
    }
    decoder_->releaseModel();

    loaded_ = true;
    SD_LOG_INFO("[SEG] MobileSAM loaded: encoder=%s, decoder=%s",
                encoderPath.c_str(), decoderPath.c_str());
    return true;
}

bool Segmenter::encodeImage(const uint8_t* rgbData, int width, int height) {
    if (!loaded_) return false;

    imageWidth_ = width;
    imageHeight_ = height;

    auto input = encoder_->getSessionInput(encoderSession_, nullptr);

    // Resize to SAM_INPUT_SIZE x SAM_INPUT_SIZE, normalize with ImageNet mean/std
    // SAM uses pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]
    static const float mean[3] = {123.675f, 116.28f, 103.53f};
    static const float std_val[3] = {58.395f, 57.12f, 57.375f};

    std::unique_ptr<MNN::Tensor> inputHost(
        new MNN::Tensor(input, MNN::Tensor::CAFFE));
    float* dst = inputHost->host<float>();

    // Compute scale to fit longest edge to SAM_INPUT_SIZE
    float scale = static_cast<float>(SAM_INPUT_SIZE) / std::max(width, height);
    int newW = static_cast<int>(width * scale + 0.5f);
    int newH = static_cast<int>(height * scale + 0.5f);

    // Zero-fill (pad)
    std::memset(dst, 0, 3 * SAM_INPUT_SIZE * SAM_INPUT_SIZE * sizeof(float));

    // Bilinear resize + normalize into NCHW buffer
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < newH; ++y) {
            float srcY = y / scale;
            int y0 = static_cast<int>(srcY);
            int y1 = std::min(y0 + 1, height - 1);
            float fy = srcY - y0;

            for (int x = 0; x < newW; ++x) {
                float srcX = x / scale;
                int x0 = static_cast<int>(srcX);
                int x1 = std::min(x0 + 1, width - 1);
                float fx = srcX - x0;

                float v00 = rgbData[(y0 * width + x0) * 3 + c];
                float v01 = rgbData[(y0 * width + x1) * 3 + c];
                float v10 = rgbData[(y1 * width + x0) * 3 + c];
                float v11 = rgbData[(y1 * width + x1) * 3 + c];

                float val = v00 * (1-fx) * (1-fy) + v01 * fx * (1-fy) +
                            v10 * (1-fx) * fy + v11 * fx * fy;
                dst[c * SAM_INPUT_SIZE * SAM_INPUT_SIZE + y * SAM_INPUT_SIZE + x] =
                    (val - mean[c]) / std_val[c];
            }
        }
    }

    input->copyFromHostTensor(inputHost.get());
    encoder_->runSession(encoderSession_);

    encoded_ = true;
    SD_LOG_INFO("[SEG] Image encoded: %dx%d -> %dx%d (scale=%.3f)", width, height, newW, newH, scale);
    return true;
}

std::vector<uint8_t> Segmenter::segmentAtPoint(float x, float y, float& score) {
    if (!encoded_) return {};

    // Transform point from image coords to SAM input coords
    float scale = static_cast<float>(SAM_INPUT_SIZE) / std::max(imageWidth_, imageHeight_);
    float samX = x * scale;
    float samY = y * scale;

    // Get encoder output (image embedding)
    auto embedding = encoder_->getSessionOutput(encoderSession_, nullptr);

    // Set decoder inputs
    auto decEmbed = decoder_->getSessionInput(decoderSession_, "image_embeddings");
    if (decEmbed) {
        embedding->copyToHostTensor(decEmbed);
    }

    auto pointCoords = decoder_->getSessionInput(decoderSession_, "point_coords");
    auto pointLabels = decoder_->getSessionInput(decoderSession_, "point_labels");

    if (pointCoords && pointLabels) {
        // 1 foreground point + 1 padding point
        decoder_->resizeTensor(pointCoords, {1, 2, 2});
        decoder_->resizeTensor(pointLabels, {1, 2});
        decoder_->resizeSession(decoderSession_);

        std::unique_ptr<MNN::Tensor> coordsHost(
            new MNN::Tensor(pointCoords, MNN::Tensor::CAFFE));
        std::unique_ptr<MNN::Tensor> labelsHost(
            new MNN::Tensor(pointLabels, MNN::Tensor::CAFFE));

        float* coords = coordsHost->host<float>();
        float* labels = labelsHost->host<float>();

        // Point 0: foreground click
        coords[0] = samX; coords[1] = samY;
        labels[0] = 1.0f;  // foreground

        // Point 1: padding (SAM expects pairs)
        coords[2] = 0.0f; coords[3] = 0.0f;
        labels[1] = -1.0f;  // padding

        pointCoords->copyFromHostTensor(coordsHost.get());
        pointLabels->copyFromHostTensor(labelsHost.get());
    }

    // Empty mask input (no previous mask)
    auto maskInput = decoder_->getSessionInput(decoderSession_, "mask_input");
    if (maskInput) {
        std::unique_ptr<MNN::Tensor> maskHost(
            new MNN::Tensor(maskInput, MNN::Tensor::CAFFE));
        std::memset(maskHost->host<float>(), 0,
                    SAM_MASK_SIZE * SAM_MASK_SIZE * sizeof(float));
        maskInput->copyFromHostTensor(maskHost.get());
    }

    auto hasMask = decoder_->getSessionInput(decoderSession_, "has_mask_input");
    if (hasMask) {
        std::unique_ptr<MNN::Tensor> hasMaskHost(
            new MNN::Tensor(hasMask, MNN::Tensor::CAFFE));
        hasMaskHost->host<float>()[0] = 0.0f;  // no previous mask
        hasMask->copyFromHostTensor(hasMaskHost.get());
    }

    decoder_->runSession(decoderSession_);

    // Get outputs: masks (1x4x256x256) and iou_predictions (1x4)
    auto masksOut = decoder_->getSessionOutput(decoderSession_, "masks");
    auto iouOut = decoder_->getSessionOutput(decoderSession_, "iou_predictions");

    std::unique_ptr<MNN::Tensor> masksHost(
        new MNN::Tensor(masksOut, MNN::Tensor::CAFFE));
    masksOut->copyToHostTensor(masksHost.get());

    std::unique_ptr<MNN::Tensor> iouHost(
        new MNN::Tensor(iouOut, MNN::Tensor::CAFFE));
    iouOut->copyToHostTensor(iouHost.get());

    // Pick best mask (highest IoU)
    float* iouScores = iouHost->host<float>();
    int bestIdx = 0;
    float bestScore = iouScores[0];
    for (int i = 1; i < 4; ++i) {
        if (iouScores[i] > bestScore) {
            bestScore = iouScores[i];
            bestIdx = i;
        }
    }
    score = bestScore;

    // Extract best mask, threshold at 0, resize to original image size
    float* bestMask = masksHost->host<float>() + bestIdx * SAM_MASK_SIZE * SAM_MASK_SIZE;

    std::vector<uint8_t> result(imageWidth_ * imageHeight_);
    float scaleX = static_cast<float>(SAM_MASK_SIZE) / imageWidth_;
    float scaleY = static_cast<float>(SAM_MASK_SIZE) / imageHeight_;

    for (int y2 = 0; y2 < imageHeight_; ++y2) {
        for (int x2 = 0; x2 < imageWidth_; ++x2) {
            int mx = std::min(static_cast<int>(x2 * scaleX), SAM_MASK_SIZE - 1);
            int my = std::min(static_cast<int>(y2 * scaleY), SAM_MASK_SIZE - 1);
            result[y2 * imageWidth_ + x2] = (bestMask[my * SAM_MASK_SIZE + mx] > 0.0f) ? 255 : 0;
        }
    }

    SD_LOG_INFO("[SEG] Point segment: (%.1f,%.1f) -> score=%.3f", x, y, score);
    return result;
}

std::vector<uint8_t> Segmenter::segmentWithBox(float x1, float y1, float x2, float y2, float& score) {
    if (!encoded_) return {};

    // Transform box from image coords to SAM input coords
    float scale = static_cast<float>(SAM_INPUT_SIZE) / std::max(imageWidth_, imageHeight_);

    // Get encoder output (image embedding)
    auto embedding = encoder_->getSessionOutput(encoderSession_, nullptr);

    auto decEmbed = decoder_->getSessionInput(decoderSession_, "image_embeddings");
    if (decEmbed) {
        embedding->copyToHostTensor(decEmbed);
    }

    auto pointCoords = decoder_->getSessionInput(decoderSession_, "point_coords");
    auto pointLabels = decoder_->getSessionInput(decoderSession_, "point_labels");

    if (pointCoords && pointLabels) {
        // Box: 2 corner points + 1 padding
        decoder_->resizeTensor(pointCoords, {1, 3, 2});
        decoder_->resizeTensor(pointLabels, {1, 3});
        decoder_->resizeSession(decoderSession_);

        std::unique_ptr<MNN::Tensor> coordsHost(
            new MNN::Tensor(pointCoords, MNN::Tensor::CAFFE));
        std::unique_ptr<MNN::Tensor> labelsHost(
            new MNN::Tensor(pointLabels, MNN::Tensor::CAFFE));

        float* coords = coordsHost->host<float>();
        float* labels = labelsHost->host<float>();

        // Box corners
        coords[0] = x1 * scale; coords[1] = y1 * scale;
        labels[0] = 2.0f;  // top-left box corner
        coords[2] = x2 * scale; coords[3] = y2 * scale;
        labels[1] = 3.0f;  // bottom-right box corner
        // Padding
        coords[4] = 0.0f; coords[5] = 0.0f;
        labels[2] = -1.0f;

        pointCoords->copyFromHostTensor(coordsHost.get());
        pointLabels->copyFromHostTensor(labelsHost.get());
    }

    // Empty mask input
    auto maskInput = decoder_->getSessionInput(decoderSession_, "mask_input");
    if (maskInput) {
        std::unique_ptr<MNN::Tensor> maskHost(
            new MNN::Tensor(maskInput, MNN::Tensor::CAFFE));
        std::memset(maskHost->host<float>(), 0,
                    SAM_MASK_SIZE * SAM_MASK_SIZE * sizeof(float));
        maskInput->copyFromHostTensor(maskHost.get());
    }

    auto hasMask = decoder_->getSessionInput(decoderSession_, "has_mask_input");
    if (hasMask) {
        std::unique_ptr<MNN::Tensor> hasMaskHost(
            new MNN::Tensor(hasMask, MNN::Tensor::CAFFE));
        hasMaskHost->host<float>()[0] = 0.0f;
        hasMask->copyFromHostTensor(hasMaskHost.get());
    }

    decoder_->runSession(decoderSession_);

    // Get outputs
    auto masksOut = decoder_->getSessionOutput(decoderSession_, "masks");
    auto iouOut = decoder_->getSessionOutput(decoderSession_, "iou_predictions");

    std::unique_ptr<MNN::Tensor> masksHost(
        new MNN::Tensor(masksOut, MNN::Tensor::CAFFE));
    masksOut->copyToHostTensor(masksHost.get());

    std::unique_ptr<MNN::Tensor> iouHost(
        new MNN::Tensor(iouOut, MNN::Tensor::CAFFE));
    iouOut->copyToHostTensor(iouHost.get());

    float* iouScores = iouHost->host<float>();
    int bestIdx = 0;
    float bestScore = iouScores[0];
    for (int i = 1; i < 4; ++i) {
        if (iouScores[i] > bestScore) {
            bestScore = iouScores[i];
            bestIdx = i;
        }
    }
    score = bestScore;

    float* bestMask = masksHost->host<float>() + bestIdx * SAM_MASK_SIZE * SAM_MASK_SIZE;
    std::vector<uint8_t> result(imageWidth_ * imageHeight_);
    float scaleX = static_cast<float>(SAM_MASK_SIZE) / imageWidth_;
    float scaleY = static_cast<float>(SAM_MASK_SIZE) / imageHeight_;

    for (int y3 = 0; y3 < imageHeight_; ++y3) {
        for (int x3 = 0; x3 < imageWidth_; ++x3) {
            int mx = std::min(static_cast<int>(x3 * scaleX), SAM_MASK_SIZE - 1);
            int my = std::min(static_cast<int>(y3 * scaleY), SAM_MASK_SIZE - 1);
            result[y3 * imageWidth_ + x3] = (bestMask[my * SAM_MASK_SIZE + mx] > 0.0f) ? 255 : 0;
        }
    }

    SD_LOG_INFO("[SEG] Box segment: (%.1f,%.1f)-(%.1f,%.1f) -> score=%.3f", x1, y1, x2, y2, score);
    return result;
}

void Segmenter::release() {
    if (decoderSession_ && decoder_) {
        decoder_->releaseSession(decoderSession_);
        decoderSession_ = nullptr;
    }
    if (encoderSession_ && encoder_) {
        encoder_->releaseSession(encoderSession_);
        encoderSession_ = nullptr;
    }
    delete decoder_;  decoder_ = nullptr;
    delete encoder_;  encoder_ = nullptr;
    loaded_ = false;
    encoded_ = false;
    imageWidth_ = 0;
    imageHeight_ = 0;
}
