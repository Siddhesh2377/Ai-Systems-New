#pragma once

/**
 * Segmenter — MobileSAM-based image segmentation.
 *
 * Phase 5.3: Background removal / tap-to-segment.
 * Uses two MNN sessions: image encoder (runs once) + mask decoder (runs per query).
 *
 * Model: MobileSAM with TinyViT encoder (~50MB total)
 * Speed: ~12ms per query on GPU after encoding
 */

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
    class Interpreter;
    class Session;
}

class Segmenter {
public:
    Segmenter() = default;
    ~Segmenter();

    Segmenter(const Segmenter&) = delete;
    Segmenter& operator=(const Segmenter&) = delete;

    /// Load encoder + decoder models.
    bool loadModel(const std::string& encoderPath, const std::string& decoderPath,
                   bool useOpenCL = false);

    /// Encode image and cache the embedding for multi-query.
    bool encodeImage(const uint8_t* rgbData, int width, int height);

    /// Segment at a point. Returns binary mask (width x height uint8).
    std::vector<uint8_t> segmentAtPoint(float x, float y, float& score);

    /// Segment within a bounding box.
    std::vector<uint8_t> segmentWithBox(float x1, float y1, float x2, float y2, float& score);

    /// Release all resources.
    void release();

    bool isLoaded() const { return loaded_; }
    bool isEncoded() const { return encoded_; }

private:
    MNN::Interpreter* encoder_ = nullptr;
    MNN::Interpreter* decoder_ = nullptr;
    MNN::Session* encoderSession_ = nullptr;
    MNN::Session* decoderSession_ = nullptr;
    bool loaded_ = false;
    bool encoded_ = false;
    int imageWidth_ = 0;
    int imageHeight_ = 0;
};
