#pragma once

/**
 * StyleTransfer — Neural style transfer.
 *
 * Phase 5.6: Apply artistic styles to images in real-time.
 *
 * Model options:
 *   Fast arbitrary style (~10MB, 30-60 FPS)
 *   Photo-realistic style (~70MB, ~1s per image)
 *
 * Input: content RGB + style image (or style embedding)
 * Output: stylized RGB image
 */

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
    class Interpreter;
    class Session;
}

class StyleTransfer {
public:
    StyleTransfer() = default;
    ~StyleTransfer();

    StyleTransfer(const StyleTransfer&) = delete;
    StyleTransfer& operator=(const StyleTransfer&) = delete;

    /// Load style transfer model.
    bool loadModel(const std::string& modelPath, bool useOpenCL = false);

    /// Apply style from a style image to content image.
    /// Returns stylized RGB bytes at content resolution.
    std::vector<uint8_t> stylize(const uint8_t* contentRgb, int contentW, int contentH,
                                  const uint8_t* styleRgb, int styleW, int styleH,
                                  float strength = 1.0f);

    /// Release all resources.
    void release();

    bool isLoaded() const { return loaded_; }

private:
    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    bool loaded_ = false;
};
