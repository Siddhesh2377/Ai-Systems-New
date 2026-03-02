#pragma once

/**
 * LamaInpainter — LaMa-based fast object removal / inpainting.
 *
 * Phase 5.4: User draws mask over object, AI fills seamlessly.
 * Fast alternative to SD inpainting (~100-300ms vs 5-30s).
 *
 * Model: LaMa-Dilated (~100MB .mnn)
 * Input: RGB image + binary mask, up to 1024x1024
 * Output: inpainted RGB image
 */

#include <cstdint>
#include <string>
#include <vector>

namespace MNN {
    class Interpreter;
    class Session;
}

class LamaInpainter {
public:
    LamaInpainter() = default;
    ~LamaInpainter();

    LamaInpainter(const LamaInpainter&) = delete;
    LamaInpainter& operator=(const LamaInpainter&) = delete;

    /// Load LaMa model.
    bool loadModel(const std::string& modelPath, bool useOpenCL = false);

    /// Inpaint masked region. Returns RGB bytes at original resolution.
    /// mask: binary uint8 (0 = keep, 255 = inpaint), same size as image.
    std::vector<uint8_t> inpaint(const uint8_t* rgbData, const uint8_t* mask,
                                  int width, int height);

    /// Release all resources.
    void release();

    bool isLoaded() const { return loaded_; }

private:
    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    bool loaded_ = false;
};
