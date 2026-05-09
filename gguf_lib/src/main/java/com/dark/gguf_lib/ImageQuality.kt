package com.dark.gguf_lib

/**
 * VLM input-image quality preset. Controls a JNI-side bilinear downscale
 * applied to the decoded bitmap before it reaches the projector.
 *
 * | Preset | Max long side | Use case |
 * |--------|---------------|----------|
 * | [LOW]    | 384 px | Fast, low-fidelity (UI thumbnails, batch screening) |
 * | [MEDIUM] | 768 px | Mobile default — matches LFM2-VL's native ~512² regime |
 * | [HIGH]   | passthrough | Full resolution, no resize |
 *
 * Lowering quality reduces:
 * - ViT compute time (smaller spatial grid, fewer patches)
 * - Image-prefill batch size in the LLM
 * - VT cache entry size
 *
 * At the cost of detail in the model's perception. For description /
 * captioning prompts, [LOW] is usually fine. For OCR or fine detail, use
 * [HIGH].
 */
enum class ImageQuality(val nativeValue: Int) {
    LOW(0),
    MEDIUM(1),
    HIGH(2);
}
