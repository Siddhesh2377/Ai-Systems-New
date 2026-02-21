#pragma once

/**
 * JNI callback utilities for Stable Diffusion pipeline
 *
 * Thread-local caching of JNI method IDs for minimal overhead.
 * Modeled after ai_gguf's proven JNI callback pattern.
 */

#include <jni.h>
#include <string>

namespace sd_jni {

/**
 * Send progress update to Java callback
 * @param step Current step number
 * @param totalSteps Total number of steps
 */
void on_progress(JNIEnv* env, jobject cb, int step, int totalSteps);

/**
 * Send progress update with intermediate image to Java callback
 * @param step Current step number
 * @param totalSteps Total number of steps
 * @param rgbData Raw RGB byte data of the intermediate image
 * @param dataLen Length of rgbData
 * @param width Image width
 * @param height Image height
 */
void on_image_progress(JNIEnv* env, jobject cb, int step, int totalSteps,
                       const uint8_t* rgbData, int dataLen, int width, int height);

/**
 * Send completion with final image to Java callback
 * @param rgbData Raw RGB byte data of the final image
 * @param dataLen Length of rgbData
 * @param width Image width
 * @param height Image height
 * @param seed Seed used for generation
 * @param generationTimeMs Total generation time in ms
 */
void on_complete(JNIEnv* env, jobject cb, const uint8_t* rgbData, int dataLen,
                 int width, int height, long seed, int generationTimeMs);

/**
 * Send error message to Java callback
 */
void on_error(JNIEnv* env, jobject cb, const char* msg);

/**
 * Reset cached JNI references
 */
void reset_cache();

} // namespace sd_jni
