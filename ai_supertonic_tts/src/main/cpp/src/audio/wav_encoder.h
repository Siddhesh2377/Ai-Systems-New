#pragma once

#include <cstdint>
#include <vector>

namespace audio {

/**
 * Clip float audio samples in-place to [-1.0, 1.0]
 */
void clip_audio(float* data, int samples);

/**
 * Encode float32 audio as 16-bit PCM WAV file bytes.
 * Returns complete WAV file (RIFF header + data).
 */
std::vector<uint8_t> encode_wav_16(const float* data, int samples, int sample_rate, int channels);

/**
 * Encode float32 audio as 32-bit float WAV file bytes (IEEE float format).
 * Returns complete WAV file (RIFF header + data).
 */
std::vector<uint8_t> encode_wav_32f(const float* data, int samples, int sample_rate, int channels);

/**
 * Encode float32 audio as raw 16-bit PCM bytes (no WAV header).
 */
std::vector<uint8_t> encode_pcm_16(const float* data, int samples);

} // namespace audio
