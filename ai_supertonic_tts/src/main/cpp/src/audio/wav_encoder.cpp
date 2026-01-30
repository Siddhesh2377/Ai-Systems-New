#include "wav_encoder.h"

#include <cmath>

namespace audio {

// Write a little-endian uint16
static void write_u16(std::vector<uint8_t>& buf, uint16_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
}

// Write a little-endian uint32
static void write_u32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

// Write 4-byte ASCII tag
static void write_tag(std::vector<uint8_t>& buf, const char* tag) {
    buf.push_back(static_cast<uint8_t>(tag[0]));
    buf.push_back(static_cast<uint8_t>(tag[1]));
    buf.push_back(static_cast<uint8_t>(tag[2]));
    buf.push_back(static_cast<uint8_t>(tag[3]));
}

void clip_audio(float* data, int samples) {
    for (int i = 0; i < samples; ++i) {
        data[i] = std::fmax(-1.0f, std::fmin(1.0f, data[i]));
    }
}

std::vector<uint8_t> encode_wav_16(const float* data, int samples, int sample_rate, int channels) {
    const int bits_per_sample = 16;
    const int byte_rate = sample_rate * channels * (bits_per_sample / 8);
    const int block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(samples * channels * (bits_per_sample / 8));

    // Header = 44 bytes, total = 44 + data_size
    std::vector<uint8_t> buf;
    buf.reserve(44 + data_size);

    // RIFF header
    write_tag(buf, "RIFF");
    write_u32(buf, 36 + data_size);  // file size - 8
    write_tag(buf, "WAVE");

    // fmt chunk
    write_tag(buf, "fmt ");
    write_u32(buf, 16);              // chunk size
    write_u16(buf, 1);               // PCM format
    write_u16(buf, static_cast<uint16_t>(channels));
    write_u32(buf, static_cast<uint32_t>(sample_rate));
    write_u32(buf, static_cast<uint32_t>(byte_rate));
    write_u16(buf, static_cast<uint16_t>(block_align));
    write_u16(buf, static_cast<uint16_t>(bits_per_sample));

    // data chunk
    write_tag(buf, "data");
    write_u32(buf, data_size);

    // Convert float -> int16 and write
    for (int i = 0; i < samples * channels; ++i) {
        float clamped = std::fmax(-1.0f, std::fmin(1.0f, data[i]));
        int16_t val = static_cast<int16_t>(clamped * 32767.0f);
        buf.push_back(static_cast<uint8_t>(val & 0xFF));
        buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    }

    return buf;
}

std::vector<uint8_t> encode_wav_32f(const float* data, int samples, int sample_rate, int channels) {
    const int bits_per_sample = 32;
    const int byte_rate = sample_rate * channels * (bits_per_sample / 8);
    const int block_align = channels * (bits_per_sample / 8);
    const uint32_t data_size = static_cast<uint32_t>(samples * channels * (bits_per_sample / 8));

    // Header = 44 bytes, total = 44 + data_size
    std::vector<uint8_t> buf;
    buf.reserve(44 + data_size);

    // RIFF header
    write_tag(buf, "RIFF");
    write_u32(buf, 36 + data_size);
    write_tag(buf, "WAVE");

    // fmt chunk (IEEE float format = 3)
    write_tag(buf, "fmt ");
    write_u32(buf, 16);
    write_u16(buf, 3);               // IEEE float format
    write_u16(buf, static_cast<uint16_t>(channels));
    write_u32(buf, static_cast<uint32_t>(sample_rate));
    write_u32(buf, static_cast<uint32_t>(byte_rate));
    write_u16(buf, static_cast<uint16_t>(block_align));
    write_u16(buf, static_cast<uint16_t>(bits_per_sample));

    // data chunk
    write_tag(buf, "data");
    write_u32(buf, data_size);

    // Write float32 samples directly (little-endian, which ARM/x86 are natively)
    const auto* raw = reinterpret_cast<const uint8_t*>(data);
    buf.insert(buf.end(), raw, raw + data_size);

    return buf;
}

std::vector<uint8_t> encode_pcm_16(const float* data, int samples) {
    std::vector<uint8_t> buf;
    buf.reserve(samples * 2);

    for (int i = 0; i < samples; ++i) {
        float clamped = std::fmax(-1.0f, std::fmin(1.0f, data[i]));
        int16_t val = static_cast<int16_t>(clamped * 32767.0f);
        buf.push_back(static_cast<uint8_t>(val & 0xFF));
        buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    }

    return buf;
}

} // namespace audio
