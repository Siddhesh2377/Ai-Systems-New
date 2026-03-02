#pragma once

/**
 * Header-only WAV file writer for 16-bit PCM audio.
 * Ported from DDATT/Chatterbox-turbo-cpp wavfile.hpp.
 *
 * Writes standard RIFF WAV format (PCM, 16-bit, mono/stereo).
 * Default: 24kHz mono (Chatterbox TTS output format).
 */

#include <fstream>
#include <cstdint>
#include <vector>
#include <string>

namespace wav {

struct WavHeader {
    uint8_t  RIFF[4]       = {'R', 'I', 'F', 'F'};
    uint32_t chunkSize      = 0;
    uint8_t  WAVE[4]       = {'W', 'A', 'V', 'E'};
    uint8_t  fmt[4]        = {'f', 'm', 't', ' '};
    uint32_t fmtSize        = 16;
    uint16_t audioFormat    = 1;  // PCM
    uint16_t numChannels    = 1;
    uint32_t sampleRate     = 24000;
    uint32_t bytesPerSec    = 0;
    uint16_t blockAlign     = 2;
    uint16_t bitsPerSample  = 16;
    uint8_t  data[4]       = {'d', 'a', 't', 'a'};
    uint32_t dataSize       = 0;
};

static_assert(sizeof(WavHeader) == 44, "WAV header must be exactly 44 bytes");

/**
 * Write PCM int16 audio data to a WAV file.
 *
 * @param path       Output file path
 * @param data       Pointer to int16 PCM samples
 * @param numSamples Number of samples (per channel)
 * @param sampleRate Sample rate in Hz (default 24000 for Chatterbox)
 * @param channels   Number of channels (default 1 = mono)
 * @return true on success, false on I/O error
 */
inline bool writeWav(const std::string& path, const int16_t* data,
                     int numSamples, int sampleRate = 24000, int channels = 1) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    WavHeader header;
    header.dataSize     = static_cast<uint32_t>(numSamples * sizeof(int16_t) * channels);
    header.chunkSize    = header.dataSize + sizeof(WavHeader) - 8;
    header.sampleRate   = static_cast<uint32_t>(sampleRate);
    header.numChannels  = static_cast<uint16_t>(channels);
    header.bytesPerSec  = static_cast<uint32_t>(sampleRate * sizeof(int16_t) * channels);
    header.blockAlign   = static_cast<uint16_t>(sizeof(int16_t) * channels);

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(data), header.dataSize);
    return file.good();
}

/**
 * Convenience overload taking a vector of int16 samples.
 */
inline bool writeWav(const std::string& path, const std::vector<int16_t>& samples,
                     int sampleRate = 24000, int channels = 1) {
    return writeWav(path, samples.data(), static_cast<int>(samples.size()), sampleRate, channels);
}

} // namespace wav
