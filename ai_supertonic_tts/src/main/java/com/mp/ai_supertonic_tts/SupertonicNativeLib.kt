package com.mp.ai_supertonic_tts

import androidx.annotation.Keep

/**
 * JNI bridge for native audio processing operations.
 *
 * Provides high-performance C++ implementations for:
 * - WAV file encoding (16-bit PCM and 32-bit float)
 * - Raw PCM encoding
 * - Audio clipping
 */
@Keep
class SupertonicNativeLib {

    /**
     * Encode float32 audio as 16-bit PCM WAV file bytes.
     *
     * @param audio Float32 audio samples in [-1.0, 1.0] range
     * @param sampleRate Sample rate in Hz (e.g. 44100)
     * @param channels Number of audio channels (1 = mono)
     * @return Complete WAV file as byte array (RIFF header + PCM data)
     */
    external fun nativeEncodeWav16(audio: FloatArray, sampleRate: Int, channels: Int): ByteArray

    /**
     * Encode float32 audio as 32-bit IEEE float WAV file bytes.
     *
     * @param audio Float32 audio samples
     * @param sampleRate Sample rate in Hz
     * @param channels Number of audio channels
     * @return Complete WAV file as byte array (RIFF header + float data)
     */
    external fun nativeEncodeWav32f(audio: FloatArray, sampleRate: Int, channels: Int): ByteArray

    /**
     * Encode float32 audio as raw 16-bit PCM bytes (no WAV header).
     *
     * @param audio Float32 audio samples in [-1.0, 1.0] range
     * @return Raw 16-bit signed PCM bytes, little-endian
     */
    external fun nativeEncodePcm16(audio: FloatArray): ByteArray

    /**
     * Clip audio samples in-place to [-1.0, 1.0] range.
     * Modifies the input array directly.
     *
     * @param audio Float32 audio samples to clip
     */
    external fun nativeClipAudio(audio: FloatArray)

    companion object {
        init {
            System.loadLibrary("ai_supertonic_tts")
        }
    }
}
