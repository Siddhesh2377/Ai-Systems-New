package com.mp.ai_supertonic_tts.models

/**
 * Audio output format options for synthesis results
 */
enum class AudioFormat {
    /** 16-bit PCM WAV file (standard, compatible with all players) */
    WAV_16,
    /** 32-bit IEEE float WAV file (highest quality, larger file) */
    WAV_32F,
    /** Raw 16-bit PCM bytes (no WAV header) */
    PCM_16,
    /** Raw 32-bit float PCM bytes (no WAV header) */
    PCM_32F,
    /** Raw float array (for custom processing pipelines) */
    RAW_FLOAT
}
