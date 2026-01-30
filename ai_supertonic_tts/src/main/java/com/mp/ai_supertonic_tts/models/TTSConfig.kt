package com.mp.ai_supertonic_tts.models

/**
 * Supported languages for Supertonic v2 (multilingual)
 */
enum class Language(val tag: String) {
    EN("en"),
    KO("ko"),
    ES("es"),
    PT("pt"),
    FR("fr");

    companion object {
        fun fromTag(tag: String): Language? = entries.find { it.tag == tag }
    }
}

/**
 * Configuration for text-to-speech synthesis
 *
 * @param speed Playback speed factor. 1.0 = normal, >1.0 = faster, <1.0 = slower.
 *              Default 1.05 per Supertonic recommendation.
 * @param steps Denoising steps for flow-matching. 2 = fast, 5 = high quality, up to 128.
 *              More steps = better quality but slower.
 * @param language Target language for synthesis.
 * @param voice Voice style name (e.g. "F1", "F2", "M1", "M2", etc.)
 * @param useNNAPI Enable NNAPI acceleration (uses device GPU/NPU if available).
 * @param chunkingEnabled Automatically split long text into chunks at sentence boundaries.
 * @param chunkSilenceMs Silence duration between chunks in milliseconds.
 */
data class TTSConfig(
    val speed: Float = 1.05f,
    val steps: Int = 2,
    val language: Language = Language.EN,
    val voice: String = "F1",
    val useNNAPI: Boolean = false,
    val chunkingEnabled: Boolean = true,
    val chunkSilenceMs: Int = 300
)
