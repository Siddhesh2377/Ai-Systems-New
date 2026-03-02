package com.dark.ai_chatterbox

/**
 * Model variant — determines architecture constants and feature availability.
 *
 * TURBO:    GPT-2 Medium 350M, 24 layers, no exaggeration, no CFG
 * ORIGINAL: Llama 500M, 30 layers, exaggeration input on embed_tokens, cfg_weight=0.5
 */
enum class ChatterboxVariant {
    TURBO,    // GPT-2 Medium 350M, 24 layers, no exaggeration
    ORIGINAL  // Llama 500M, 30 layers, has exaggeration
}

/**
 * Configuration for loading a Chatterbox TTS model.
 */
data class ChatterboxConfig(
    val modelDir: String,
    val tokenizerPath: String,
    val voicePresetDir: String? = null,
    val repetitionPenalty: Float = 1.2f,
    val maxTokens: Int = 1024,
    val variant: ChatterboxVariant = ChatterboxVariant.TURBO,
    /** Emotion exaggeration: 0.0=flat, 1.0=normal, 2.0=very expressive. Only effective for ORIGINAL variant. */
    val exaggeration: Float = 1.0f
)

/**
 * State of the Chatterbox TTS engine.
 */
sealed class ChatterboxState {
    data object Idle : ChatterboxState()
    data object Loading : ChatterboxState()
    data object Ready : ChatterboxState()
    data class Generating(val tokensGenerated: Int) : ChatterboxState()
    data class Complete(val pcmData: ShortArray, val sampleRate: Int) : ChatterboxState() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Complete) return false
            return pcmData.contentEquals(other.pcmData) && sampleRate == other.sampleRate
        }
        override fun hashCode(): Int = 31 * pcmData.contentHashCode() + sampleRate
    }
    data class Error(val message: String) : ChatterboxState()
}
