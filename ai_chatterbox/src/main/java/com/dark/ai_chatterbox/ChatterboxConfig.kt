package com.dark.ai_chatterbox

/**
 * Configuration for loading a Chatterbox TTS model.
 */
data class ChatterboxConfig(
    val modelDir: String,
    val tokenizerPath: String,
    val voicePresetDir: String? = null,
    val repetitionPenalty: Float = 1.2f,
    val maxTokens: Int = 1024
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
