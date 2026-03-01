package com.dark.gguf_lib

/**
 * CharacterEngine - Personality and behavior control.
 *
 * Controls model behavior at the logit level with mood states,
 * personality traits, and sampling parameter adjustments.
 */
class CharacterEngine : AutoCloseable {

    private var nativeHandle: Long = 0L

    init {
        nativeHandle = nativeCreate()
    }

    fun setPersonality(personality: Personality) {
        check(nativeHandle != 0L) { "CharacterEngine already destroyed" }
        nativeSetPersonality(
            nativeHandle,
            personality.name,
            personality.persona,
            personality.temperature,
            personality.topP,
            personality.repetitionPenalty,
            personality.creativity,
            personality.verbosity,
            personality.formality,
        )
    }

    fun setMood(mood: Mood) {
        check(nativeHandle != 0L) { "CharacterEngine already destroyed" }
        nativeSetMood(nativeHandle, mood.ordinal)
    }

    fun getContext(): String {
        check(nativeHandle != 0L) { "CharacterEngine already destroyed" }
        return nativeGetContext(nativeHandle)
    }

    fun getEffectiveParams(): CharacterParams {
        check(nativeHandle != 0L) { "CharacterEngine already destroyed" }
        val vals = nativeGetParams(nativeHandle)
        return CharacterParams(
            temperature = vals[0],
            topP = vals[1],
            minP = vals[2],
            repetitionPenalty = vals[3],
            topK = vals[4].toInt(),
        )
    }

    /**
     * Build SamplingParams from the character engine state.
     * Use this to pass directly to GGMLEngine.generate().
     */
    fun toSamplingParams(maxTokens: Int = 256): SamplingParams {
        val p = getEffectiveParams()
        return SamplingParams(
            temperature = p.temperature,
            topK = p.topK,
            topP = p.topP,
            minP = p.minP,
            repeatPenalty = p.repetitionPenalty,
            maxTokens = maxTokens,
        )
    }

    override fun close() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    // ---- Native methods ----

    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSetPersonality(
        handle: Long, name: String, persona: String,
        temperature: Float, topP: Float, repPenalty: Float,
        creativity: Float, verbosity: Float, formality: Float
    )
    private external fun nativeSetMood(handle: Long, mood: Int)
    private external fun nativeGetContext(handle: Long): String
    private external fun nativeGetParams(handle: Long): FloatArray

    companion object {
        init {
            System.loadLibrary("gguf_lib")
        }
    }
}

// ---- Data classes ----

enum class Mood {
    NEUTRAL, HAPPY, SAD, EXCITED, CALM, ANGRY, CURIOUS, CREATIVE, FOCUSED, CUSTOM
}

data class Personality(
    val name: String,
    val persona: String,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val repetitionPenalty: Float = 1.1f,
    val creativity: Float = 0.5f,
    val verbosity: Float = 0.5f,
    val formality: Float = 0.5f,
)

data class CharacterParams(
    val temperature: Float,
    val topP: Float,
    val minP: Float,
    val repetitionPenalty: Float,
    val topK: Int,
)
