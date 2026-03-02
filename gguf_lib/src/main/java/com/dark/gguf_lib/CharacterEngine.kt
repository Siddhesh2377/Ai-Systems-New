package com.dark.gguf_lib

import org.json.JSONObject

/**
 * CharacterEngine - Personality, mood, and uncensored mode control.
 *
 * Controls model behavior at the logit/sampling level via the native
 * character engine. Supports personality traits, mood states, control
 * vectors, and uncensored mode (vocabulary-level refusal suppression).
 *
 * Usage:
 * ```
 * val engine = GGMLEngine()
 * val character = CharacterEngine(engine)
 *
 * character.setPersonality(Personality(
 *     name = "Luna",
 *     persona = "A warm, empathetic AI companion",
 *     temperature = 0.8f,
 *     creativity = 0.7f,
 * ))
 *
 * character.setMood(Mood.HAPPY)
 * character.setUncensored(true)
 * ```
 */
class CharacterEngine(private val engine: GGMLEngine) {

    /**
     * Set the character personality.
     * This adjusts sampling parameters (temperature, top_p, repetition penalty)
     * and applies logit biases based on the personality traits.
     */
    fun setPersonality(personality: Personality) {
        val json = JSONObject().apply {
            put("name", personality.name)
            put("persona", personality.persona)
            put("temperature", personality.temperature)
            put("topP", personality.topP)
            put("repetitionPenalty", personality.repetitionPenalty)
            put("creativity", personality.creativity)
            put("verbosity", personality.verbosity)
            put("formality", personality.formality)
        }
        GGUFNativeLib.nativeSetPersonality(json.toString())
    }

    /**
     * Set the mood state. Adjusts sampling parameters for the mood.
     */
    fun setMood(mood: Mood) {
        GGUFNativeLib.nativeSetMood(mood.ordinal)
    }

    /**
     * Get the formatted personality/mood context block.
     * Can be injected into the system prompt for reinforcement.
     */
    fun getContext(): String = GGUFNativeLib.nativeGetCharacterContext()

    /**
     * Enable uncensored mode.
     * Scans the model vocabulary for refusal pattern tokens and applies
     * negative logit biases to suppress them. This is vocabulary-level
     * manipulation, not prompt engineering.
     */
    fun setUncensored(enabled: Boolean) {
        GGUFNativeLib.nativeSetUncensored(enabled)
    }

    val isUncensored: Boolean get() = GGUFNativeLib.nativeGetUncensored()

    /**
     * Load control vectors for fine-grained emotional/behavioral tuning.
     * @param vectors List of control vector configs
     */
    fun loadControlVectors(vectors: List<ControlVectorConfig>): Boolean {
        val json = org.json.JSONArray()
        vectors.forEach { cv ->
            json.put(JSONObject().apply {
                put("path", cv.path)
                put("scale", cv.strength)
            })
        }
        return engine.loadControlVectors(json.toString())
    }

    fun clearControlVectors() = engine.clearControlVector()

    /**
     * Set per-token logit biases for persona control.
     * @param biases Map of token string to bias value
     */
    fun setLogitBias(biases: Map<String, Float>) {
        val json = JSONObject()
        biases.forEach { (token, bias) -> json.put(token, bias) }
        GGUFNativeLib.nativeSetLogitBias(json.toString())
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

data class ControlVectorConfig(
    val path: String,
    val strength: Float = 1.0f,
)
