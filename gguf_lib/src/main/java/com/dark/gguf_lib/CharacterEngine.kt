package com.dark.gguf_lib

import org.json.JSONArray
import org.json.JSONObject

/**
 * Sampler-level personality, mood, and uncensored-mode controls.
 *
 * No separate model, no extra memory — every knob here drives [GGMLEngine]'s
 * sampler params (temperature, top_p, repetition penalty, logit biases) or
 * loads control vector adapters into the live `llama_context`.
 *
 * ```kotlin
 * val character = CharacterEngine(engine)
 * character.setPersonality(Personality(name = "Luna", persona = "...", temperature = 0.8f))
 * character.setMood(Mood.HAPPY)
 * character.setUncensored(true)
 * ```
 */
class CharacterEngine(private val engine: GGMLEngine) {

    /**
     * Apply a personality preset. Maps personality traits to sampler params:
     * `temperature`, `topP`, `repetitionPenalty`, and a creativity-derived
     * `min_p` (higher creativity → looser min_p filter).
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

    /** Set the mood preset. Adjusts temperature and repetition penalty. */
    fun setMood(mood: Mood) = GGUFNativeLib.nativeSetMood(mood.ordinal)

    /**
     * Returns the current sampling state as a JSON snapshot
     * (`{"temperature":...,"top_p":...,"penalty":...}`). Useful for system
     * prompt injection or UI display.
     */
    fun getContext(): String = GGUFNativeLib.nativeGetCharacterContext()

    /**
     * Toggle uncensored mode. When enabled, scans the model vocabulary on
     * first call for refusal-pattern tokens (cached afterwards) and applies
     * a strong negative logit bias to suppress them. Operates entirely at
     * the vocabulary level — no prompt engineering.
     */
    fun setUncensored(enabled: Boolean) = GGUFNativeLib.nativeSetUncensored(enabled)

    val isUncensored: Boolean get() = GGUFNativeLib.nativeGetUncensored()

    /**
     * Load control vectors for fine-grained behavioural tuning.
     *
     * @param vectors Each entry's `path` points to a control-vector GGUF;
     *                `strength` scales it (1.0 = as trained).
     */
    fun loadControlVectors(vectors: List<ControlVectorConfig>): Boolean {
        val json = JSONArray()
        vectors.forEach { cv ->
            json.put(JSONObject().apply {
                put("path", cv.path)
                put("scale", cv.strength)
            })
        }
        return engine.loadControlVectors(json.toString())
    }

    /** Detach any active control vectors. */
    fun clearControlVectors() = engine.clearControlVector()

    /** Apply per-token logit biases. Map keys may be token IDs (as strings) or token text. */
    fun setLogitBias(biases: Map<String, Float>) {
        val json = JSONObject()
        biases.forEach { (token, bias) -> json.put(token, bias) }
        GGUFNativeLib.nativeSetLogitBias(json.toString())
    }
}

/** Pre-baked mood presets. Index matches the C++ side; do not reorder. */
enum class Mood {
    NEUTRAL, HAPPY, SAD, EXCITED, CALM, ANGRY, CURIOUS, CREATIVE, FOCUSED, CUSTOM
}

/**
 * Personality preset.
 *
 * @param name Display name; not used by sampling, surfaces in [CharacterEngine.getContext].
 * @param persona Free-form description; not used by sampling.
 * @param temperature 0.0 = deterministic, 2.0 = chaotic. Typical 0.6-1.0.
 * @param topP Nucleus sampling cutoff.
 * @param repetitionPenalty 1.0 = no penalty.
 * @param creativity 0..1 — drives min_p inversely (higher → lower min_p).
 * @param verbosity 0..1 — currently advisory (passed through but not applied).
 * @param formality  0..1 — currently advisory (passed through but not applied).
 */
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

/**
 * Reference to a control-vector GGUF on disk plus an application strength.
 *
 * @param path Absolute path to the control-vector GGUF.
 * @param strength Multiplier applied to the vector. 1.0 = as trained.
 */
data class ControlVectorConfig(
    val path: String,
    val strength: Float = 1.0f,
)
