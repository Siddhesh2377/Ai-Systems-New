package com.dark.gguf_lib

/**
 * Task profile modes for task-aware sampling.
 *
 * Each mode auto-tunes sampling parameters (temperature, top_p, min_p, etc.)
 * for the specific task type. Values must match the C enum in task-profile.h.
 *
 * Architecture: Base params → CharacterEngine → TaskProfile → Sampler
 */
enum class TaskProfileMode(val value: Int) {

    // ── General ──
    /** Default — pass through base params unchanged */
    GENERAL_CHAT(0),
    /** User-defined sampling overrides */
    CUSTOM(1),

    // ── Precision Tasks (low temp, tight sampling) ──
    /** Near-greedy, grammar-locked, structured JSON output */
    TOOL_CALLING(10),
    /** Factual, low temp, suppress hedging — faithful to retrieved context */
    RAG_GROUNDED(11),
    /** Precise, deterministic, focused reasoning */
    ANALYSIS(12),
    /** Follow instructions exactly */
    INSTRUCTION(13),
    /** Structured code output, moderate temp, high min_p */
    CODE_GENERATION(14),
    /** Low temp, concise, high repetition penalty */
    SUMMARIZATION(15),

    // ── Creative Tasks (high temp, loose sampling) ──
    /** Creative, mood-driven, character personality */
    ROLEPLAY(20),
    /** High temp, high top_p, low repetition penalty */
    CREATIVE_WRITING(21),
    /** Very high temp, diverse outputs */
    BRAINSTORMING(22),

    // ── Agent Phases ──
    /** Moderate — structured but not grammar-locked */
    AGENT_PLANNING(30),
    /** Same as TOOL_CALLING — grammar-constrained execution */
    AGENT_EXECUTING(31),
    /** Same as SUMMARIZATION — concise final output */
    AGENT_SUMMARIZING(32);

    companion object {
        private val map = entries.associateBy { it.value }
        fun fromValue(value: Int): TaskProfileMode = map[value] ?: GENERAL_CHAT
    }
}
