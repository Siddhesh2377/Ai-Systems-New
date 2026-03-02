package com.dark.gguf_lib.toolcalling

/**
 * Configuration for the tool calling system.
 */
data class ToolCallingConfig(
    val grammarMode: GrammarMode = GrammarMode.LAZY,
    val useTypedGrammar: Boolean = true,
    val maxRounds: Int = 3,
    val maxTokensPerTurn: Int = 512,
)
