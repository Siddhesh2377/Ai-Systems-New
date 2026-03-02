package com.dark.gguf_lib.toolcalling

/**
 * Grammar enforcement mode for tool calling.
 */
enum class GrammarMode(val value: Int) {
    /** Forces JSON tool call output via grammar constraint */
    STRICT(0),
    /** Model chooses between text or tool call freely */
    LAZY(1)
}
