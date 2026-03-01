package com.dark.backend_plugin_api.capability

/**
 * Tool calling capability — grammar-constrained JSON generation.
 * Implement alongside [TextGenBackend] if the backend supports TOOL_CALLING.
 */
interface ToolCallingBackend {

    /** Enable tool calling with tool definitions JSON */
    fun enableToolCalling(toolsJson: String)

    /** Disable tool calling, revert to free text generation */
    fun disableToolCalling()

    /** Check if the loaded model supports tool calling */
    fun isToolCallingSupported(): Boolean

    /** Set grammar mode: 0 = STRICT (always constrained), 1 = LAZY (activate on detection) */
    fun setGrammarMode(mode: Int)
}
