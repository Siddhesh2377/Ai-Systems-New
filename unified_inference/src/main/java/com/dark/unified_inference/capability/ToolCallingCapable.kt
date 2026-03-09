package com.dark.unified_inference.capability

interface ToolCallingCapable {
    fun isToolCallingSupported(): Boolean
    fun enableToolCalling(toolsJson: String, grammarMode: Int, useTypedGrammar: Boolean): Boolean
    fun clearTools()
}
