package com.dark.gguf_lib.models

/**
 * Callback interface for the native agent engine.
 * Called from C++ during agent orchestration (plan→execute→summarize loop).
 */
interface AgentCallback {
    fun onPlan(plan: String)
    fun onToolCall(round: Int, toolName: String, argsJson: String)
    fun onToolResult(round: Int, toolName: String, resultJson: String, success: Boolean, timeMs: Long)
    fun onToken(token: String, isSummary: Boolean)
    fun onSummary(summary: String)
    fun onComplete()
    fun onError(message: String)

    /**
     * Synchronous upcall from native code to execute a tool.
     * C++ blocks until this returns the tool result JSON string.
     */
    fun executeToolFromNative(toolName: String, argsJson: String): String
}
