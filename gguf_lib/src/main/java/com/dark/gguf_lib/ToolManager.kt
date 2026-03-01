package com.dark.gguf_lib

/**
 * ToolManager - Model-agnostic tool calling system.
 *
 * Register tools with JSON schema, get tool-description prompts to inject
 * into conversations, and parse model outputs for tool calls.
 */
class ToolManager : AutoCloseable {

    private var nativeHandle: Long = 0L

    init {
        nativeHandle = nativeCreate()
    }

    fun registerTool(tool: ToolDefinition) {
        check(nativeHandle != 0L) { "ToolManager already destroyed" }
        nativeRegisterTool(
            nativeHandle,
            tool.name,
            tool.description,
            tool.params.map { it.name }.toTypedArray(),
            tool.params.map { it.description }.toTypedArray(),
            tool.params.map { it.type.ordinal }.toIntArray(),
            tool.params.map { it.required }.toBooleanArray(),
        )
    }

    fun getToolPrompt(): String {
        check(nativeHandle != 0L) { "ToolManager already destroyed" }
        return nativeGetPrompt(nativeHandle)
    }

    /**
     * Parse model output for tool calls.
     * Returns null if no tool call was detected.
     */
    fun parseOutput(modelOutput: String): ToolCallResult? {
        check(nativeHandle != 0L) { "ToolManager already destroyed" }
        val json = nativeParseOutput(nativeHandle, modelOutput) ?: return null
        return ToolCallResult(json)
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
    private external fun nativeRegisterTool(
        handle: Long, name: String, description: String,
        paramNames: Array<String>, paramDescs: Array<String>,
        paramTypes: IntArray, paramRequired: BooleanArray
    )
    private external fun nativeGetPrompt(handle: Long): String
    private external fun nativeParseOutput(handle: Long, output: String): String?

    companion object {
        init {
            System.loadLibrary("gguf_lib")
        }
    }
}

// ---- Data classes ----

enum class ToolParamType {
    STRING, NUMBER, BOOLEAN, ARRAY, OBJECT
}

data class ToolParam(
    val name: String,
    val description: String,
    val type: ToolParamType = ToolParamType.STRING,
    val required: Boolean = true,
)

data class ToolDefinition(
    val name: String,
    val description: String,
    val params: List<ToolParam> = emptyList(),
)

data class ToolCallResult(
    val rawJson: String,
) {
    val toolName: String
        get() {
            val match = Regex("\"tool\"\\s*:\\s*\"([^\"]+)\"").find(rawJson)
            return match?.groupValues?.get(1) ?: ""
        }

    val argumentsJson: String
        get() {
            val idx = rawJson.indexOf("\"arguments\"")
            if (idx < 0) return "{}"
            val start = rawJson.indexOf('{', idx + 11)
            if (start < 0) return "{}"
            var depth = 0
            for (i in start until rawJson.length) {
                when (rawJson[i]) {
                    '{' -> depth++
                    '}' -> { depth--; if (depth == 0) return rawJson.substring(start, i + 1) }
                }
            }
            return "{}"
        }
}
