package com.mp.ai_gguf.toolcalling

import com.mp.ai_gguf.GGUFNativeLib
import com.mp.ai_gguf.models.StreamCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

enum class GrammarMode(val value: Int) {
    STRICT(0),
    LAZY(1)
}

data class ToolCallingConfig(
    val maxRounds: Int = 5,
    val grammarMode: GrammarMode = GrammarMode.STRICT,
    val useTypedGrammar: Boolean = true,
    val maxTokensPerTurn: Int = 256
)

data class ToolResult(
    val toolName: String,
    val result: String,
    val isError: Boolean = false
)

fun interface ToolExecutor {
    suspend fun execute(call: ToolCall): ToolResult
}

data class ChatMessage(
    val role: String,
    val content: String
)

class ToolCallManager(private val nativeLib: GGUFNativeLib) {

    private val registeredTools = mutableListOf<ToolDefinition>()
    private var enabled = false

    var lastError: String? = null
        private set

    fun isEnabled(): Boolean = enabled && nativeLib.nativeIsToolCallingEnabled()

    fun getRegisteredTools(): List<ToolDefinition> = registeredTools.toList()

    fun registerTool(tool: ToolDefinition): Boolean {
        if (registeredTools.any { it.name == tool.name }) {
            lastError = "Tool '${tool.name}' is already registered"
            return false
        }
        registeredTools.add(tool)
        lastError = null
        return true
    }

    fun registerTools(vararg tools: ToolDefinition): Boolean = tools.all { registerTool(it) }

    fun unregisterTool(name: String): Boolean {
        val removed = registeredTools.removeIf { it.name == name }
        if (removed && enabled) {
            disable()
            enable()
        }
        return removed
    }

    fun clearTools() {
        registeredTools.clear()
        if (enabled) disable()
    }

    fun enable(config: ToolCallingConfig = ToolCallingConfig()): Boolean {
        try {
            if (!nativeLib.nativeIsLoaded()) {
                lastError = "No model loaded"
                return false
            }

            if (registeredTools.isEmpty()) {
                lastError = "No tools registered. Call registerTool() first."
                return false
            }

            nativeLib.nativeSetGrammarMode(config.grammarMode.value)
            nativeLib.nativeSetSystemPrompt(buildMinimalSystemPrompt())
            nativeLib.nativeSetToolsJson(buildToolsJson())
            nativeLib.nativeEnableToolCalling(true)

            enabled = true
            lastError = null
            return true
        } catch (e: Exception) {
            lastError = "Failed to enable tool calling: ${e.message}"
            enabled = false
            return false
        }
    }

    fun disable() {
        nativeLib.nativeEnableToolCalling(false)
        enabled = false
        lastError = null
    }

    fun reset() {
        disable()
        clearTools()
    }

    private fun buildToolsJson(): String {
        val toolsArray = JSONArray()
        registeredTools.forEach { toolsArray.put(it.toOpenAIFormat()) }
        return toolsArray.toString()
    }

    fun parseToolCall(jsonResponse: String): ToolCall? {
        return try {
            val json = JSONObject(jsonResponse)
            val toolCalls = json.optJSONArray("tool_calls") ?: return null
            if (toolCalls.length() == 0) return null

            val firstCall = toolCalls.getJSONObject(0)
            ToolCall(firstCall.getString("name"), firstCall.getJSONObject("arguments"))
        } catch (e: Exception) {
            lastError = "Failed to parse tool call: ${e.message}"
            null
        }
    }

    suspend fun generateWithTools(
        userMessage: String,
        executor: ToolExecutor,
        config: ToolCallingConfig = ToolCallingConfig(),
        onToken: (String) -> Unit = {},
        onToolCallDetected: (ToolCall) -> Unit = {},
        onError: (String) -> Unit = {},
        onDone: (String) -> Unit = {}
    ) {
        if (!isEnabled()) {
            onError("Tool calling not enabled. Call enable() first.")
            return
        }

        withContext(Dispatchers.IO) {
            val messages = mutableListOf<ChatMessage>()

            val systemContent = buildString {
                append(buildMinimalSystemPrompt())
                append("\nYou may call tools by emitting ONLY the JSON object:\n")
                append("{\"tool_calls\":[{\"name\":\"NAME\",\"arguments\":{...}}]}\n")
                append("Available tools (OpenAI schema):\n")
                append(buildToolsJson())
            }
            messages.add(ChatMessage("system", systemContent))
            messages.add(ChatMessage("user", userMessage))

            for (round in 0 until config.maxRounds) {
                val messagesJson = buildMessagesJson(messages)
                val roundText = StringBuilder()
                var detectedToolCall: Pair<String, String>? = null
                var errorMsg: String? = null

                val callback = object : StreamCallback {
                    override fun onToken(token: String) {
                        roundText.append(token)
                        onToken(token)
                    }
                    override fun onToolCall(name: String, argsJson: String) {
                        detectedToolCall = name to argsJson
                    }
                    override fun onDone() {}
                    override fun onError(message: String) { errorMsg = message }
                }

                nativeLib.nativeGenerateStreamMultiTurn(
                    messagesJson, config.maxTokensPerTurn, callback
                )

                val error = errorMsg
                if (error != null) {
                    onError(error)
                    return@withContext
                }

                val toolCallPair = detectedToolCall
                if (toolCallPair != null) {
                    val (toolName, payload) = toolCallPair
                    val toolCall = parseToolCall(payload)
                    if (toolCall == null) {
                        onError("Failed to parse tool call from model output")
                        return@withContext
                    }
                    onToolCallDetected(toolCall)
                    messages.add(ChatMessage("assistant", payload))

                    val toolResult = try {
                        executor.execute(toolCall)
                    } catch (e: Exception) {
                        ToolResult(toolName, "Error: ${e.message}", isError = true)
                    }
                    messages.add(ChatMessage("tool", toolResult.result))
                    continue
                }

                onDone(roundText.toString())
                return@withContext
            }

            onError("Maximum tool call rounds exceeded (${config.maxRounds})")
        }
    }

    private fun buildMessagesJson(messages: List<ChatMessage>): String {
        val array = JSONArray()
        for (msg in messages) {
            array.put(JSONObject().apply {
                put("role", msg.role)
                put("content", msg.content)
            })
        }
        return array.toString()
    }

    companion object {
        fun buildMinimalSystemPrompt(): String {
            return "You are a helpful assistant with access to tools. " +
                    "When a tool is needed, respond with the tool call JSON. " +
                    "Otherwise, respond in plain text."
        }

        fun withCommonTools(nativeLib: GGUFNativeLib): ToolCallManager {
            return ToolCallManager(nativeLib).apply {
                registerTools(
                    tool("get_current_time", "Get the current time") {
                        stringParam("format", "Time format: 'full', 'time', or 'date'",
                            enum = listOf("full", "time", "date"))
                    },
                    tool("show_message", "Display a message to the user") {
                        stringParam("message", "The message to display", required = true)
                        stringParam("duration", "How long to show",
                            enum = listOf("short", "long"))
                    },
                    tool("get_device_info", "Get information about the device") {
                        stringParam("info_type", "Type of info: 'basic', 'system', or 'all'",
                            enum = listOf("basic", "system", "all"))
                    }
                )
            }
        }
    }
}

data class ToolCall(
    val name: String,
    val arguments: JSONObject
) {
    fun getString(key: String, default: String = ""): String = arguments.optString(key, default)
    fun getInt(key: String, default: Int = 0): Int = arguments.optInt(key, default)
    fun getBoolean(key: String, default: Boolean = false): Boolean = arguments.optBoolean(key, default)
    fun getDouble(key: String, default: Double = 0.0): Double = arguments.optDouble(key, default)
    fun has(key: String): Boolean = arguments.has(key)

    fun toJson(): JSONObject = JSONObject().apply {
        put("name", name)
        put("arguments", arguments)
    }

    override fun toString(): String = "ToolCall(name='$name', arguments=$arguments)"
}
