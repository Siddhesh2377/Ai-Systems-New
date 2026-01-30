package com.mp.ai_gguf.toolcalling

import com.mp.ai_gguf.GGUFNativeLib
import com.mp.ai_gguf.models.StreamCallback
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Grammar enforcement mode for tool calling
 */
enum class GrammarMode(val value: Int) {
    /** Grammar active from first token - forces JSON tool call output */
    STRICT(0),
    /** Grammar activates only on "{" trigger - model chooses tool vs text */
    LAZY(1)
}

/**
 * Configuration for multi-turn tool calling
 *
 * @param maxRounds Maximum number of tool call rounds before stopping
 * @param grammarMode Grammar enforcement mode (STRICT forces tool calls, LAZY lets model choose)
 * @param useTypedGrammar Use parameter-aware GBNF (enforces exact param names/types/enums)
 * @param maxTokensPerTurn Maximum tokens to generate per turn
 */
data class ToolCallingConfig(
    val maxRounds: Int = 5,
    val grammarMode: GrammarMode = GrammarMode.STRICT,
    val useTypedGrammar: Boolean = true,
    val maxTokensPerTurn: Int = 256
)

/**
 * Result from executing a tool
 *
 * @param toolName Name of the tool that was executed
 * @param result The result content (will be sent back to the model as a tool message)
 * @param isError Whether the execution resulted in an error
 */
data class ToolResult(
    val toolName: String,
    val result: String,
    val isError: Boolean = false
)

/**
 * Interface for executing tool calls. Implement this to handle tool invocations.
 *
 * Example:
 * ```kotlin
 * val executor = ToolExecutor { call ->
 *     when (call.name) {
 *         "get_weather" -> ToolResult("get_weather", """{"temp": 15, "conditions": "sunny"}""")
 *         else -> ToolResult(call.name, "Unknown tool", isError = true)
 *     }
 * }
 * ```
 */
fun interface ToolExecutor {
    suspend fun execute(call: ToolCall): ToolResult
}

/**
 * Chat message for building multi-turn conversation history
 */
data class ChatMessage(
    val role: String,    // "system", "user", "assistant", "tool"
    val content: String
)

/**
 * SDK for managing tool calling with GGUF models
 *
 * This SDK provides a clean interface for:
 * - Registering tools with typed parameters
 * - Enabling/disabling tool calling mode
 * - Multi-turn tool calling with automatic orchestration
 * - Grammar-constrained generation (STRICT or LAZY mode)
 * - Parameter-aware GBNF grammars
 *
 * Supports any model with a chat template. Grammar enforcement ensures
 * valid JSON output regardless of model architecture.
 *
 * Example usage:
 * ```kotlin
 * val toolCallManager = ToolCallManager(ggufLib)
 *
 * // Register tools
 * toolCallManager.registerTool(
 *     tool("get_weather", "Get current weather for a location") {
 *         stringParam("location", "City name", required = true)
 *         stringParam("units", "Temperature units", enum = listOf("celsius", "fahrenheit"))
 *     }
 * )
 *
 * // Enable tool calling (with optional config)
 * toolCallManager.enable(ToolCallingConfig(grammarMode = GrammarMode.LAZY))
 *
 * // Multi-turn tool calling
 * toolCallManager.generateWithTools(
 *     userMessage = "What's the weather in London?",
 *     executor = { call -> ToolResult(call.name, """{"temp": 15}""") },
 *     onToken = { print(it) },
 *     onDone = { println("\nFinal: $it") }
 * )
 * ```
 */
class ToolCallManager(private val nativeLib: GGUFNativeLib) {

    private val registeredTools = mutableListOf<ToolDefinition>()
    private var enabled = false

    /**
     * Last error message if an operation failed
     */
    var lastError: String? = null
        private set

    /**
     * Get the current model architecture
     */
    val modelArchitecture: String
        get() = nativeLib.nativeGetModelArchitecture()

    /**
     * Check if the loaded model supports tool calling.
     *
     * Returns true for any model with a chat template.
     * Grammar enforcement ensures valid JSON output regardless of architecture.
     */
    fun isModelCompatible(): Boolean {
        return nativeLib.nativeIsToolCallingSupported()
    }

    /**
     * Check if tool calling is currently enabled
     */
    fun isEnabled(): Boolean {
        return enabled && nativeLib.nativeIsToolCallingEnabled()
    }

    /**
     * Get list of registered tools
     */
    fun getRegisteredTools(): List<ToolDefinition> {
        return registeredTools.toList()
    }

    /**
     * Register a tool definition
     *
     * Tools must be registered before calling enable()
     *
     * @param tool Tool definition to register
     * @return true if registration succeeded
     */
    fun registerTool(tool: ToolDefinition): Boolean {
        try {
            // Check if tool with same name already exists
            if (registeredTools.any { it.name == tool.name }) {
                lastError = "Tool '${tool.name}' is already registered"
                return false
            }

            registeredTools.add(tool)
            lastError = null
            return true
        } catch (e: Exception) {
            lastError = "Failed to register tool: ${e.message}"
            return false
        }
    }

    /**
     * Register multiple tools at once
     *
     * @param tools List of tool definitions
     * @return true if all tools were registered successfully
     */
    fun registerTools(vararg tools: ToolDefinition): Boolean {
        return tools.all { registerTool(it) }
    }

    /**
     * Unregister a tool by name
     *
     * Note: You must call enable() again after modifying tools
     *
     * @param name Tool name to unregister
     * @return true if tool was found and removed
     */
    fun unregisterTool(name: String): Boolean {
        val removed = registeredTools.removeIf { it.name == name }
        if (removed && enabled) {
            // Re-enable with updated tools
            disable()
            enable()
        }
        return removed
    }

    /**
     * Clear all registered tools
     *
     * This also disables tool calling if it was enabled
     */
    fun clearTools() {
        registeredTools.clear()
        if (enabled) {
            disable()
        }
    }

    /**
     * Enable tool calling mode
     *
     * This will:
     * 1. Check if model is compatible (has chat template)
     * 2. Apply grammar configuration (mode + typed grammar)
     * 3. Set a minimal system prompt for tool calling
     * 4. Build tools JSON and initialize grammar sampler
     *
     * @param config Optional configuration for grammar mode, typed grammar, etc.
     * @return true if tool calling was enabled successfully
     */
    fun enable(config: ToolCallingConfig = ToolCallingConfig()): Boolean {
        try {
            // Check model compatibility
            if (!isModelCompatible()) {
                val arch = modelArchitecture
                lastError = if (arch.isEmpty()) {
                    "No model loaded"
                } else {
                    "Model does not have a chat template, current architecture: $arch"
                }
                return false
            }

            // Check if we have tools
            if (registeredTools.isEmpty()) {
                lastError = "No tools registered. Call registerTool() first."
                return false
            }

            // Apply grammar configuration
            nativeLib.nativeSetGrammarMode(config.grammarMode.value)
            nativeLib.nativeSetTypedGrammar(config.useTypedGrammar)

            // Set minimal system prompt for tool calling
            nativeLib.nativeSetSystemPrompt(buildMinimalSystemPrompt())

            // Build tools JSON and enable
            val toolsJson = buildToolsJson()
            val success = nativeLib.nativeEnableToolCalling(toolsJson)
            if (!success) {
                lastError = "Native tool calling initialization failed"
                return false
            }

            enabled = true
            lastError = null
            return true
        } catch (e: Exception) {
            lastError = "Failed to enable tool calling: ${e.message}"
            enabled = false
            return false
        }
    }

    /**
     * Disable tool calling and revert to normal chat mode
     *
     * This clears the tool calling configuration but keeps registered tools
     * so you can enable() again later without re-registering
     */
    fun disable() {
        nativeLib.nativeDisableToolCalling()
        enabled = false
        lastError = null
    }

    /**
     * Reset tool calling completely
     *
     * This disables tool calling and clears all registered tools
     */
    fun reset() {
        disable()
        clearTools()
    }

    /**
     * Build OpenAI-compatible tools JSON array
     */
    private fun buildToolsJson(): String {
        val toolsArray = JSONArray()
        registeredTools.forEach { tool ->
            toolsArray.put(tool.toOpenAIFormat())
        }
        return toolsArray.toString()
    }

    /**
     * Parse a tool call response
     *
     * @param jsonResponse Raw JSON response from the model
     * @return Parsed tool call, or null if parsing failed
     */
    fun parseToolCall(jsonResponse: String): ToolCall? {
        return try {
            val json = JSONObject(jsonResponse)
            val toolCalls = json.optJSONArray("tool_calls") ?: return null
            if (toolCalls.length() == 0) return null

            val firstCall = toolCalls.getJSONObject(0)
            val name = firstCall.getString("name")
            val arguments = firstCall.getJSONObject("arguments")

            ToolCall(name, arguments)
        } catch (e: Exception) {
            lastError = "Failed to parse tool call: ${e.message}"
            null
        }
    }

    // ========================================================================
    // MULTI-TURN TOOL CALLING ORCHESTRATOR
    // ========================================================================

    /**
     * Run a multi-turn tool calling conversation.
     *
     * This method orchestrates the full tool calling loop:
     * 1. Send user message with system prompt + tool definitions
     * 2. If model outputs a tool call: execute it, feed result back, repeat
     * 3. If model outputs text: return it as the final response
     * 4. Repeat up to [ToolCallingConfig.maxRounds] times
     *
     * Each turn clears the KV cache and re-encodes the full conversation.
     * On CPU, prefill runs at 100-300 t/s so this costs ~2-5s per turn
     * for 500-1000 token conversations.
     *
     * @param userMessage The user's input message
     * @param executor Implementation that handles tool call execution
     * @param config Configuration for grammar mode, max rounds, etc.
     * @param onToken Called for each streamed token (on IO thread)
     * @param onToolCallDetected Called when a tool call is detected (before execution)
     * @param onError Called on error (generation failure, parse failure, max rounds)
     * @param onDone Called with the final text response
     */
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

            // System message with tool instructions and available tools
            val systemContent = buildString {
                append(buildMinimalSystemPrompt())
                append("\n")
                append("You may call tools by emitting ONLY the JSON object:\n")
                append("{\"tool_calls\":[{\"name\":\"NAME\",\"arguments\":{...}}]}\n")
                append("Available tools (OpenAI schema):\n")
                append(buildToolsJson())
            }
            messages.add(ChatMessage("system", systemContent))

            // User message
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

                    override fun onError(message: String) {
                        errorMsg = message
                    }
                }

                nativeLib.nativeGenerateStreamMultiTurn(
                    messagesJson, config.maxTokensPerTurn, callback
                )

                // Check for generation error
                val error = errorMsg
                if (error != null) {
                    onError(error)
                    return@withContext
                }

                // Check for tool call
                val toolCallPair = detectedToolCall
                if (toolCallPair != null) {
                    val (toolName, payload) = toolCallPair

                    val toolCall = parseToolCall(payload)
                    if (toolCall == null) {
                        onError("Failed to parse tool call from model output")
                        return@withContext
                    }

                    onToolCallDetected(toolCall)

                    // Add assistant message with the raw tool call JSON
                    messages.add(ChatMessage("assistant", payload))

                    // Execute the tool
                    val toolResult = try {
                        executor.execute(toolCall)
                    } catch (e: Exception) {
                        ToolResult(toolName, "Error: ${e.message}", isError = true)
                    }

                    // Add tool result message
                    messages.add(ChatMessage("tool", toolResult.result))

                    // Continue to next round
                    continue
                }

                // No tool call detected - this is a text response
                val response = roundText.toString()
                onDone(response)
                return@withContext
            }

            // Max rounds exceeded
            onError("Maximum tool call rounds exceeded (${config.maxRounds})")
        }
    }

    /**
     * Build JSON array of messages for the native multi-turn function
     */
    private fun buildMessagesJson(messages: List<ChatMessage>): String {
        val array = JSONArray()
        for (msg in messages) {
            val obj = JSONObject()
            obj.put("role", msg.role)
            obj.put("content", msg.content)
            array.put(obj)
        }
        return array.toString()
    }

    companion object {
        /**
         * Build a minimal system prompt for tool calling (~30 tokens).
         * Keeps context usage low on small models.
         */
        fun buildMinimalSystemPrompt(): String {
            return "You are a helpful assistant with access to tools. " +
                    "When a tool is needed, respond with the tool call JSON. " +
                    "Otherwise, respond in plain text."
        }

        /**
         * Create a ToolCallManager with pre-registered common tools
         *
         * Common tools included:
         * - get_current_time
         * - get_current_date
         * - show_message
         *
         * @param nativeLib GGUFNativeLib instance
         * @return ToolCallManager with common tools registered
         */
        fun withCommonTools(nativeLib: GGUFNativeLib): ToolCallManager {
            return ToolCallManager(nativeLib).apply {
                registerTools(
                    tool("get_current_time", "Get the current time") {
                        stringParam(
                            "format",
                            "Time format: 'full' for complete, 'time' for time only, 'date' for date only",
                            enum = listOf("full", "time", "date")
                        )
                    },
                    tool("show_message", "Display a message to the user") {
                        stringParam("message", "The message to display", required = true)
                        stringParam(
                            "duration",
                            "How long to show the message",
                            enum = listOf("short", "long")
                        )
                    },
                    tool("get_device_info", "Get information about the device") {
                        stringParam(
                            "info_type",
                            "Type of info: 'basic', 'system', or 'all'",
                            enum = listOf("basic", "system", "all")
                        )
                    }
                )
            }
        }
    }
}

/**
 * Represents a parsed tool call from the model
 *
 * @param name Tool name that was called
 * @param arguments JSON object containing the tool arguments
 */
data class ToolCall(
    val name: String,
    val arguments: JSONObject
) {
    /**
     * Get a string argument
     */
    fun getString(key: String, default: String = ""): String {
        return arguments.optString(key, default)
    }

    /**
     * Get an integer argument
     */
    fun getInt(key: String, default: Int = 0): Int {
        return arguments.optInt(key, default)
    }

    /**
     * Get a boolean argument
     */
    fun getBoolean(key: String, default: Boolean = false): Boolean {
        return arguments.optBoolean(key, default)
    }

    /**
     * Get a double argument
     */
    fun getDouble(key: String, default: Double = 0.0): Double {
        return arguments.optDouble(key, default)
    }

    /**
     * Check if an argument exists
     */
    fun has(key: String): Boolean {
        return arguments.has(key)
    }

    /**
     * Get raw JSON object
     */
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("name", name)
            put("arguments", arguments)
        }
    }

    override fun toString(): String {
        return "ToolCall(name='$name', arguments=$arguments)"
    }
}
