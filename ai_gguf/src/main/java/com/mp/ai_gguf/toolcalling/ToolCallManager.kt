package com.mp.ai_gguf.toolcalling

import com.mp.ai_gguf.GGUFNativeLib
import org.json.JSONArray
import org.json.JSONObject

/**
 * Professional SDK for managing tool calling with Qwen models
 *
 * This SDK provides a clean interface for:
 * - Registering tools
 * - Enabling/disabling tool calling mode
 * - Validating model compatibility
 * - Managing tool calling state
 *
 * IMPORTANT: Tool calling is ONLY supported for Qwen models.
 * Attempting to enable tool calling on other architectures will fail.
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
 * // Enable tool calling
 * if (toolCallManager.enable()) {
 *     println("Tool calling enabled!")
 * } else {
 *     println("Failed: ${toolCallManager.lastError}")
 * }
 *
 * // Later, disable tool calling to use normal chat
 * toolCallManager.disable()
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
     * Check if the loaded model supports tool calling
     *
     * @return true if model is a Qwen model
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
     * 1. Check if model is compatible (Qwen only)
     * 2. Build tools JSON from registered tools
     * 3. Configure the model for tool calling
     *
     * @return true if tool calling was enabled successfully
     */
    fun enable(): Boolean {
        try {
            // Check model compatibility
            if (!isModelCompatible()) {
                val arch = modelArchitecture
                lastError = if (arch.isEmpty()) {
                    "No model loaded"
                } else {
                    "Tool calling only supported for Qwen models, current model: $arch"
                }
                return false
            }

            // Check if we have tools
            if (registeredTools.isEmpty()) {
                lastError = "No tools registered. Call registerTool() first."
                return false
            }

            // Build tools JSON
            val toolsJson = buildToolsJson()

            // Enable via native
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

    companion object {
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
