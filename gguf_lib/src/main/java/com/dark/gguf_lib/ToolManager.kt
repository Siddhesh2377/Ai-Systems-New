package com.dark.gguf_lib

import com.dark.gguf_lib.toolcalling.GrammarMode
import com.dark.gguf_lib.toolcalling.ToolCall
import com.dark.gguf_lib.toolcalling.ToolCallingConfig
import com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder
import org.json.JSONArray

/**
 * ToolManager - High-level tool calling API.
 *
 * Wraps the native dual-strategy tool calling system:
 * 1. Template-aware parsing (llama.cpp chat template grammar)
 * 2. ToolManager fallback (JSON + XML + function-call format detection)
 *
 * Usage:
 * ```
 * val engine = GGMLEngine()
 * val toolManager = ToolManager(engine)
 *
 * val weather = ToolDefinitionBuilder("get_weather", "Get current weather")
 *     .stringParam("location", "City name")
 *     .build()
 *
 * toolManager.registerTools(listOf(weather))
 *
 * engine.generateFlow("What's the weather in Tokyo?", 512).collect { event ->
 *     when (event) {
 *         is GenerationEvent.ToolCall -> {
 *             val call = ToolCall.fromJson(event.argsJson)
 *             // Execute tool and feed result back
 *         }
 *     }
 * }
 * ```
 */
class ToolManager(private val engine: GGMLEngine) {

    private val registeredTools = mutableListOf<ToolDefinitionBuilder.ToolDefinition>()
    private var config = ToolCallingConfig()

    /**
     * Register tools and enable tool calling on the engine.
     */
    fun registerTools(
        tools: List<ToolDefinitionBuilder.ToolDefinition>,
        config: ToolCallingConfig = ToolCallingConfig(),
    ) {
        this.config = config
        registeredTools.clear()
        registeredTools.addAll(tools)
        engine.enableToolCalling(tools, config)
    }

    /**
     * Register a single tool.
     */
    fun registerTool(tool: ToolDefinitionBuilder.ToolDefinition) {
        registeredTools.add(tool)
        engine.enableToolCalling(registeredTools, config)
    }

    /**
     * Set the grammar enforcement mode.
     */
    fun setGrammarMode(mode: GrammarMode) {
        config = config.copy(grammarMode = mode)
        GGUFNativeLib.nativeSetGrammarMode(mode.value)
    }

    /**
     * Get the current tools as OpenAI-format JSON string.
     */
    fun getToolsJson(): String {
        val arr = JSONArray()
        registeredTools.forEach { arr.put(it.toOpenAIFormat()) }
        return arr.toString()
    }

    /**
     * Check if the loaded model supports tool calling.
     */
    fun isSupported(): Boolean = engine.isToolCallingSupported()

    /**
     * Clear all registered tools and disable tool calling.
     */
    fun clearTools() {
        registeredTools.clear()
        engine.clearTools()
    }

    val toolCount: Int get() = registeredTools.size
}
