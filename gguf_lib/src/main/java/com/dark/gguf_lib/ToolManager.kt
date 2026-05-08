package com.dark.gguf_lib

import com.dark.gguf_lib.toolcalling.GrammarMode
import com.dark.gguf_lib.toolcalling.ToolCallingConfig
import com.dark.gguf_lib.toolcalling.ToolDefinitionBuilder
import org.json.JSONArray

/**
 * High-level tool calling API. Wraps the native dual-strategy parser:
 *
 * 1. The chat-template-aware parser (uses the model's grammar if it has one).
 * 2. A multi-format fallback that detects JSON, XML and function-call shapes.
 *
 * Tool calls surface as [com.dark.gguf_lib.models.GenerationEvent.ToolCall]
 * during generation flows.
 *
 * ```kotlin
 * val toolManager = ToolManager(engine)
 * val getWeather = ToolDefinitionBuilder("get_weather", "Get current weather")
 *     .stringParam("location", "City name")
 *     .build()
 * toolManager.registerTools(listOf(getWeather))
 * ```
 */
class ToolManager(private val engine: GGMLEngine) {

    private val registeredTools = mutableListOf<ToolDefinitionBuilder.ToolDefinition>()
    private var config = ToolCallingConfig()

    /** Replace the registered tools and (re-)apply the calling config. */
    fun registerTools(
        tools: List<ToolDefinitionBuilder.ToolDefinition>,
        config: ToolCallingConfig = ToolCallingConfig(),
    ) {
        this.config = config
        registeredTools.clear()
        registeredTools.addAll(tools)
        engine.enableToolCalling(tools, config)
    }

    /** Add a single tool to the registered set and re-apply. */
    fun registerTool(tool: ToolDefinitionBuilder.ToolDefinition) {
        registeredTools.add(tool)
        engine.enableToolCalling(registeredTools, config)
    }

    /**
     * Switch grammar enforcement mode at runtime. Cheap; safe between turns.
     * STRICT forces the model to emit a tool call; LAZY lets it choose.
     */
    fun setGrammarMode(mode: GrammarMode) {
        config = config.copy(grammarMode = mode)
        GGUFNativeLib.nativeSetGrammarMode(mode.value)
    }

    /** Currently registered tools as OpenAI-format JSON. */
    fun getToolsJson(): String {
        val arr = JSONArray()
        registeredTools.forEach { arr.put(it.toOpenAIFormat()) }
        return arr.toString()
    }

    /** Whether the loaded model's chat template advertises tool calling. */
    fun isSupported(): Boolean = engine.isToolCallingSupported()

    /** Drop all registered tools and disable tool calling. */
    fun clearTools() {
        registeredTools.clear()
        engine.clearTools()
    }

    val toolCount: Int get() = registeredTools.size
}
