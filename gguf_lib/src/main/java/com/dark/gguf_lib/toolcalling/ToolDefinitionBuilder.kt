package com.dark.gguf_lib.toolcalling

import org.json.JSONArray
import org.json.JSONObject

/**
 * Builder for tool definitions in OpenAI function-calling format.
 *
 * Usage:
 * ```
 * val tool = ToolDefinitionBuilder("get_weather", "Get current weather")
 *     .stringParam("location", "City name", required = true)
 *     .enumParam("unit", "Temperature unit", listOf("celsius", "fahrenheit"))
 *     .build()
 *
 * val json = tool.toOpenAIFormat().toString()
 * ```
 */
class ToolDefinitionBuilder(val name: String, val description: String) {

    private val properties = JSONObject()
    private val required = JSONArray()

    fun stringParam(name: String, description: String, required: Boolean = true): ToolDefinitionBuilder {
        addParam(name, "string", description, required)
        return this
    }

    fun numberParam(name: String, description: String, required: Boolean = true): ToolDefinitionBuilder {
        addParam(name, "number", description, required)
        return this
    }

    fun integerParam(name: String, description: String, required: Boolean = true): ToolDefinitionBuilder {
        addParam(name, "integer", description, required)
        return this
    }

    fun booleanParam(name: String, description: String, required: Boolean = true): ToolDefinitionBuilder {
        addParam(name, "boolean", description, required)
        return this
    }

    fun enumParam(
        name: String, description: String, values: List<String>, required: Boolean = true
    ): ToolDefinitionBuilder {
        val param = JSONObject().apply {
            put("type", "string")
            put("description", description)
            put("enum", JSONArray(values))
        }
        properties.put(name, param)
        if (required) this.required.put(name)
        return this
    }

    fun build(): ToolDefinition = ToolDefinition(name, description, properties, required)

    private fun addParam(name: String, type: String, description: String, required: Boolean) {
        val param = JSONObject().apply {
            put("type", type)
            put("description", description)
        }
        properties.put(name, param)
        if (required) this.required.put(name)
    }

    /**
     * Immutable tool definition ready to be serialized.
     */
    class ToolDefinition(
        val name: String,
        val description: String,
        private val properties: JSONObject,
        private val required: JSONArray,
    ) {
        fun toOpenAIFormat(): JSONObject = JSONObject().apply {
            put("type", "function")
            put("function", JSONObject().apply {
                put("name", name)
                put("description", description)
                put("parameters", JSONObject().apply {
                    put("type", "object")
                    put("properties", properties)
                    put("required", required)
                })
            })
        }
    }
}
