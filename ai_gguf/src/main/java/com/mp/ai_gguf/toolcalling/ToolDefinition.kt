package com.mp.ai_gguf.toolcalling

import org.json.JSONArray
import org.json.JSONObject

/**
 * Represents a parameter definition for a tool
 *
 * @param type Parameter type (string, number, boolean, object, array)
 * @param description Description of the parameter
 * @param enum Optional list of allowed values
 * @param properties For object types, the nested properties
 * @param items For array types, the item type definition
 */
data class ToolParameter(
    val type: String,
    val description: String? = null,
    val enum: List<String>? = null,
    val properties: Map<String, ToolParameter>? = null,
    val items: ToolParameter? = null
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("type", type)
            description?.let { put("description", it) }
            enum?.let { put("enum", JSONArray(it)) }
            properties?.let { props ->
                put("properties", JSONObject().apply {
                    props.forEach { (key, value) ->
                        put(key, value.toJson())
                    }
                })
            }
            items?.let { put("items", it.toJson()) }
        }
    }
}

/**
 * Represents a complete tool/function definition
 *
 * @param name Unique tool name (alphanumeric + underscores only)
 * @param description Clear description of what the tool does
 * @param parameters Map of parameter name to parameter definition
 * @param required List of required parameter names
 */
data class ToolDefinition(
    val name: String,
    val description: String,
    val parameters: Map<String, ToolParameter> = emptyMap(),
    val required: List<String> = emptyList()
) {
    init {
        require(name.matches(Regex("^[a-zA-Z0-9_]+$"))) {
            "Tool name must be alphanumeric with underscores only: $name"
        }
        require(description.isNotBlank()) {
            "Tool description cannot be blank"
        }
        required.forEach { requiredParam ->
            require(parameters.containsKey(requiredParam)) {
                "Required parameter '$requiredParam' not defined in parameters"
            }
        }
    }

    /**
     * Convert to OpenAI-compatible tool definition JSON
     */
    fun toOpenAIFormat(): JSONObject {
        val parametersObj = JSONObject().apply {
            put("type", "object")
            put("properties", JSONObject().apply {
                parameters.forEach { (key, value) ->
                    put(key, value.toJson())
                }
            })
            if (required.isNotEmpty()) {
                put("required", JSONArray(required))
            }
        }

        val functionObj = JSONObject().apply {
            put("name", name)
            put("description", description)
            put("parameters", parametersObj)
        }

        return JSONObject().apply {
            put("type", "function")
            put("function", functionObj)
        }
    }

    companion object {
        /**
         * Create a simple tool with primitive parameters
         *
         * @param name Tool name
         * @param description Tool description
         * @param params Map of parameter name to (type, description, required)
         */
        fun simple(
            name: String,
            description: String,
            vararg params: Triple<String, String, Boolean>
        ): ToolDefinition {
            val parameters = params.associate { (paramName, paramDesc, _) ->
                paramName to ToolParameter(
                    type = inferType(paramName),
                    description = paramDesc
                )
            }
            val required = params.filter { it.third }.map { it.first }

            return ToolDefinition(name, description, parameters, required)
        }

        private fun inferType(paramName: String): String {
            return when {
                paramName.contains("count", ignoreCase = true) -> "number"
                paramName.contains("is_", ignoreCase = true) -> "boolean"
                paramName.contains("enabled", ignoreCase = true) -> "boolean"
                else -> "string"
            }
        }
    }
}

/**
 * Builder for tool definitions
 */
class ToolDefinitionBuilder(
    private val name: String,
    private val description: String
) {
    private val parameters = mutableMapOf<String, ToolParameter>()
    private val required = mutableListOf<String>()

    fun stringParam(
        name: String,
        description: String,
        required: Boolean = false,
        enum: List<String>? = null
    ) = apply {
        parameters[name] = ToolParameter("string", description, enum)
        if (required) this.required.add(name)
    }

    fun numberParam(
        name: String,
        description: String,
        required: Boolean = false
    ) = apply {
        parameters[name] = ToolParameter("number", description)
        if (required) this.required.add(name)
    }

    fun booleanParam(
        name: String,
        description: String,
        required: Boolean = false
    ) = apply {
        parameters[name] = ToolParameter("boolean", description)
        if (required) this.required.add(name)
    }

    fun objectParam(
        name: String,
        description: String,
        properties: Map<String, ToolParameter>,
        required: Boolean = false
    ) = apply {
        parameters[name] = ToolParameter("object", description, properties = properties)
        if (required) this.required.add(name)
    }

    fun arrayParam(
        name: String,
        description: String,
        items: ToolParameter,
        required: Boolean = false
    ) = apply {
        parameters[name] = ToolParameter("array", description, items = items)
        if (required) this.required.add(name)
    }

    fun build(): ToolDefinition {
        return ToolDefinition(name, description, parameters, required)
    }
}

/**
 * DSL for building tool definitions
 */
fun tool(name: String, description: String, builder: ToolDefinitionBuilder.() -> Unit): ToolDefinition {
    return ToolDefinitionBuilder(name, description).apply(builder).build()
}
