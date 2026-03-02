package com.dark.gguf_lib.toolcalling

import org.json.JSONObject

/**
 * Parsed tool call from model output.
 */
data class ToolCall(
    val name: String,
    val arguments: JSONObject = JSONObject()
) {
    fun getString(key: String, default: String = ""): String =
        arguments.optString(key, default)

    fun getInt(key: String, default: Int = 0): Int =
        arguments.optInt(key, default)

    fun getDouble(key: String, default: Double = 0.0): Double =
        arguments.optDouble(key, default)

    fun getBoolean(key: String, default: Boolean = false): Boolean =
        arguments.optBoolean(key, default)

    fun has(key: String): Boolean = arguments.has(key)

    companion object {
        /**
         * Parse a tool call from JSON string: {"name":"...","arguments":{...}}
         */
        fun fromJson(json: String): ToolCall? = try {
            val j = JSONObject(json)
            val name = j.optString("name", "")
            if (name.isEmpty()) null
            else ToolCall(name, j.optJSONObject("arguments") ?: JSONObject())
        } catch (_: Exception) {
            null
        }
    }
}
