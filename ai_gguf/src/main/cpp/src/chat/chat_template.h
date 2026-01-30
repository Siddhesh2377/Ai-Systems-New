/*=============================================================
 *   utils/chat_template.h
 *=============================================================
 *
 *  Routines that handle chat‑template formatting, JSON escaping,
 *  tool‑preamble generation, and GBNF grammar for tool calls.
 *
 *  The implementation is massively simplified: it either uses
 *  the default template from `llama_model_chat_template()` or a
 *  user‑supplied Jinja‑2 string.
 *  All string manipulation is pure C++; no external JSON parser.
 *
 *  Multi-turn support:
 *  - apply_template_multi() handles arbitrary role sequences
 *  - build_tool_grammar_typed() generates parameter-aware GBNF
 *  - extract_tool_info() parses OpenAI-format tools JSON
 *============================================================*/

#pragma once

#include <string>
#include <vector>

namespace chat {

    // ========================================================================
    // TYPES
    // ========================================================================

    /**
     * Chat message for multi-turn conversations.
     * Supports system, user, assistant, and tool roles.
     */
    struct ChatMessage {
        std::string role;    // "system", "user", "assistant", "tool"
        std::string content;
    };

    /**
     * Parsed tool parameter info for typed grammar generation
     */
    struct ToolParamInfo {
        std::string name;
        std::string type;   // "string", "number", "boolean", "object", "array"
        std::vector<std::string> enum_values;
    };

    /**
     * Parsed tool info for typed grammar generation
     */
    struct ToolInfo {
        std::string name;
        std::vector<ToolParamInfo> params;
        std::vector<std::string> required;
    };

    // ========================================================================
    // CORE FUNCTIONS
    // ========================================================================

    // Escape a string for inclusion in JSON literals
    std::string json_escape(const std::string& s);

    // Return the full prompt to be tokenised (single user message)
    std::string apply_template(const struct llama_model* model,
                               const std::string& system_prompt,
                               const std::string& user_message,
                               const std::string& custom_template = "",
                               bool add_assistant = true);

    /**
     * Apply chat template with arbitrary message array (multi-turn).
     *
     * Converts ChatMessage vector to llama_chat_message array and
     * calls llama_chat_apply_template() with the full history.
     * Supports system, user, assistant, and tool roles.
     */
    std::string apply_template_multi(const struct llama_model* model,
                                     const std::vector<ChatMessage>& messages,
                                     const std::string& custom_template = "",
                                     bool add_assistant = true);

    // Helper that prepends tool‑calling instructions to the system prompt
    std::string build_tool_preamble(const std::string& tools_json);

    // Extract a vector of tool names from a JSON‑style tool array
    std::vector<std::string> extract_tool_names(const std::string& tools_json);

    // Generates GBNF grammar string for the minimum tool‑call JSON pattern
    std::string build_tool_grammar(const std::string& tools_json);

    /**
     * Extract tool info from OpenAI-format tools JSON.
     *
     * Hand-rolled parser for the known schema (no external JSON lib).
     * Returns empty vector on parse failure.
     */
    std::vector<ToolInfo> extract_tool_info(const std::string& tools_json);

    /**
     * Build parameter-aware GBNF grammar from tools JSON.
     *
     * Generates per-tool rules enforcing exact parameter names,
     * types, and enum values. Returns empty string on failure
     * (caller should fall back to build_tool_grammar).
     */
    std::string build_tool_grammar_typed(const std::string& tools_json);

    /**
     * Normalize tools JSON by unwrapping double-nested function objects.
     *
     * Some callers produce:
     *   {"type":"function","function":{"type":"function","function":{...}}}
     * This unwraps to the correct OpenAI format:
     *   {"type":"function","function":{...}}
     *
     * Returns the input unchanged if no double nesting is detected.
     */
    std::string normalize_tools_json(const std::string& tools_json);
}