#include "chat_template.h"
#include "llama.h"

#include <iomanip>
#include <sstream>
#include <algorithm>

namespace chat {

/* --------------------------------------------------------------------
 *  JSON escape helpers
 * -------------------------------------------------------------------- */
    std::string json_escape(const std::string &s) {
        std::ostringstream oss;
        for (auto c: s) {
            switch (c) {
                case '\\':
                    oss << "\\\\";
                    break;
                case '\"':
                    oss << "\\\"";
                    break;
                case '\n':
                    oss << "\\n";
                    break;
                case '\r':
                    oss << "\\r";
                    break;
                case '\t':
                    oss << "\\t";
                    break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<int>(c);
                    } else {
                        oss << c;
                    }
            }
        }
        return oss.str();
    }

/* --------------------------------------------------------------------
 *  Build tool preamble
 * -------------------------------------------------------------------- */
    std::string build_tool_preamble(const std::string &tools_json) {
        std::ostringstream preamble;
        preamble << "You may call tools by emitting ONLY the JSON object:\n"
                 << "{\"tool_calls\":[{\"name\":\"NAME\",\"arguments\":{...}}]}\n"
                 << "Available tools (OpenAI schema):\n" << tools_json << "\n";
        return preamble.str();
    }

/* --------------------------------------------------------------------
 *  𝙱𝙸𝙰𝙶 𝚂𝚗𝚎𝚝𝚎𝚍 𝚐𝚓𝚙-𝚕𝚒𝚗𝚐 𝙱𝚢𝚝𝚞𝚝𝚑
 * -------------------------------------------------------------------- */
    std::string build_tool_grammar(const std::string &tools_json) {
        auto names = extract_tool_names(tools_json);
        std::ostringstream g;

        g << "root         ::= json\n"
          << "json         ::= ws toolcall ws\n"
          << "toolcall     ::= \"{\" ws \"\\\"tool_calls\\\"\" ws \":\" ws \"[\" ws call ws \"]\" ws \"}\"\n"
          << "call         ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws toolname ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws object ws \"}\"\n";

        // Tool names
        g << "toolname     ::= ";
        if (!names.empty()) {
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) g << " | ";
                g << R"("\")" << names[i] << R"(\"")";
            }
        } else {
            g << R"("\"unknown\"")";
        }
        g << "\n";

        // Common rules (no leading blank line)
        g << R"(object       ::= "{" ws "}" | "{" ws member (ws "," ws member)* ws "}"
member       ::= string ws ":" ws value
array        ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
value        ::= string | number | object | array | "true" | "false" | "null"
string       ::= "\"" ([^"\\\n] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number       ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws           ::= [ \t\n\r]*
)";

        return g.str();
    }

/* --------------------------------------------------------------------
 *  Extract tool names from JSON
 * -------------------------------------------------------------------- */
    std::vector<std::string> extract_tool_names(const std::string &tools_json) {
        std::vector<std::string> out;
        size_t pos = 0;
        while (true) {
            size_t k = tools_json.find("\"name\"", pos);
            if (k == std::string::npos) break;
            size_t colon = tools_json.find(':', k);
            if (colon == std::string::npos) break;
            size_t q1 = tools_json.find('"', colon + 1);
            if (q1 == std::string::npos) break;
            size_t q2 = tools_json.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            out.emplace_back(tools_json.substr(q1 + 1, q2 - q1 - 1));
            pos = q2 + 1;
        }
        return out;
    }

/* --------------------------------------------------------------------
 *  Apply chat template (LLaMA’s built‑in style)
 * -------------------------------------------------------------------- */
    std::string apply_template(const ::llama_model *model, const std::string &system_prompt,
                               const std::string &user_message, const std::string &custom_template,
                               bool add_assistant) {

        /* Prefer the user‑supplied template, otherwise fall back to the model’s
           default chat template. */
        const char *tmpl = custom_template.empty() ? ::llama_model_chat_template(model, nullptr)
                                                   : custom_template.c_str();

        if (!tmpl || !*tmpl) {
            // Fallback – plain textual prompt
            std::string out;
            if (!system_prompt.empty())
                out += "System: " + system_prompt + "\n";
            out += "User: " + user_message + "\n";
            if (add_assistant) out += "Assistant: ";
            return out;
        }

        // Build the minimal list of messages
        std::vector<llama_chat_message> msgs;
        if (!system_prompt.empty())
            msgs.emplace_back(llama_chat_message{"system", system_prompt.c_str()});
        msgs.emplace_back(llama_chat_message{"user", user_message.c_str()});

        // Compute required buffer size
        int32_t need = ::llama_chat_apply_template(tmpl, msgs.data(),
                                                   static_cast<int32_t>(msgs.size()), add_assistant,
                                                   nullptr, 0);
        if (need < 0) need = -need;

        std::string out(static_cast<size_t>(need), '\0');
        int32_t written = ::llama_chat_apply_template(tmpl, msgs.data(),
                                                      static_cast<int32_t>(msgs.size()),
                                                      add_assistant, out.data(), need);
        if (written < 0) written = -written;
        out.resize(static_cast<size_t>(written));
        return out;
    }

/* --------------------------------------------------------------------
 *  Minimal JSON helpers for parsing known OpenAI tools schema.
 *  These operate on raw strings with no external JSON library.
 * -------------------------------------------------------------------- */
namespace {

    size_t skip_ws(const std::string& s, size_t pos) {
        while (pos < s.size() &&
               (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            ++pos;
        return pos;
    }

    // Extract a quoted string starting at pos (pos must be at the opening '"').
    // Advances pos past the closing '"'. Returns empty string on failure.
    std::string extract_quoted(const std::string& s, size_t& pos) {
        if (pos >= s.size() || s[pos] != '"') return "";
        ++pos; // skip opening quote
        std::string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                result += s[pos + 1];
                pos += 2;
            } else {
                result += s[pos++];
            }
        }
        if (pos < s.size()) ++pos; // skip closing quote
        return result;
    }

    // Find the matching '}' or ']' for an opening '{' or '[' at pos.
    // Skips nested braces/brackets and quoted strings.
    // Returns index of the closing character, or npos on failure.
    size_t find_matching_close(const std::string& s, size_t pos) {
        if (pos >= s.size()) return std::string::npos;
        char open_ch  = s[pos];
        char close_ch = (open_ch == '{') ? '}' : ']';
        int depth = 1;
        ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '"') {
                // Skip quoted string
                ++pos;
                while (pos < s.size() && s[pos] != '"') {
                    if (s[pos] == '\\') ++pos; // skip escaped char
                    ++pos;
                }
                if (pos < s.size()) ++pos; // skip closing quote
                continue;
            }
            if (s[pos] == open_ch)  ++depth;
            if (s[pos] == close_ch) --depth;
            ++pos;
        }
        return (depth == 0) ? pos - 1 : std::string::npos;
    }

    // Find value position for a JSON key within a string.
    // Searches for "key" followed by ':' and returns position of the value.
    // Skips occurrences where "key" appears as a value (not followed by ':').
    size_t find_key_value(const std::string& s, const std::string& key, size_t start) {
        std::string needle = "\"" + key + "\"";
        size_t pos = start;
        while (true) {
            pos = s.find(needle, pos);
            if (pos == std::string::npos) return std::string::npos;
            size_t after = pos + needle.size();
            after = skip_ws(s, after);
            if (after < s.size() && s[after] == ':') {
                ++after;
                return skip_ws(s, after);
            }
            pos += needle.size(); // not a key, try next occurrence
        }
    }

} // anonymous namespace

/* --------------------------------------------------------------------
 *  Multi-turn template application
 * -------------------------------------------------------------------- */
    std::string apply_template_multi(const ::llama_model *model,
                                     const std::vector<ChatMessage> &messages,
                                     const std::string &custom_template,
                                     bool add_assistant) {
        if (messages.empty()) return "";

        const char *tmpl = custom_template.empty()
                           ? ::llama_model_chat_template(model, nullptr)
                           : custom_template.c_str();

        if (!tmpl || !*tmpl) {
            // Fallback – plain textual prompt
            std::string out;
            for (const auto &msg: messages) {
                if (msg.role == "system")
                    out += "System: " + msg.content + "\n";
                else if (msg.role == "user")
                    out += "User: " + msg.content + "\n";
                else if (msg.role == "assistant")
                    out += "Assistant: " + msg.content + "\n";
                else if (msg.role == "tool")
                    out += "Tool: " + msg.content + "\n";
            }
            if (add_assistant) out += "Assistant: ";
            return out;
        }

        // Convert ChatMessage vector to llama_chat_message array.
        // llama_chat_message holds const char* pointers, so the ChatMessage
        // strings must outlive this array (they do -- messages is const ref).
        std::vector<llama_chat_message> msgs;
        msgs.reserve(messages.size());
        for (const auto &msg: messages) {
            msgs.push_back({msg.role.c_str(), msg.content.c_str()});
        }

        // First call: compute required buffer size
        int32_t need = ::llama_chat_apply_template(
                tmpl, msgs.data(),
                static_cast<int32_t>(msgs.size()),
                add_assistant,
                nullptr, 0);
        if (need < 0) need = -need;

        std::string out(static_cast<size_t>(need) + 1, '\0');
        int32_t written = ::llama_chat_apply_template(
                tmpl, msgs.data(),
                static_cast<int32_t>(msgs.size()),
                add_assistant,
                out.data(),
                static_cast<int32_t>(out.size()));
        if (written < 0) written = -written;
        out.resize(static_cast<size_t>(written));
        return out;
    }

/* --------------------------------------------------------------------
 *  Extract tool info from OpenAI-format tools JSON
 *
 *  Expected input format (produced by ToolDefinition.toOpenAIFormat):
 *  [{"type":"function","function":{"name":"...","description":"...",
 *    "parameters":{"type":"object","properties":{...},"required":[...]}}}]
 * -------------------------------------------------------------------- */
    std::vector<ToolInfo> extract_tool_info(const std::string &tools_json) {
        std::vector<ToolInfo> tools;
        size_t pos = 0;

        while (true) {
            // Find next "function" key (the key, not the value in "type":"function")
            size_t func_val = find_key_value(tools_json, "function", pos);
            if (func_val == std::string::npos) break;

            // The value should be a '{' (function object)
            if (func_val >= tools_json.size() || tools_json[func_val] != '{') {
                pos = func_val + 1;
                continue;
            }

            size_t func_end = find_matching_close(tools_json, func_val);
            if (func_end == std::string::npos) break;

            // Extract function object substring for scoped searching
            std::string func_obj = tools_json.substr(func_val, func_end - func_val + 1);

            ToolInfo info;

            // --- Extract name ---
            size_t name_val = find_key_value(func_obj, "name", 0);
            if (name_val != std::string::npos && name_val < func_obj.size()
                && func_obj[name_val] == '"') {
                info.name = extract_quoted(func_obj, name_val);
            }

            // --- Extract parameters ---
            size_t params_val = find_key_value(func_obj, "parameters", 0);
            if (params_val != std::string::npos && params_val < func_obj.size()
                && func_obj[params_val] == '{') {

                size_t params_end = find_matching_close(func_obj, params_val);
                if (params_end != std::string::npos) {
                    std::string params_obj = func_obj.substr(
                            params_val, params_end - params_val + 1);

                    // --- Extract properties ---
                    size_t props_val = find_key_value(params_obj, "properties", 0);
                    if (props_val != std::string::npos && props_val < params_obj.size()
                        && params_obj[props_val] == '{') {

                        size_t props_end = find_matching_close(params_obj, props_val);
                        if (props_end != std::string::npos) {
                            // Content between { and } of properties
                            std::string props_inner = params_obj.substr(
                                    props_val + 1, props_end - props_val - 1);

                            // Parse each property: "param_name": { "type": "...", ... }
                            size_t pp = 0;
                            while (pp < props_inner.size()) {
                                pp = skip_ws(props_inner, pp);
                                if (pp >= props_inner.size()) break;

                                // Expect opening '"' of param name
                                if (props_inner[pp] != '"') {
                                    ++pp;
                                    continue;
                                }

                                std::string param_name = extract_quoted(props_inner, pp);
                                if (param_name.empty()) break;

                                pp = skip_ws(props_inner, pp);
                                if (pp >= props_inner.size() || props_inner[pp] != ':') break;
                                ++pp;
                                pp = skip_ws(props_inner, pp);

                                // Param value should be a '{' object
                                if (pp >= props_inner.size() || props_inner[pp] != '{') break;
                                size_t prop_obj_end = find_matching_close(props_inner, pp);
                                if (prop_obj_end == std::string::npos) break;

                                std::string prop_obj = props_inner.substr(
                                        pp, prop_obj_end - pp + 1);

                                ToolParamInfo param;
                                param.name = param_name;

                                // Extract type
                                size_t type_val = find_key_value(prop_obj, "type", 0);
                                if (type_val != std::string::npos && type_val < prop_obj.size()
                                    && prop_obj[type_val] == '"') {
                                    param.type = extract_quoted(prop_obj, type_val);
                                }

                                // Extract enum values
                                size_t enum_val = find_key_value(prop_obj, "enum", 0);
                                if (enum_val != std::string::npos && enum_val < prop_obj.size()
                                    && prop_obj[enum_val] == '[') {
                                    size_t enum_end = find_matching_close(prop_obj, enum_val);
                                    if (enum_end != std::string::npos) {
                                        std::string enum_inner = prop_obj.substr(
                                                enum_val + 1, enum_end - enum_val - 1);
                                        size_t ep = 0;
                                        while (ep < enum_inner.size()) {
                                            ep = skip_ws(enum_inner, ep);
                                            if (ep >= enum_inner.size()) break;
                                            if (enum_inner[ep] == '"') {
                                                param.enum_values.push_back(
                                                        extract_quoted(enum_inner, ep));
                                            } else {
                                                ++ep; // skip commas, etc.
                                            }
                                        }
                                    }
                                }

                                info.params.push_back(param);
                                pp = prop_obj_end + 1;
                                pp = skip_ws(props_inner, pp);
                                if (pp < props_inner.size() && props_inner[pp] == ',') ++pp;
                            }
                        }
                    }

                    // --- Extract required array ---
                    size_t req_val = find_key_value(params_obj, "required", 0);
                    if (req_val != std::string::npos && req_val < params_obj.size()
                        && params_obj[req_val] == '[') {
                        size_t req_end = find_matching_close(params_obj, req_val);
                        if (req_end != std::string::npos) {
                            std::string req_inner = params_obj.substr(
                                    req_val + 1, req_end - req_val - 1);
                            size_t rp = 0;
                            while (rp < req_inner.size()) {
                                rp = skip_ws(req_inner, rp);
                                if (rp >= req_inner.size()) break;
                                if (req_inner[rp] == '"') {
                                    info.required.push_back(
                                            extract_quoted(req_inner, rp));
                                } else {
                                    ++rp;
                                }
                            }
                        }
                    }
                }
            }

            if (!info.name.empty()) {
                tools.push_back(std::move(info));
            }

            pos = func_end + 1;
        }

        return tools;
    }

/* --------------------------------------------------------------------
 *  Build parameter-aware GBNF grammar
 *
 *  For each tool, generates specific rules enforcing exact parameter
 *  names, types, and enum values. Parameters are ordered: required
 *  first (fixed), then optional (nested optional groups).
 *
 *  Example output for get_weather(location: string required, units: enum):
 *    call_get_weather ::= "{" ws ... ws args_get_weather ws "}"
 *    args_get_weather ::= "{" ws kv_get_weather_location
 *                          (ws "," ws kv_get_weather_units)? ws "}"
 *    kv_get_weather_location ::= "\"location\"" ws ":" ws string
 *    kv_get_weather_units    ::= "\"units\"" ws ":" ws ("\"celsius\"" | ...)
 * -------------------------------------------------------------------- */
    std::string build_tool_grammar_typed(const std::string &tools_json) {
        auto tools = extract_tool_info(tools_json);
        if (tools.empty()) return ""; // signal fallback to generic grammar

        std::ostringstream g;

        // Header rules
        g << "root         ::= json\n"
          << "json         ::= ws toolcall ws\n"
          << "toolcall     ::= \"{\" ws \"\\\"tool_calls\\\"\" ws \":\" ws \"[\" ws call ws \"]\" ws \"}\"\n";

        // call rule: union of all tool-specific call rules
        g << "call         ::= ";
        for (size_t i = 0; i < tools.size(); ++i) {
            if (i) g << " | ";
            g << "call_" << tools[i].name;
        }
        g << "\n";

        // Per-tool rules
        for (const auto &tool: tools) {
            // call_TOOLNAME
            g << "call_" << tool.name
              << " ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws \"\\\""
              << tool.name
              << "\\\"\" ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws args_"
              << tool.name << " ws \"}\"\n";

            // Split params into required and optional (preserving declaration order)
            std::vector<const ToolParamInfo *> req_params;
            std::vector<const ToolParamInfo *> opt_params;
            for (const auto &param: tool.params) {
                bool is_req = std::find(tool.required.begin(), tool.required.end(),
                                        param.name) != tool.required.end();
                if (is_req) req_params.push_back(&param);
                else opt_params.push_back(&param);
            }

            // args_TOOLNAME
            if (req_params.empty() && opt_params.empty()) {
                g << "args_" << tool.name << " ::= \"{\" ws \"}\"\n";
            } else {
                g << "args_" << tool.name << " ::= \"{\" ws ";

                // Required params (fixed order, comma-separated)
                for (size_t i = 0; i < req_params.size(); ++i) {
                    if (i > 0) g << " ws \",\" ws ";
                    g << "kv_" << tool.name << "_" << req_params[i]->name;
                }

                // Optional params (nested optional groups)
                if (!opt_params.empty()) {
                    if (req_params.empty()) {
                        // All optional: (kv_p0 (ws "," ws kv_p1 (...)?)? )?
                        g << "(";
                        g << "kv_" << tool.name << "_" << opt_params[0]->name;
                        for (size_t i = 1; i < opt_params.size(); ++i) {
                            g << " (ws \",\" ws kv_" << tool.name
                              << "_" << opt_params[i]->name;
                        }
                        // Close inner optionals
                        for (size_t i = 1; i < opt_params.size(); ++i) {
                            g << ")?";
                        }
                        g << ")?"; // close outer group
                    } else {
                        // Some required, then optional:
                        // (ws "," ws kv_opt0 (ws "," ws kv_opt1 ...)?)?
                        g << " (ws \",\" ws kv_" << tool.name
                          << "_" << opt_params[0]->name;
                        for (size_t i = 1; i < opt_params.size(); ++i) {
                            g << " (ws \",\" ws kv_" << tool.name
                              << "_" << opt_params[i]->name;
                        }
                        // Close all nested optionals
                        for (size_t i = 0; i < opt_params.size(); ++i) {
                            g << ")?";
                        }
                    }
                }

                g << " ws \"}\"\n";
            }

            // kv rules for each parameter
            for (const auto &param: tool.params) {
                g << "kv_" << tool.name << "_" << param.name
                  << " ::= \"\\\"" << param.name << "\\\"\" ws \":\" ws ";

                if (!param.enum_values.empty()) {
                    // Enum: one of the allowed quoted values
                    g << "(";
                    for (size_t i = 0; i < param.enum_values.size(); ++i) {
                        if (i) g << " | ";
                        g << "\"\\\"" << param.enum_values[i] << "\\\"\"";
                    }
                    g << ")";
                } else if (param.type == "string") {
                    g << "string";
                } else if (param.type == "number" || param.type == "integer") {
                    g << "number";
                } else if (param.type == "boolean") {
                    g << "(\"true\" | \"false\")";
                } else if (param.type == "object") {
                    g << "object";
                } else if (param.type == "array") {
                    g << "array";
                } else {
                    g << "value"; // fallback for unknown types
                }
                g << "\n";
            }
        }

        // Common rules (no leading blank line)
        g << R"(object       ::= "{" ws "}" | "{" ws member (ws "," ws member)* ws "}"
member       ::= string ws ":" ws value
array        ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
value        ::= string | number | object | array | "true" | "false" | "null"
string       ::= "\"" ([^"\\\n] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number       ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws           ::= [ \t\n\r]*
)";

        return g.str();
    }

/* --------------------------------------------------------------------
 *  Normalize double-nested tools JSON
 *
 *  Some callers produce:
 *    [{"type":"function","function":{"type":"function","function":{...}}}]
 *  This unwraps to:
 *    [{"type":"function","function":{...}}]
 * -------------------------------------------------------------------- */
    std::string normalize_tools_json(const std::string &tools_json) {
        // Quick check: is there a double-nesting pattern?
        if (tools_json.find("\"function\":{\"type\":\"function\"") == std::string::npos) {
            return tools_json; // No double nesting detected
        }

        std::string result;
        result.reserve(tools_json.size());

        size_t pos = 0;
        while (pos < tools_json.size()) {
            // Find next "function" JSON key (a key followed by ':')
            size_t func_val = find_key_value(tools_json, "function", pos);
            if (func_val == std::string::npos) {
                // No more "function" keys, copy remainder
                result.append(tools_json, pos, tools_json.size() - pos);
                break;
            }

            // Copy everything before this value (includes key + colon)
            result.append(tools_json, pos, func_val - pos);

            if (func_val >= tools_json.size() || tools_json[func_val] != '{') {
                // Value isn't an object – copy the char and continue
                result += tools_json[func_val];
                pos = func_val + 1;
                continue;
            }

            size_t outer_end = find_matching_close(tools_json, func_val);
            if (outer_end == std::string::npos) {
                // Malformed JSON – copy rest as-is
                result.append(tools_json, func_val, tools_json.size() - func_val);
                break;
            }

            // Check if this object itself contains another "function" key
            size_t inner_val = find_key_value(tools_json, "function", func_val + 1);
            if (inner_val != std::string::npos && inner_val < outer_end
                && inner_val < tools_json.size() && tools_json[inner_val] == '{') {
                // Double nested – use the inner function value
                size_t inner_end = find_matching_close(tools_json, inner_val);
                if (inner_end != std::string::npos && inner_end <= outer_end) {
                    result.append(tools_json, inner_val, inner_end - inner_val + 1);
                    pos = outer_end + 1;
                    continue;
                }
            }

            // Not double nested – copy outer value as-is
            result.append(tools_json, func_val, outer_end - func_val + 1);
            pos = outer_end + 1;
        }

        return result;
    }

} // namespace chat