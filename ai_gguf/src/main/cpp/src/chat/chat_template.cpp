#include "chat_template.h"
#include "llama.h"

#include <iomanip>
#include <sstream>
#include <algorithm>

namespace chat {

    std::string json_escape(const std::string &s) {
        std::string out;
        out.reserve(s.size() + s.size() / 4);

        for (auto c: s) {
            switch (c) {
                case '\\': out += "\\\\"; break;
                case '\"': out += "\\\""; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<int>(c));
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    }

    std::string build_tool_preamble(const std::string &tools_json) {
        std::string preamble;
        preamble.reserve(256 + tools_json.size());
        preamble += "You may call tools by emitting ONLY the JSON object:\n"
                    "{\"tool_calls\":[{\"name\":\"NAME\",\"arguments\":{...}}]}\n"
                    "Available tools (OpenAI schema):\n";
        preamble += tools_json;
        preamble += "\n";
        return preamble;
    }

    std::string build_tool_grammar(const std::string &tools_json) {
        auto names = extract_tool_names(tools_json);
        std::ostringstream g;
        g.str().reserve(2048);

        g << R"(root         ::= json
json         ::= ws toolcall ws
toolcall     ::= "{" ws "\"tool_calls\"" ws ":" ws "[" ws call ws "]" ws "}"
call         ::= "{" ws "\"name\"" ws ":" ws toolname ws "," ws "\"arguments\"" ws ":" ws object ws "}"
)";

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

        g << R"(
object       ::= "{" ws "}" | "{" ws member (ws "," ws member)* ws "}"
member       ::= string ws ":" ws value
array        ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
value        ::= string | number | object | array | "true" | "false" | "null"
string       ::= "\"" ([^"\\\n] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
number       ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws           ::= [ \t\n\r]*
)";

        return g.str();
    }

    std::vector<std::string> extract_tool_names(const std::string &tools_json) {
        std::vector<std::string> out;
        out.reserve(8);

        size_t pos = 0;
        static constexpr const char* name_key = "\"name\"";
        static constexpr size_t name_key_len = 6;

        while ((pos = tools_json.find(name_key, pos)) != std::string::npos) {
            pos += name_key_len;

            size_t colon = tools_json.find(':', pos);
            if (colon == std::string::npos) break;

            size_t q1 = tools_json.find('"', colon + 1);
            if (q1 == std::string::npos) break;

            size_t q2 = tools_json.find('"', q1 + 1);
            if (q2 == std::string::npos) break;

            out.emplace_back(tools_json.data() + q1 + 1, q2 - q1 - 1);
            pos = q2 + 1;
        }

        return out;
    }

    std::string apply_template(const ::llama_model *model, const std::string &system_prompt,
                               const std::string &user_message, const std::string &custom_template,
                               bool add_assistant) {

        const char *tmpl = custom_template.empty() ? ::llama_model_chat_template(model, nullptr)
                                                   : custom_template.c_str();

        if (!tmpl || !*tmpl) {
            std::string out;
            out.reserve(system_prompt.size() + user_message.size() + 64);
            if (!system_prompt.empty()) {
                out += "System: ";
                out += system_prompt;
                out += "\n";
            }
            out += "User: ";
            out += user_message;
            out += "\n";
            if (add_assistant) out += "Assistant: ";
            return out;
        }

        std::vector<llama_chat_message> msgs;
        msgs.reserve(2);
        if (!system_prompt.empty())
            msgs.push_back({"system", system_prompt.c_str()});
        msgs.push_back({"user", user_message.c_str()});

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

}