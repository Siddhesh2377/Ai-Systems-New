#include "tool_call_state.h"

#include "llama.h"
#include "ggml-backend.h"
#include <sstream>
#include <algorithm>

#include <jni.h>
#include <string>
#include <mutex>

bool ToolCallState::accumulate(const std::string& chunk) {
    for (char c : chunk) {
        if (!collecting) {
            if (c == '{') {
                collecting = true;
                brace_depth = 1;
                buf.clear();
                buf.push_back(c);
                continue;
            }
        } else {
            buf.push_back(c);
            if (c == '{') ++brace_depth;
            else if (c == '}') {
                --brace_depth;
                if (brace_depth == 0) return true;     // complete JSON
            }
        }
    }
    return false;
}

bool ToolCallState::extract_tool_call(std::string& name, std::string& payload) const {
    bool has_wrapper = has_tool_calls_wrapper();

    // Also accept bare tool call: {"name":"...","arguments":{...}}
    bool has_name = buf.find("\"name\"") != std::string::npos;
    bool has_args = buf.find("\"arguments\"") != std::string::npos;

    if (!has_wrapper && !(has_name && has_args))
        return false;   // not a tool call

    // Extract the tool name
    size_t npos = buf.find("\"name\"");
    if (npos != std::string::npos) {
        size_t colon = buf.find(':', npos);
        if (colon != std::string::npos) {
            size_t q1 = buf.find('"', colon + 1);
            size_t q2 = buf.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                name = buf.substr(q1 + 1, q2 - q1 - 1);
        }
    }
    if (name.empty()) name = "tool";

    // Wrap bare format so the Kotlin parser always gets {"tool_calls":[...]}
    if (!has_wrapper && has_name && has_args) {
        payload = "{\"tool_calls\":[" + buf + "]}";
    } else {
        payload = buf;
    }
    return true;
}

bool ToolCallState::has_tool_calls_wrapper() const {
    return buf.find("\"tool_calls\"") != std::string::npos;
}

bool ToolCallState::extract_arguments(std::string& args_json) const {
    // Find "arguments" key
    size_t apos = buf.find("\"arguments\"");
    if (apos == std::string::npos) return false;

    // Find the colon after "arguments"
    size_t colon = buf.find(':', apos + 11);
    if (colon == std::string::npos) return false;

    // Find the opening '{' of the arguments object
    size_t obj_start = std::string::npos;
    for (size_t i = colon + 1; i < buf.size(); ++i) {
        char c = buf[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
        if (c == '{') { obj_start = i; break; }
        break; // unexpected character
    }
    if (obj_start == std::string::npos) return false;

    // Brace-count to find the matching '}'
    int depth = 1;
    size_t pos = obj_start + 1;
    while (pos < buf.size() && depth > 0) {
        char c = buf[pos];
        if (c == '"') {
            // Skip quoted string
            ++pos;
            while (pos < buf.size() && buf[pos] != '"') {
                if (buf[pos] == '\\') ++pos;
                ++pos;
            }
            if (pos < buf.size()) ++pos;
            continue;
        }
        if (c == '{') ++depth;
        else if (c == '}') --depth;
        ++pos;
    }

    if (depth != 0) return false;

    // pos-1 is the closing '}', extract the object
    args_json = buf.substr(obj_start, pos - obj_start);
    return true;
}