#include "tool_call_state.h"

#include "llama.h"
#include "ggml-backend.h"
#include <sstream>
#include <algorithm>

#include <jni.h>
#include <string>
#include <mutex>

bool ToolCallState::accumulate(const std::string& chunk) {
    buf.reserve(buf.size() + chunk.size());

    for (char c : chunk) {
        if (!collecting) {
            if (c == '{') {
                collecting = true;
                brace_depth = 1;
                buf.clear();
                buf.push_back(c);
            }
        } else {
            buf.push_back(c);
            if (c == '{') {
                ++brace_depth;
            } else if (c == '}') {
                if (--brace_depth == 0) return true;
            }
        }
    }
    return false;
}

bool ToolCallState::extract_tool_call(std::string& name, std::string& payload) const {
    static constexpr const char* tool_calls_key = "\"tool_calls\"";
    static constexpr const char* name_key = "\"name\"";

    if (buf.find(tool_calls_key) == std::string::npos)
        return false;

    size_t npos = buf.find(name_key);
    if (npos != std::string::npos) {
        size_t colon = buf.find(':', npos + 6);
        if (colon != std::string::npos) {
            size_t q1 = buf.find('"', colon + 1);
            size_t q2 = buf.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos) {
                name.assign(buf.data() + q1 + 1, q2 - q1 - 1);
            }
        }
    }
    if (name.empty()) name = "tool";

    payload = buf;
    return true;
}