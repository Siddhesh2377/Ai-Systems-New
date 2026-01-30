#pragma once

#include <string>
#include <string_view>

/**
 * Optimized Tool Call State Manager
 *
 * Key optimizations for low-end devices:
 * 1. Pre-allocated buffer to avoid reallocations during streaming
 * 2. String view parsing to minimize copies
 * 3. Early detection of tool call patterns
 * 4. Efficient brace counting with validation
 */
class ToolCallState {
public:
    // Default buffer size - tool calls are typically small JSON objects
    static constexpr size_t DEFAULT_BUFFER_CAPACITY = 1024;

    ToolCallState() {
        buf.reserve(DEFAULT_BUFFER_CAPACITY);
    }

    // Called for every generated piece; returns true when
    // a complete JSON object has been accumulated.
    bool accumulate(const std::string& chunk);

    // Fast check if chunk might start a tool call (optimization)
    bool might_be_tool_call(const std::string& chunk) const;

    // Extract name and full payload from the accumulated JSON.
    // Returns false if parsing fails or too short.
    bool extract_tool_call(std::string& name, std::string& payload) const;

    // Extract tool call using string_view for zero-copy parsing
    bool extract_tool_call_sv(std::string_view& name, std::string& payload) const;

    // Whether we are currently collecting JSON.
    bool is_collecting() const { return collecting; }

    // Check if we've detected a likely tool call pattern
    bool is_likely_tool_call() const { return likely_tool_call; }

    // Get current buffer size (for debugging/metrics)
    size_t buffer_size() const { return buf.size(); }

    // Check if the accumulated JSON contains a "tool_calls" wrapper key
    bool has_tool_calls_wrapper() const;

    // Extract just the "arguments" JSON object from the accumulated buffer.
    // Returns false if no arguments found or parsing fails.
    bool extract_arguments(std::string& args_json) const;

    // Reset helpers - preserves capacity for next tool call
    void reset() {
        collecting = false;
        brace_depth = 0;
        likely_tool_call = false;
        buf.clear();
        // Note: clear() doesn't release capacity - this is intentional
    }

    // Full reset including buffer deallocation (use sparingly)
    void reset_full() {
        reset();
        buf.shrink_to_fit();
        buf.reserve(DEFAULT_BUFFER_CAPACITY);
    }

private:
    std::string buf;
    int brace_depth = 0;
    bool collecting = false;
    bool likely_tool_call = false;  // Early detection flag

    // Helper to check for tool_calls pattern
    bool check_tool_pattern() const;
};