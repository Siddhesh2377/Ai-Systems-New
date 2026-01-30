# Multi-Turn Tool Calling System - Integration Guide

## Overview

The `ai_gguf` module now supports **multi-turn tool calling** on CPU-only Android devices (Snapdragon 7s Gen 3+). This replaces the previous single-turn, Qwen-only implementation with a model-agnostic system that supports any model with a chat template.

### Key capabilities
- **Multi-turn conversations**: Model calls a tool → gets result → calls another tool or responds with text
- **Model-agnostic**: Works with Qwen, Llama, Gemma, Phi, Mistral — any model with a chat template
- **Grammar modes**: STRICT (forces JSON) or LAZY (model chooses tool vs text)
- **Typed GBNF grammars**: Enforces exact parameter names, types, and enum values per tool
- **CPU-optimized**: ~80 token system prompt (was ~350), KV cache clear-and-re-prompt strategy

---

## Breaking Changes

### 1. `nativeIsToolCallingSupported()` now returns `true` for any model with a chat template
**Before**: Only returned `true` for Qwen models.
**After**: Returns `true` for any model that has a built-in chat template.
**Impact**: Code that relied on this returning `false` for non-Qwen models will now see `true`. This is intentional — grammar enforcement ensures valid JSON regardless of architecture.

### 2. `nativeEnableToolCalling(toolsJson)` no longer sets system prompt or chat template
**Before**: Hardcoded a 350-token system prompt and a Qwen-specific Jinja template.
**After**: Only sets `tools_json`, `tools_enabled`, and initializes grammar. System prompt and chat template must be set separately.
**Impact**: If you were relying on the hardcoded prompt/template, you must now call `nativeSetSystemPrompt()` before `nativeEnableToolCalling()`. The Kotlin `ToolCallManager.enable()` handles this automatically.

### 3. `ToolCallManager.enable()` now accepts an optional `ToolCallingConfig` parameter
**Before**: `fun enable(): Boolean`
**After**: `fun enable(config: ToolCallingConfig = ToolCallingConfig()): Boolean`
**Impact**: Fully backward compatible — calling `enable()` with no args uses defaults (STRICT grammar, typed=true).

---

## New Kotlin API

### New types (in `com.mp.ai_gguf.toolcalling`)

```kotlin
// Grammar enforcement mode
enum class GrammarMode(val value: Int) {
    STRICT(0),  // Forces JSON tool call output from first token
    LAZY(1)     // Model chooses: text response OR tool call (grammar activates on "{")
}

// Configuration for multi-turn tool calling
data class ToolCallingConfig(
    val maxRounds: Int = 5,              // Max tool call rounds before stopping
    val grammarMode: GrammarMode = GrammarMode.STRICT,
    val useTypedGrammar: Boolean = true, // Enforce exact param names/types/enums
    val maxTokensPerTurn: Int = 256
)

// Result from executing a tool
data class ToolResult(
    val toolName: String,
    val result: String,          // JSON string sent back to model as tool message
    val isError: Boolean = false
)

// Interface for handling tool execution
fun interface ToolExecutor {
    suspend fun execute(call: ToolCall): ToolResult
}

// Chat message for conversation history
data class ChatMessage(
    val role: String,    // "system", "user", "assistant", "tool"
    val content: String
)
```

### New methods on `ToolCallManager`

```kotlin
// Multi-turn tool calling orchestrator (suspend function)
suspend fun generateWithTools(
    userMessage: String,
    executor: ToolExecutor,
    config: ToolCallingConfig = ToolCallingConfig(),
    onToken: (String) -> Unit = {},
    onToolCallDetected: (ToolCall) -> Unit = {},
    onError: (String) -> Unit = {},
    onDone: (String) -> Unit = {}
)
```

### New methods on `GGUFNativeLib`

```kotlin
// Multi-turn generation (processes full conversation history)
external fun nativeGenerateStreamMultiTurn(
    messagesJson: String,  // JSON array of {role, content} objects
    maxTokens: Int,
    callback: StreamCallback
): Boolean

// Grammar mode: 0=STRICT, 1=LAZY
external fun nativeSetGrammarMode(mode: Int)

// Enable/disable parameter-aware typed grammar
external fun nativeSetTypedGrammar(enabled: Boolean)
```

---

## Usage Examples

### Basic multi-turn tool calling

```kotlin
val ggufLib = GGUFNativeLib()
val toolManager = ToolCallManager(ggufLib)

// Register tools
toolManager.registerTool(
    tool("get_weather", "Get current weather for a location") {
        stringParam("location", "City name", required = true)
        stringParam("units", "Temperature units",
            enum = listOf("celsius", "fahrenheit"))
    }
)

// Enable with LAZY grammar (model can choose text OR tool call)
toolManager.enable(ToolCallingConfig(
    grammarMode = GrammarMode.LAZY,
    maxRounds = 3,
    maxTokensPerTurn = 256
))

// Run multi-turn tool calling (in a coroutine scope)
toolManager.generateWithTools(
    userMessage = "What's the weather in London?",
    executor = { call ->
        when (call.name) {
            "get_weather" -> {
                val location = call.getString("location")
                // Your actual weather API call here
                ToolResult("get_weather",
                    """{"temperature": 15, "conditions": "cloudy", "location": "$location"}""")
            }
            else -> ToolResult(call.name, "Unknown tool", isError = true)
        }
    },
    onToken = { token ->
        // Stream tokens to UI (runs on IO thread, dispatch to main if needed)
        print(token)
    },
    onToolCallDetected = { toolCall ->
        // Optional: show "Calling get_weather..." in UI
        Log.d("ToolCall", "Calling ${toolCall.name} with ${toolCall.arguments}")
    },
    onError = { error ->
        Log.e("ToolCall", "Error: $error")
    },
    onDone = { finalResponse ->
        // The model's final text response after tool execution
        Log.d("ToolCall", "Response: $finalResponse")
    }
)
```

### Single-turn tool calling (backward compatible)

```kotlin
// Old code still works exactly the same
val toolManager = ToolCallManager(ggufLib)
toolManager.registerTools(/* ... */)
toolManager.enable()  // Uses default config

// Single-turn via existing nativeGenerateStream
ggufLib.nativeGenerateStream(prompt, 256, callback)
// callback.onToolCall() fires if model outputs a tool call
```

### Using STRICT vs LAZY grammar mode

```kotlin
// STRICT: Model is FORCED to output a tool call JSON.
// Use when you know the user's request requires a tool.
toolManager.enable(ToolCallingConfig(grammarMode = GrammarMode.STRICT))

// LAZY: Model can freely output text OR a tool call.
// Grammar only activates if the model starts outputting "{".
// Use for general chat where tools are optional.
toolManager.enable(ToolCallingConfig(grammarMode = GrammarMode.LAZY))
```

### Registering multiple tools for chaining

```kotlin
toolManager.registerTools(
    tool("search_web", "Search the web for information") {
        stringParam("query", "Search query", required = true)
    },
    tool("summarize", "Summarize text content") {
        stringParam("text", "Text to summarize", required = true)
        numberParam("max_words", "Maximum words in summary")
    }
)

// With maxRounds=5, the model can:
// 1. Call search_web("latest AI news")
// 2. Get search results
// 3. Call summarize(results, max_words=100)
// 4. Get summary
// 5. Respond with final text to user
toolManager.enable(ToolCallingConfig(maxRounds = 5))
```

---

## Architecture (for understanding, not for changes)

### Multi-turn flow
```
User message
    ↓
Kotlin: Build messages JSON [system, user]
    ↓
C++: nativeGenerateStreamMultiTurn()
  → Clear KV cache
  → Rebuild sampler (clone grammar into chain)
  → apply_template_multi() with full message array
  → Tokenize + decode prompt (prefill: 100-300 t/s)
  → Generate tokens (decode: 10-18 t/s)
  → Tool call detected? → callback.onToolCall()
    ↓
Kotlin: Execute tool, append [assistant, tool] messages
    ↓
C++: nativeGenerateStreamMultiTurn() (next turn)
  → Clear KV cache, re-encode full conversation
  → Generate response or next tool call
    ↓
Kotlin: Text response → callback.onDone()
```

### KV cache strategy
Each turn clears the KV cache and re-encodes the full conversation. This avoids complex position tracking bugs. Prefill runs at 100-300 t/s on CPU, so re-encoding 500-1000 tokens costs ~2-5s — acceptable for interactive tool flows.

### Grammar enforcement
- **Typed GBNF**: Per-tool rules enforce exact parameter keys, value types, and enum values
- **Generic GBNF fallback**: If typed grammar parsing fails, falls back to generic JSON grammar
- **Lazy mode**: Uses `llama_sampler_init_grammar_lazy_patterns()` with `"\\{"` trigger pattern

---

## Files modified in ai_gguf module

| File | What changed |
|------|-------------|
| `src/main/cpp/src/state/model_state.h` | Added `GrammarMode` enum, `SamplerParams` struct, `grammar_mode`/`use_typed_grammar` members, `rebuild_sampler_cached()`/`reset_grammar_sampler()` methods |
| `src/main/cpp/src/state/model_state.cpp` | Fixed grammar clone bug in `rebuild_sampler()`, added lazy grammar + typed grammar support in `update_grammar_if_needed()`, implemented `rebuild_sampler_cached()` and `reset_grammar_sampler()` |
| `src/main/cpp/src/chat/chat_template.h` | Added `ChatMessage`, `ToolParamInfo`, `ToolInfo` structs; declared `apply_template_multi()`, `extract_tool_info()`, `build_tool_grammar_typed()` |
| `src/main/cpp/src/chat/chat_template.cpp` | Implemented multi-turn template application, hand-rolled JSON parser for OpenAI tools format, parameter-aware GBNF generator |
| `src/main/cpp/src/tool_calling/tool_call_state.h` | Added `has_tool_calls_wrapper()` and `extract_arguments()` declarations |
| `src/main/cpp/src/tool_calling/tool_call_state.cpp` | Implemented `has_tool_calls_wrapper()` and `extract_arguments()` with brace-counting |
| `src/main/cpp/src/ai_gguf.cpp` | Moved mutex to file scope; added `nativeGenerateStreamMultiTurn` JNI, `nativeSetGrammarMode`, `nativeSetTypedGrammar`; removed Qwen-only gate from `nativeIsToolCallingSupported` and `nativeEnableToolCalling`; added JSON message parsing helpers |
| `src/main/java/.../GGUFNativeLib.kt` | Added 3 new native declarations: `nativeGenerateStreamMultiTurn`, `nativeSetGrammarMode`, `nativeSetTypedGrammar`; updated KDoc |
| `src/main/java/.../toolcalling/ToolCallManager.kt` | Added `GrammarMode`, `ToolCallingConfig`, `ToolResult`, `ToolExecutor`, `ChatMessage` types; updated `enable()` to accept config; added `generateWithTools()` orchestrator and `buildMessagesJson()`/`buildMinimalSystemPrompt()` helpers |
