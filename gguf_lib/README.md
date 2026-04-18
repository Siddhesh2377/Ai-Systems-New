# gguf_lib

Android AAR module providing a Kotlin SDK + JNI bridge for on-device LLM/VLM inference.
Built on the Tool-Neuron GGML Backend — a CPU-only, ARM-optimized fork of llama.cpp.

## Architecture

```
Kotlin SDK
  GGMLEngine          model load/unload, generation, KV cache, thread mode
  CharacterEngine     personality, mood, uncensored mode (sampler-level)
  ToolManager         tool registration, grammar mode, multi-format parsing
  RAGEngine           late-chunking retrieval, binary quantization index
  EmbeddingEngine     standalone text embedding
    |
GGUFNativeLib         JNI bridge (extern "C", method ID caching, zero-copy tokens)
    |
gguf_lib.so           JNI bridge + engine sources compiled into single .so
    |
llama.cpp engine/     GGMLEngine, ThreadEngine, ToolManager, RAGEngine, VLM
llama.cpp src/        model loading, tokenization, inference, sampling
ggml/                 CPU backend — NEON, i8mm, dotprod, KleidiAI ARM micro-kernels
```

---

## Model Loading

```kotlin
val engine = GGMLEngine()

// From file path
engine.load(
    path        = "/data/local/tmp/model.gguf",
    contextSize = 4096,
    threadMode  = 1,        // 0=power_saving, 1=balanced, 2=performance
    flashAttn   = false,
    cacheTypeK  = "q8_0",  // KV quantization (q4_0, q8_0, f16)
    cacheTypeV  = "q8_0"
)

// From Android SAF content:// URI
engine.load(context, uri, contextSize = 4096, threadMode = 1)

// From file descriptor (AIDL service / SAF)
engine.loadFromFd(fd, contextSize = 4096, threadMode = 1)
```

### KV Cache Quantization

KV cache is the biggest memory consumer for long contexts. Always quantize:

| Type | KV Memory | Quality |
|------|-----------|---------|
| `f16` | 100% | lossless |
| `q8_0` | ~50% | near-lossless — recommended default |
| `q4_0` | ~25% | slight quality loss — use on low-RAM devices |

The KV memory formula: `n_layers × n_ctx × 2 × n_kv_heads × d_head × dtype_bytes`.
A 7B model with 4096 ctx and q8_0 uses roughly **~500 MB** for KV vs ~1 GB for f16.

---

## Thread Mode

Controls big.LITTLE core usage. Switch at runtime without reloading:

```kotlin
engine.setThreadMode(0) // power saving — 1 thread, E-cores, n_batch=128
engine.setThreadMode(1) // balanced   — 2 P-cores gen, all P-cores batch (default)
engine.setThreadMode(2) // performance — 4 P-cores gen, all cores batch, n_batch=512
```

| Mode | Value | Gen Threads | Batch Threads | Pins to P-cores |
|------|-------|-------------|---------------|-----------------|
| Power Saving | 0 | 1 | E-cores only | No |
| Balanced | 1 | 2 P-cores | All P-cores | Yes |
| Performance | 2 | min(4, P) | All cores | Yes |

---

## Generation

```kotlin
// Streaming (single-turn)
engine.generateRawFlow("Hello!", maxTokens = 512).collect { event ->
    when (event) {
        is GenerationEvent.Token    -> print(event.text)
        is GenerationEvent.Done     -> println("\nDone")
        is GenerationEvent.Metrics  -> println("${event.metrics.tokensPerSecond} t/s")
        is GenerationEvent.ToolCall -> handleTool(event.name, event.argsJson)
        is GenerationEvent.Error    -> println("Error: ${event.message}")
        else -> {}
    }
}

// Streaming (multi-turn)
val messages = """[{"role":"user","content":"Hi"}]"""
engine.generateMultiTurnRawFlow(messages, maxTokens = 512).collect { ... }

// Non-streaming
val result = engine.generate("Hello!", maxTokens = 512)
println(result.text)
```

---

## Sampling

```kotlin
engine.setSampling(
    temperature = 0.7f,
    topK        = 40,
    topP        = 0.9f,
    minP        = 0.05f,
    seed        = -1
)

// JSON update — supports camelCase and snake_case keys
engine.updateSamplerParams("""{"temperature":0.8,"top_p":0.95}""")

// Per-token logit bias (token id → bias float)
engine.setLogitBias("""{"1234": -100.0}""")
```

---

## KV Cache Management

### Context Usage

```kotlin
val usage = engine.getContextUsage()  // 0.0 = empty, 1.0 = full
val info  = engine.getContextInfo()   // total, used, remaining, promptEstimate
```

### StreamingLLM Eviction Policy

For long conversations that exceed the context window, instead of hard stopping,
the StreamingLLM policy continuously evicts old tokens while keeping two regions:

- **Sink tokens** `[0, nSink)` — first few tokens contain critical attention anchors; never evict
- **Recency window** `[nPast-nWindow, nPast)` — the most recent tokens; always kept

Everything in between is dropped and tail positions are shifted to stay contiguous.

```kotlin
// Keep 4 sink tokens + last 512 tokens. Auto-evict when context fills.
engine.setKvPolicy(
    nSink       = 4,
    nWindow     = 512,
    evictAtFull = true
)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nSink` | Tokens at start to never evict (attention sinks) | 4 |
| `nWindow` | Recency tail kept after eviction | 0 (disabled) |
| `evictAtFull` | Auto-evict when context would overflow at generation start | false |

Set `nWindow = 0` to disable and fall back to the default half-discard context shift.

### Post-Prefill Budget (SnapKV-style)

After feeding a long system prompt, trim the KV cache immediately to the window budget:

```kotlin
engine.setSystemPrompt(longSystemPrompt)
engine.evictToBudget()  // apply eviction right now, free KV memory for generation
```

Useful when you know the system prompt is the longest thing you'll process.

### Session Save / Restore

Save the full KV cache state to disk and restore it later with the same model —
eliminates the prompt re-processing cost on cold start:

```kotlin
// Save after building up context
engine.stateSaveToFile(context.filesDir.absolutePath + "/session.bin")

// Restore on next launch — skips prompt re-decode
engine.stateLoadFromFile(context.filesDir.absolutePath + "/session.bin")
```

### Disk-Backed Prompt Cache

Automatically caches the system prompt KV state to disk. On subsequent loads with
the same system prompt, the KV cache is restored from disk instead of re-decoded:

```kotlin
engine.setPromptCacheDir(context.cacheDir.absolutePath)
// Now the first generate() after load will write the prompt KV to disk.
// All future loads with the same prompt skip re-evaluation entirely.
```

---

## Tool Calling

```kotlin
val weather = ToolDefinitionBuilder("get_weather", "Get current weather")
    .stringParam("location", "City name", required = true)
    .build()

val toolManager = ToolManager(engine)
toolManager.registerTools(
    listOf(weather),
    ToolCallingConfig(grammarMode = GrammarMode.LAZY)
)

engine.generateRawFlow(prompt, 512).collect { event ->
    if (event is GenerationEvent.ToolCall) {
        val result = callTool(event.name, event.argsJson)
        // Feed result back as a new turn
    }
}
```

Grammar modes:
- `NONE` — prompt-only, JSON extraction via regex
- `LAZY` — grammar activates only when a tool call is detected mid-stream
- `STRICT` — grammar-constrained from the first token

---

## Vision (VLM)

```kotlin
// Load text model first, then the vision projector
engine.load("/path/to/model.gguf")
engine.loadVlmProjector("/path/to/mmproj.gguf")

val imageBytes = File("/path/to/image.jpg").readBytes()
val marker     = engine.getVlmDefaultMarker()   // e.g. "<__image__>"

val messages = """[{"role":"user","content":"Describe this: $marker"}]"""
engine.generateVlmFlow(messages, listOf(imageBytes), maxTokens = 512).collect { ... }
```

---

## RAG (Retrieval-Augmented Generation)

```kotlin
val rag = RAGEngine()
rag.create(dims = 256, topK = 32, topN = 5, lateChunking = true)
rag.loadModel("/path/to/embedding-model.gguf")

// Index
rag.addDocument("Full document text...", docId = "doc-1")

// Query
val results = rag.query("search query")
results.forEach { println("${it.docId} (${it.score}): ${it.text}") }

// Or build an augmented prompt directly
val prompt = rag.buildPrompt("user question", "Answer based on context:")
rag.close()
```

---

## Character / Personality

All personality/mood/uncensored state lives purely in sampler params — no separate model, no extra memory.

```kotlin
val character = CharacterEngine(engine)

// Personality — maps creativity/temperature/topP to sampler params
character.setPersonality(Personality(
    name        = "Luna",
    persona     = "A warm, empathetic assistant",
    temperature = 0.8f,
    creativity  = 0.7f
))

// Mood — shifts temperature and repetition penalty via lookup table
character.setMood(Mood.HAPPY)

// Uncensored mode — vocab-scan for refusal tokens on first call,
// then applies logit bias -100 to suppress them. Cached after first scan.
character.setUncensored(true)
```

---

## AIDL Service Optimization

When running inside an AIDL service, each `onToken` callback crosses the Binder boundary (~20–50 µs per call). Increase the token batch size:

```kotlin
// Default: 256 bytes. Tune per use case:
engine.setTokenBatchSize(64)   // direct in-process JNI — low latency
engine.setTokenBatchSize(256)  // default
engine.setTokenBatchSize(512)  // AIDL service — amortize Binder overhead
```

Tokens accumulate in native memory until the threshold is reached, then one Binder
transaction delivers the batch. The delivery buffer is pre-allocated and reused
(zero-copy `SetByteArrayRegion`).

---

## Device Sizing

```kotlin
val tier   = GGMLEngine.detectDeviceTier(context)   // LOW_END / MID_RANGE / HIGH_END
val params = GGMLEngine.getRecommendedParams(context)
engine.load(path, params.contextSize, params.threadMode, cacheTypeK = params.cacheTypeK)
```

| Tier | RAM | threadMode | contextSize | KV Cache |
|------|-----|------------|-------------|----------|
| LOW_END | < 4 GB | 0 | 2048 | q4_0 |
| MID_RANGE | 4–8 GB | 1 | 4096 | q8_0 |
| HIGH_END | > 8 GB | 2 | 8192 | q8_0 |

---

## JNI Optimizations

| Optimization | Description |
|---|---|
| Method ID caching | Callback method IDs resolved once per class, stored globally |
| Pre-allocated ByteArray | Token bytes written via `SetByteArrayRegion` — no alloc per flush |
| Token batcher | Text accumulated in C++ to threshold, then one JNI/Binder call |
| Prompt batch reuse | Prompt decode batch allocated once at load, reused per generate |
| Single-token batch | Generation loop batch allocated once, reused per token step |
| Sampler reuse | Sampler rebuilt only on structural param changes |
| Refusal token cache | Vocab scan runs once on first `setUncensored`, IDs reused forever |
| KV eviction | Native `llama_memory_seq_rm` + `seq_add` — no KV re-copy, just pointer removal |

---

## Build Integration

```cmake
set(LLAMA_DIR "/path/to/llama.cpp")
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)

add_library(gguf_lib SHARED
    gguf_lib.cpp
    ${LLAMA_DIR}/engine/tool-manager.cpp
    ${LLAMA_DIR}/engine/thread-engine.cpp
    ${LLAMA_DIR}/engine/rag-engine.cpp
)

target_link_libraries(gguf_lib llama common android log)
```

Key CMake variables:

| Variable | Value | Purpose |
|---|---|---|
| `GGML_CPU_ARM_ARCH` | `armv8-a` | Baseline ARM — KleidiAI dispatches to i8mm/dotprod at runtime |
| `GGML_CPU_KLEIDIAI` | ON | ARM KleidiAI micro-kernels for Q4/Q8 GEMM |
| `GGML_LTO` | ON | Link-time optimization |
| `BUILD_SHARED_LIBS` | OFF | Static link all into single .so |

---

## License

MIT — see root LICENSE.
