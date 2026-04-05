# gguf_lib

Android AAR module providing Kotlin SDK + JNI bridge for on-device LLM/VLM inference.
Built on the [Tool-Neuron GGML Backend](https://github.com/Siddhesh2377/ToolNeuron) — a CPU-only,
ARM-optimized fork of llama.cpp.

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

## Usage

### Model Loading

```kotlin
val engine = GGMLEngine()

// From file path
engine.load(
    path = "/data/local/tmp/model.gguf",
    contextSize = 4096,
    threadMode = 1,          // 0=power_saving, 1=balanced, 2=performance
    flashAttn = false,
    cacheTypeK = "q8_0",
    cacheTypeV = "q8_0"
)

// From Android SAF content:// URI
engine.load(context, uri, contextSize = 4096, threadMode = 1)

// From file descriptor (AIDL service / SAF)
engine.loadFromFd(fd, contextSize = 4096, threadMode = 1)
```

### Thread Mode

Thread mode controls big.LITTLE core usage. Switch at runtime without reloading the model:

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

Recommended to expose as a 0–2 seekbar in the UI.

### Generation

```kotlin
// Streaming (single-turn)
engine.generateRawFlow("Hello!", maxTokens = 512).collect { event ->
    when (event) {
        is GenerationEvent.Token   -> print(event.text)
        is GenerationEvent.Done    -> println("\nDone")
        is GenerationEvent.Metrics -> println("${event.metrics.tokensPerSecond} t/s")
        is GenerationEvent.ToolCall -> handleTool(event.name, event.argsJson)
        is GenerationEvent.Error   -> println("Error: ${event.message}")
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

### Sampling

```kotlin
engine.setSampling(
    temperature = 0.7f,
    topK = 40,
    topP = 0.9f,
    minP = 0.05f,
    seed = -1
)

// JSON update (supports camelCase and snake_case)
engine.updateSamplerParams("""{"temperature":0.8,"top_p":0.95}""")

// Per-token logit bias
engine.setLogitBias("""{"1234": -100.0}""")
```

### Tool Calling

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

### Vision (VLM)

```kotlin
// Load text model first, then the vision projector
engine.load("/path/to/model.gguf")
engine.loadVlmProjector("/path/to/mmproj.gguf")

val imageBytes = File("/path/to/image.jpg").readBytes()
val marker = engine.getVlmDefaultMarker()  // e.g. "<__image__>"

val messages = """[{"role":"user","content":"Describe this: $marker"}]"""
engine.generateVlmFlow(messages, listOf(imageBytes), maxTokens = 512).collect { ... }
```

### RAG (Retrieval-Augmented Generation)

```kotlin
val rag = RAGEngine()
rag.create(dims = 256, topK = 32, topN = 5, lateChunking = true)
rag.loadModel("/path/to/embedding-model.gguf")

// Index documents
rag.addDocument("Full document text...", docId = "doc-1")

// Query
val results = rag.query("search query")
results.forEach { println("${it.docId} (${it.score}): ${it.text}") }

// Or build an augmented prompt directly
val prompt = rag.buildPrompt("user question", "Answer based on context:")
rag.close()
```

### Character / Personality

```kotlin
val character = CharacterEngine(engine)

// Set personality — maps to sampler params (temp, top_p, penalty, min_p)
character.setPersonality(Personality(
    name = "Luna",
    persona = "A warm, empathetic assistant",
    temperature = 0.8f,
    creativity = 0.7f
))

// Set mood — shifts temp and repetition penalty
character.setMood(Mood.HAPPY)

// Uncensored mode — vocab-level refusal token suppression (logit bias -100)
character.setUncensored(true)
```

### AIDL Service Optimization

When running inside an AIDL service, each `onToken` callback crosses the Binder boundary.
Increase the token batch size to reduce IPC call frequency:

```kotlin
// Default is 256 bytes. For AIDL, use 512+ to amortize Binder overhead (~20-50µs/call).
engine.setTokenBatchSize(512)

// The JNI layer uses a pre-allocated ByteArray (zero-copy) for delivery.
// Tokens are batched in native until threshold, then one Binder transaction per batch.
```

### KV Cache & State

```kotlin
// Context usage (0.0 = empty, 1.0 = full)
val usage = engine.getContextUsage()

// Save/restore KV state across sessions
engine.stateSaveToFile("/path/to/state.bin")
engine.stateLoadFromFile("/path/to/state.bin")

// Disk-backed prompt cache — system prompt KV reloaded from disk on cold start
engine.setPromptCacheDir(context.cacheDir.absolutePath)
```

### Control Vectors

```kotlin
engine.loadControlVectors("""[{"path":"/path/to/vector.gguf","scale":1.0}]""")
engine.clearControlVector()
```

## Device Sizing

```kotlin
// Pick thread mode and context size based on device RAM
val tier = GGMLEngine.detectDeviceTier(context)   // LOW_END / MID_RANGE / HIGH_END
val params = GGMLEngine.getRecommendedParams(context)
engine.load(path, params.contextSize, params.threadMode, cacheTypeK = params.cacheTypeK)
```

| Tier | RAM | threadMode | contextSize | KV Cache |
|------|-----|------------|-------------|----------|
| LOW_END | < 4 GB | 0 | 2048 | q4_0 |
| MID_RANGE | 4–8 GB | 1 | 4096 | q8_0 |
| HIGH_END | > 8 GB | 2 | 8192 | q8_0 |

## JNI Optimizations

| Optimization | Description |
|---|---|
| Method ID caching | Resolved once per callback class, reused across all calls |
| Pre-allocated ByteArray | Token bytes written via `SetByteArrayRegion`, no alloc per flush |
| Token batcher | Text accumulated in C++ until threshold, then one JNI/Binder call |
| Sampler reuse | Sampler rebuilt only when structural params change |
| Cached batches | Prompt batch + single-token batch allocated at load, reused per generate |
| Refusal token cache | Vocab scan runs once on first `setUncensored`, IDs cached for all subsequent calls |

## Build Integration

```cmake
set(LLAMA_DIR "/path/to/llama.cpp")
add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)

add_library(gguf_lib SHARED
    gguf_lib.cpp
    ${LLAMA_DIR}/engine/tool-manager.cpp
    ${LLAMA_DIR}/engine/thread-engine.cpp
    ${LLAMA_DIR}/engine/rag-engine.cpp
    # VLM sources ...
)

target_link_libraries(gguf_lib llama common android log)
```

Key CMake variables:

| Variable | Value | Purpose |
|---|---|---|
| `GGML_CPU_ARM_ARCH` | `armv8-a` | Baseline ARM (KleidiAI dispatches to i8mm/dotprod at runtime) |
| `GGML_CPU_KLEIDIAI` | ON | ARM KleidiAI micro-kernels for Q4/Q8 GEMM |
| `GGML_LTO` | ON | Link-time optimization |
| `BUILD_SHARED_LIBS` | OFF | Static link all into single .so |

## License

MIT — see root LICENSE.
