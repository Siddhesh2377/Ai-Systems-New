# gguf_lib

Android AAR module providing a Kotlin SDK + JNI bridge for on-device LLM/VLM
inference. Built on llama.cpp + the tool-neuron engine helpers (CPU-only,
ARM-optimized).

**Companion docs**
- [`CLAUDE.md`](CLAUDE.md) — native code map, invariants, what's intentionally removed
- [`VLM.md`](VLM.md) — vision-language integration guide (VT cache + VLM-KV cache + opOffload)
- [`DEVICE.md`](DEVICE.md) — Snapdragon 7s Gen 3 benchmark data; the basis for every perf claim below

> **Removed in May 2026:** tool calling, control vectors, personality / mood / uncensored mode. AAR will not link if host code references these symbols. See [`CLAUDE.md`](CLAUDE.md) for the full list.

## Architecture

```
Kotlin SDK
  GGMLEngine          model load/unload, generation, KV cache, thread mode, VLM
  RAGEngine           late-chunking retrieval, binary quantization index
  EmbeddingEngine     standalone text embedding
  TextDigest          extractive summarization (CPU-only, no model)
    |                 (GGUFNativeLib — internal JNI bridge)
gguf_lib.so           JNI + engine sources compiled into a single .so
    |
llama.cpp engine/     thread-engine, rag-engine, mtmd (VLM)
llama.cpp src/        model loading, tokenization, inference, sampling
ggml/                 CPU backend — NEON, i8mm, dotprod, KleidiAI ARM kernels
```

## Loading

```kotlin
val engine = GGMLEngine()
engine.load(
    path        = "/data/local/tmp/model.gguf",
    contextSize = 4096,
    threads     = 0,           // 0 = auto from current thread mode
    batchSize   = 0,           // 0 = auto
    flashAttn   = false,
    useMmap     = true,
    useMlock    = false,
    cacheTypeK  = "q8_0",
    cacheTypeV  = "q8_0",
)

// SAF / file descriptor variants
engine.loadFromFd(fd)
engine.load(context, uri)
```

### KV cache quantization

| Type   | KV memory | Quality          |
|--------|-----------|------------------|
| `f16`  | 100%      | lossless         |
| `q8_0` | ~50%      | near-lossless (default) |
| `q4_0` | ~25%      | slight quality loss; use on low-RAM devices |

Memory: `n_layers x n_ctx x 2 x n_kv_heads x d_head x dtype_bytes`. A 7B model
with 4096 ctx and `q8_0` uses ~500 MB for KV vs ~1 GB for `f16`.

### Thread mode

```kotlin
engine.setThreadMode(0)  // power saving
engine.setThreadMode(1)  // balanced (default)
engine.setThreadMode(2)  // performance
```

| Mode        | Gen threads | Batch threads | Pins to P-cores |
|-------------|-------------|---------------|-----------------|
| Power saving | 1          | E-cores only  | no              |
| Balanced    | 2 P-cores   | All P-cores   | yes             |
| Performance | min(4, P)   | All cores     | yes             |

Note: the VLM projector binds `n_threads` at init. After `setThreadMode()` you
must `releaseVlmProjector()` then `loadVlmProjector()` to re-bind.

### Thermal-adaptive auto mode

```kotlin
engine.setAutoMode(true)
engine.setThermalThresholds(warmMilliC = 65_000, hotMilliC = 75_000, critMilliC = 85_000)
// Call periodically (e.g. once per second) from a background coroutine:
engine.autoModeTick()
val thermalJson = engine.getThermalState()  // {"zone_max_milli_c": ..., "mode": ...}
```

Reads `/sys/class/thermal/thermal_zoneN` and de-rates the thread mode when the SoC
gets hot. Useful for sustained generation on power-limited devices.

### Thinking mode (Qwen3 etc.)

```kotlin
if (engine.supportsThinking()) {
    engine.setThinkingEnabled(true)   // reasoning blocks emitted as part of token stream
}
```

## Generation

```kotlin
// Single-turn streaming
engine.generateFlow("Hello!", maxTokens = 512).collect { event ->
    when (event) {
        is GenerationEvent.Token    -> print(event.text)
        is GenerationEvent.Done     -> {}
        is GenerationEvent.Metrics  -> log(event.metrics.tokensPerSecond)
        is GenerationEvent.Error    -> log(event.message)
        is GenerationEvent.Progress -> updateProgress(event.progress)
        else -> {}
    }
}

// Multi-turn streaming
val messages = """[{"role":"user","content":"Hi"}]"""
engine.generateMultiTurnFlow(messages, maxTokens = 512).collect { /* ... */ }

// Non-streaming
val result = engine.generate("Hello!", maxTokens = 512)
```

Cancellation: closing the collecting coroutine calls `nativeStopGeneration()`
and the in-flight generate returns immediately.

## Sampling

```kotlin
engine.setSampling(temperature = 0.7f, topK = 40, topP = 0.9f, minP = 0.05f)
engine.updateSamplerParams("""{"temperature":0.8,"top_p":0.95}""")
engine.setLogitBias("""{"1234": -100.0}""")
```

`updateSamplerParams` accepts both camelCase and snake_case; recognized keys:
`temperature`, `topK`/`top_k`, `topP`/`top_p`, `minP`/`min_p`,
`repeatPenalty`, `frequencyPenalty`, `presencePenalty`, `penaltyLastN`,
`dryMultiplier`, `dryBase`, `dryAllowedLength`, `dryPenaltyLastN`,
`xtcProbability`, `xtcThreshold`, `mirostat`, `mirostatTau`, `mirostatEta`,
`seed`.

## KV cache management

```kotlin
val usage = engine.getContextUsage()  // 0.0..1.0

// StreamingLLM eviction: keep [0, nSink) + tail of nWindow tokens, drop middle.
engine.setKvPolicy(nSink = 4, nWindow = 512, evictAtFull = true)
engine.evictToBudget()  // SnapKV-style post-prefill trim

// Session save/restore
engine.stateSaveToFile("$filesDir/session.bin")
engine.stateLoadFromFile("$filesDir/session.bin")

// Disk-backed prompt cache: system prompt KV is auto-saved on first eval and
// restored on subsequent loads with the same prompt.
engine.setPromptCacheDir(context.cacheDir.absolutePath)
```

## Vision (VLM)

```kotlin
engine.load("/path/to/model.gguf")
engine.loadVlmProjector(
    path           = "/path/to/mmproj.gguf",
    threads        = 0,
    imageMinTokens = -1,
    imageMaxTokens = 128,
)

val marker   = engine.getVlmDefaultMarker()
val messages = """[{"role":"user","content":"Describe: $marker"}]"""
engine.generateVlmFlow(messages, listOf(imageBytes), maxTokens = 256).collect { /* ... */ }

engine.releaseVlmProjector()
```

`imageMaxTokens` caps the *overview* image budget. For LFM2-VL the per-tile
grid is a compile-time constant in `clip.cpp` and is unaffected by this knob.

`GenerationEvent.VlmStageMetrics` reports `vlmEncodeMs` (ViT forward), `vlmDecodeMs`
(LLM running prompt-eval on image embeddings) and `imageTokens` once per call.

### Disk-backed caches (the TTFT win)

Two persistent caches compose. **VT cache** stores ViT output embeddings; **VLM-KV
cache** stores the LLM context state at the post-image boundary. On a VLM-KV hit,
both the ~10 s ViT pass and the ~9 s image-prefill are skipped — TTFT drops from
~18.7 s to hundreds of ms.

See [`VLM.md`](VLM.md) for the full integration pattern, key derivation, budget
sizing, opOffload tradeoffs, and pre-warming via `precomputeVisionEmbeddings`.

```kotlin
engine.vtCacheInit(dir, budgetBytes = 200L * 1024 * 1024)
engine.vlmKvCacheInit(dir, budgetBytes = 300L * 1024 * 1024)
```

## RAG

```kotlin
val rag = RAGEngine()
rag.create(dims = 256, topK = 32, topN = 5, lateChunking = true)
rag.loadModel("/path/to/embedding-model.gguf")
rag.addDocument("Full document text...", docId = "doc-1")

val results = rag.query("search query")
val prompt  = rag.buildPrompt("user question", "Answer based on context:")

// Persist & restore
val blob = rag.exportIndex()
rag.importIndex(blob!!)

rag.close()
```

## Embedding (standalone)

```kotlin
EmbeddingEngine().use { embedder ->
    embedder.load("/path/to/embedding.gguf")
    val v = embedder.embed("hello world")
}
```

Independent of `GGMLEngine` — both can run concurrently.

## Extractive summarization (no model required)

`TextDigest` is a CPU-only TextRank + MMR + lead-bias summarizer. No GGUF model
needed; uses sentence-level features to rank.

```kotlin
val digest = TextDigest.compress(
    text         = longText,
    query        = "what we shipped this week",
    targetTokens = 256,
)
```

## Hardware introspection

```kotlin
val gpus = HardwareEngine.probe()        // List<GpuProfile>
val firstGpu = HardwareEngine.firstGpu()
val backendsJson = engine.listBackendsJson()  // diagnostic only
```

## AIDL service tuning

When running inside an AIDL service, each token callback crosses Binder
(~20-50 us per call). Increase the token batch threshold:

```kotlin
engine.setTokenBatchSize(64)   // direct in-process JNI
engine.setTokenBatchSize(256)  // default
engine.setTokenBatchSize(512)  // AIDL service — amortize Binder overhead
```

Tokens accumulate in native memory until the threshold is reached, then a
single Binder transaction delivers the batch via a pre-allocated, reused
`byte[]` (zero-copy `SetByteArrayRegion`).

## Device sizing

```kotlin
val tier   = GGMLEngine.detectDeviceTier(context)        // LOW_END / MID_RANGE / HIGH_END
val params = GGMLEngine.getRecommendedParams(context)
engine.load(path, params.contextSize, cacheTypeK = params.cacheTypeK, cacheTypeV = params.cacheTypeV)
```

| Tier      | RAM   | contextSize | KV cache |
|-----------|-------|-------------|----------|
| LOW_END   | <4 GB | 2048        | q4_0     |
| MID_RANGE | 4-8 GB| 4096        | q8_0     |
| HIGH_END  | >8 GB | 8192        | q8_0     |

## Build integration

1. Add this module as a Gradle subproject (also include `:tn_security` — it's the
   transitive log/error sink).
2. Update `LLAMA_DIR` in `src/main/cpp/CMakeLists.txt` to point at your
   llama.cpp checkout.
3. The native library loads via `System.loadLibrary("gguf_lib")` automatically
   on first access to `GGUFNativeLib` (called from `GGMLEngine`).
4. All public APIs live in `com.dark.gguf_lib.*`. `GGUFNativeLib` is `internal`
   — go through `GGMLEngine`, `EmbeddingEngine`, `RAGEngine`, `TextDigest`,
   `VlmEncoder`, or `HardwareEngine`.

## Error reporting

Every native log line and every structured error emitted by `gguf_lib` (and the
underlying llama.cpp / ggml libraries) is delivered through the unified
`:tn_security` event stream, tagged `TN_MODULE_GGUF_LIB` / `TN_MODULE_LLAMA_CPP`
/ `TN_MODULE_GGML`. Host apps read errors off `TnSecurity` rather than catching
them per-call. See [`../tn_security/README.md`](../tn_security/README.md).
