# ai_rmg

Android AAR module providing a Kotlin SDK + JNI bridge for [rm-graph](https://github.com/) — a small, purpose-built C++ inference engine for `.rmg` model files. CPU-only, mmap'd, runtime-dispatched NEON / AVX2 kernels, single-file model format.

The SDK exposes the engine's three surfaces — one-shot greedy decode, native per-token streaming, and token-at-a-time forward — to Kotlin via suspend functions and a `Flow<GenerationEvent>`. Decoding from token ids to UTF-8 text is handled natively when the `.rmg` file embeds a tokenizer blob.

## Architecture

```
Kotlin SDK
  RmgEngine          load/unload, suspend generate, Flow streaming, forward/reset, decode
    |
RmgNativeLib         JNI bridge (internal)
    |
libai_rmg.so         JNI wrapper + rm-graph engine sources
    |
rm-graph engine      mmap'd .rmg loader, fp16/int4 kernels, KV cache, RoPE, RMSNorm,
                     embedded tokenizer (id ↔ raw bytes, zero-copy view)
```

## Public API surface

```
com.dark.ai_rmg
  RmgEngine                            // main entry point, Closeable
com.dark.ai_rmg.models
  RmgDims                              // model dimensions (d_model, n_layers, ...)
  RmgLogLevel                          // Debug / Info / Warn / Error
  GenerationEvent                      // sealed: Token / Progress / Metrics / Error / Done
  DecodingMetrics                      // tokens/sec, time-to-first-token, totals
  GenerationResult                     // sync generate() return — tokens + decoded text
```

## Loading

```kotlin
val engine = RmgEngine()
engine.load("/data/local/tmp/model.rmg")

val d: RmgDims = engine.dims!!   // populated after load()
println("vocab=${d.vocabSize} ctx=${d.maxSeq} layers=${d.nLayers}")
println("tokenizer embedded: ${engine.hasTokenizer}")
```

`load()` is synchronous — call it from a background dispatcher to avoid blocking. `isLoaded`, `dims`, `hasTokenizer` reflect engine state. Re-loading requires `unload()` first.

The engine reads the optional tokenizer blob from the `.rmg` mmap during `load`. When present, single-call text decode and per-token byte views are available.

## Generation — non-streaming

```kotlin
val result: GenerationResult = engine.generate(
    promptIds = intArrayOf(1, 2, 3, 4),
    maxNew    = 64,
    stopId    = 2          // -1 to disable early stop
)

if (result.success) {
    println(result.text ?: "[no tokenizer; got ${result.tokenIds.size} ids]")
    println("${result.metrics?.tokensPerSecond} t/s")
} else {
    println("error: ${result.error}")
}
```

`generate()` is a `suspend fun` — runs on `Dispatchers.IO`. It dispatches a single native call into `engine_generate` (greedy, no per-token JNI overhead) and, when the model has an embedded tokenizer, follows up with a single `engine_decode_tokens` call to populate `result.text`.

## Generation — streaming Flow

```kotlin
val sb = StringBuilder()
engine.generateFlow(promptIds, maxNew = 128, stopId = 2).collect { event ->
    when (event) {
        is GenerationEvent.Token   -> sb.append(String(event.bytes, Charsets.UTF_8))
        is GenerationEvent.Metrics -> log("${event.metrics.tokensPerSecond} t/s")
        is GenerationEvent.Error   -> showError(event.message)
        GenerationEvent.Done       -> render(sb.toString())
        else                       -> {}
    }
}
```

`generateFlow()` is driven by the **native** `engine_generate_stream` callback — one C-to-Kotlin trip per token, no Kotlin-side argmax loop. Each `Token` event carries the raw byte form of the token from the embedded tokenizer (empty `ByteArray` when the model has no tokenizer).

Cancelling the collector aborts the in-flight generation cleanly: the next callback invocation returns non-zero, the native code unwinds, and `Done` is emitted.

Event ordering on a successful run:

```
Token(...)      // one per generated token
...
Metrics(...)    // tokens/sec + totals
Done
```

### UTF-8 across token boundaries

BPE tokens often split multi-byte UTF-8 sequences (emojis, CJK). Two safe patterns:

```kotlin
// 1. Accumulate raw bytes, decode at end:
val buf = java.io.ByteArrayOutputStream()
engine.generateFlow(...).collect { e ->
    if (e is GenerationEvent.Token) buf.write(e.bytes)
    if (e is GenerationEvent.Done) println(buf.toString(Charsets.UTF_8))
}

// 2. Collect ids during stream, decode in one native call at end:
val ids = mutableListOf<Int>()
engine.generateFlow(...).collect { e ->
    if (e is GenerationEvent.Token) ids += e.tokenId
}
println(engine.decode(ids.toIntArray()))
```

For incremental UI rendering, decode-on-each-token mostly works — replacement chars only appear briefly at the end of partial multi-byte sequences.

## Token-at-a-time (manual loop)

For custom samplers (top-k, top-p, temperature, etc.) skip the Flow and drive the engine yourself:

```kotlin
val logits = FloatArray(engine.dims!!.vocabSize)
engine.reset()

for (id in promptIds) engine.forward(id, logits)

while (!shouldStop()) {
    val next = mySampler(logits)
    if (next == eosId) break
    onToken(next)
    engine.forward(next, logits)
}
```

`forward()` advances `seqPos` by 1 and writes `vocabSize` logits into the buffer you provide. Pre-allocate once.

## Tokenizer access

```kotlin
if (engine.hasTokenizer) {
    val bytes: ByteArray? = engine.tokenBytes(tokenId = 42)   // null if id out of range
    val text:  String?    = engine.decode(intArrayOf(1, 2, 3))
}
```

Both routes call directly into the engine's mmap-resident tokenizer table — no allocation per byte view, single allocation for the joined text.

## KV cache lifecycle

```kotlin
engine.reset()          // clears KV cache, seqPos -> 0
println(engine.seqPos)  // current position in the KV cache
```

`generate()` and `generateFlow()` reset the KV cache internally before prefill. Direct `forward()` calls do not — call `reset()` between independent prompts.

## Logging

The native engine emits short structured log lines (`[rmg INFO] model: ...`, dispatch decisions, errors). They are routed to Android logcat under tag `rmg`.

```kotlin
RmgEngine.setLogLevel(RmgLogLevel.Warn)   // suppress info-level chatter
```

## Lifecycle

`RmgEngine` is `Closeable`. Always close it when done — the native engine holds an mmap and pre-allocated buffers (KV cache, RoPE tables, scratch).

```kotlin
engine.use { e ->
    e.load(path)
    val r = e.generate(promptIds, maxNew = 64)
    ...
}   // close() called automatically
```

`close()` is idempotent and `@Synchronized`. **Do not call it concurrently with `forward` / `generate` / `generateFlow` / `reset`** — the engine itself is single-threaded per the upstream contract.

## Threading model

One `RmgEngine` instance == one rm-graph engine == single-threaded. Multiple instances on the same `.rmg` file are allowed (each holds its own KV cache and scratch). Do not share an instance across coroutines without external synchronization.

## Build requirements

- Android NDK `27.3.13750724`
- CMake `3.31.4`
- minSdk `29`, compileSdk `36`
- ABI: `arm64-v8a` only (rm-graph kernels need `armv8.2-a + fp16 + dotprod`; NDK clang rejects `__builtin_cpu_supports("f16c")` so x86_64 is unsupported on Android)

The CMake build references the rm-graph source tree via `RMG_ROOT` (default `/home/home/CLionProjects/rm-graph`). Override with `-DRMG_ROOT=<path>` in `defaultConfig.externalNativeBuild.cmake.arguments` for a different layout.

## Native compile flags

| Flag | Purpose |
|---|---|
| `-O3 -DNDEBUG` | release optimization |
| `-march=armv8.2-a+fp16+fp16fml+dotprod+...` | required by rm-graph kernels.cc NEON intrinsics |
| `-ffp-contract=fast` | fuse a*b+c into FMA |
| `-fno-math-errno -fno-signed-zeros -fno-trapping-math` | safe FP perf subset (NOT `-ffast-math`) |
| `-fvisibility=hidden -fvisibility-inlines-hidden` | shrink .so, eliminate PLT |
| `-ffunction-sections -fdata-sections` | enable `--gc-sections` dead-code stripping |
| `-Wl,--gc-sections -Wl,--icf=safe` | strip unreferenced code, merge identical sections |
| `-Wl,-z,max-page-size=16384` | Android 15+ 16 KB page support |

## What this SDK does not do

- **Tokenization (encode).** The Kotlin API speaks `IntArray` for input. Decoding ids→text is supported via the embedded tokenizer; encoding text→ids is not — bring your own BPE for the prompt side.
- **Custom sampling.** Native `generate` is greedy-only; for top-k / top-p / temperature drive the engine yourself via `forward()` and apply your sampler in Kotlin.
- **GPU.** rm-graph is CPU-only; this module mirrors that.
- **Multiple architectures.** `.rmg v1` ships LLaMA-shaped models only.

## License

MIT — see root LICENSE.
