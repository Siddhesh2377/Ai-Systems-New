# VLM integration guide — gguf_lib

How to drive a vision-language model (text + images) through the SDK,
including the persistent **VT (Vision Token) cache** that skips the
~10s ViT pass on repeat queries against the same image.

This file targets the **host app's Claude Code client** — copy the
patterns below into your own ViewModel / repository layer.

---

## 0. What was removed (and is no longer in the AAR)

The following features were intentionally stripped from both `gguf_lib`
and the underlying llama.cpp fork. Don't reference them in host code
— the symbols don't exist anymore, and the AAR will not link if you
try.

| Removed | Old surface (now gone) |
|---|---|
| **Tool calling** | `ToolManager`, `nativeSetToolsJson`, grammar modes, agent loop, `<tool_call>` detector |
| **Control vectors** | `nativeLoadControlVectors`, `llama_set_adapter_cvec`, axis cache files |
| **Personality / mood** | `CharacterEngine`, refusal-token scan, dynamic emotional steering, fast-weight memory, attention temperature / head rescaling |

Inside llama.cpp `llama_adapter_cvec` and `build_cvec` calls remain as
inert no-op infrastructure (no public API to populate them). The
deeper `common/chat-parser*` machinery is kept because chat templating
needs it; only the user-facing tool-call surface that sat on top is
gone.

If your host app has dead code referencing any of the above, delete
those code paths before you upgrade the AAR.

---

## 1. Module setup

```kotlin
// settings.gradle.kts
include(":gguf_lib")

// app/build.gradle.kts
dependencies {
    implementation(project(":gguf_lib"))
}
```

The shared library auto-loads via `System.loadLibrary("gguf_lib")`
the first time `GGUFNativeLib` is referenced. No manual init.

### Native libraries shipped in the AAR

- `libgguf_lib.so` — JNI bridge + the inference engine
- `libllama.so` + `libggml*.so` — llama.cpp core (multi-variant: armv8.0 → armv9.2+SME)
- `libggml-vulkan.so` — Vulkan backend (compiled in; **not yet routed** — see §8)

### Required system libs

The Vulkan backend needs `libvulkan.so` from the device. Already
declared in `gguf_lib/AndroidManifest.xml`:

```xml
<uses-native-library android:name="libvulkan.so" android:required="false" />
```

Don't redeclare in the host manifest — manifest merging picks it up.

---

## 2. Engine lifecycle (text + projector)

```kotlin
class MyVlmRepo(app: Application) {
    private val engine = GGMLEngine()

    suspend fun load(textGgufPath: String, projectorGgufPath: String) {
        // 1) Load the text model
        val ok = engine.load(
            path        = textGgufPath,
            contextSize = 4096,
            flashAttn   = true,
            cacheTypeK  = "q8_0",
            cacheTypeV  = "q8_0",
        )
        require(ok) { "text model load failed" }

        // 2) Load the projector (mmproj GGUF)
        val vlmOk = engine.loadVlmProjector(
            path           = projectorGgufPath,
            threads        = 0,            // 0 = inherit batch threads
            imageMinTokens = -1,           // model default
            imageMaxTokens = 256,          // 256 is a good Qwen3-VL default
        )
        require(vlmOk) { "projector load failed" }

        // 3) Open the persistent VT cache (once per process)
        engine.vtCacheInit(
            dir         = File(app.filesDir, "vt_cache").absolutePath,
            budgetBytes = 200L * 1024L * 1024L,        // 200 MB LRU budget
        )
    }

    fun release() {
        // Order matters: VT first, then projector, then text model
        engine.vtCacheRelease()
        engine.releaseVlmProjector()
        // engine.unload() is suspend — call from a coroutine
    }
}
```

Notes:
- The VLM projector binds `n_threads` at init. If you change thread mode
  via `engine.setThreadMode(...)`, call `releaseVlmProjector()` and reload
  to re-bind. (Doesn't apply to the text model.)
- One model + one projector at a time, app-wide. If you need to switch,
  release first.

---

## 3. Streaming generation with images

```kotlin
suspend fun ask(prompt: String, imageBytes: ByteArray) {
    val marker = engine.getVlmDefaultMarker()        // e.g. "<__image__>"

    // Multi-turn message JSON. Place the marker where the image goes.
    val messagesJson = JSONArray().apply {
        put(JSONObject().apply {
            put("role", "user")
            put("content", "$marker\n${prompt.trim()}")
        })
    }.toString()

    // VT cache key (32-byte SHA256). Optional but strongly recommended.
    // Two different JPEG/PNG encodings of the same picture intentionally
    // hit different slots — caching is byte-content addressed.
    val vtKey: ByteArray = engine.computeVtKey(
        imageBytes     = imageBytes,
        projectorPath  = projectorGgufPath,           // same string used at load
        imageMaxTokens = 256,                          // same value used at load
    )

    engine.generateVlmFlow(
        messagesJson = messagesJson,
        imageData    = listOf(imageBytes),
        maxTokens    = 512,
        vtKeys       = listOf(vtKey),                  // null to bypass cache
    ).collect { event ->
        when (event) {
            is GenerationEvent.Token          -> append(event.text)
            is GenerationEvent.Progress       -> updatePrefillProgress(event.progress)
            is GenerationEvent.VtCacheStatus  -> showCacheChip(event.hit)         // see §4
            is GenerationEvent.VlmStageMetrics-> showEncodeDecode(event)          // see §4
            is GenerationEvent.Metrics        -> showFinalMetrics(event.metrics)
            is GenerationEvent.Done           -> onDone()
            is GenerationEvent.Error          -> onError(event.message)
        }
    }
}
```

Cancelling the collecting coroutine is the canonical way to stop —
`engine.stopGeneration()` is also exposed and is idempotent.

---

## 4. Event timeline

For a single-image, single-turn call, the event order is:

```
VtCacheStatus(hit=…)             ← per image, before any decode
VlmStageMetrics(encMs, decMs, T) ← once, after image+text prompt-eval
Progress(p)…                     ← repeated; 0..1 over prompt-eval
Token("text")…                   ← one per native batch
Metrics(...)                     ← once, terminal
Done                             ← terminal
```

- `VtCacheStatus.hit == true`  → cached embeddings reused; `VlmStageMetrics.vlmEncodeMs ≈ 0`
- `VtCacheStatus.hit == false` → ViT ran fresh; embeddings stored to disk on the way out

Use `VtCacheStatus` to drive a UI chip ("⚡ cached" / "miss"). See
`app/src/main/java/com/dark/demon_system/ui/vlm/VlmScreen.kt` for a
concrete example.

---

## 5. VT cache management

The cache is **content-addressed by SHA256** of `(image bytes ∥
projector path ∥ image_max_tokens)`. Files live under the directory
you passed to `vtCacheInit(...)`. Format: a small header
(`{magic=0x4E4B5456, version=1, n_tokens, n_embd, …}`) followed by raw
float32 embeddings. Atomic writes (`.tmp` + rename), LRU eviction by
`last_access_ms` once the budget is exceeded.

```kotlin
engine.vtCacheInit(dir, budgetBytes = 200L * 1024L * 1024L)

engine.vtCacheStatsJson()
//  {"initialized":true,"total_bytes":7340032,"budget_bytes":209715200,
//   "entry_count":1,"hits":3,"misses":1}

engine.vtCacheListEntriesJson()
//  [{"hash":"3f2c…","n_tokens":234,"n_embd":8192,
//    "size_bytes":7340032,"last_access_ms":1714060000000}]

engine.vtCacheRemove(hashByteArray)     // drop one entry
engine.vtCacheClear()                    // drop everything on disk
engine.vtCacheSetBudget(500L*1024*1024)  // resize at runtime; LRU-evicts immediately if over
engine.vtCacheRelease()                  // close index (files persist)
```

### Choosing a budget

Per-image cost is `n_image_tokens × n_embd_inp × 4 bytes`. For
Qwen3-VL-2B at `imageMaxTokens=256`:

| `imageMaxTokens` | tokens/image | bytes/image       |
|---:|---:|---:|
| 64  | ~64  | 2.0 MB            |
| 256 | ~234 | 7.5 MB            |
| 512 | ~478 | 15.3 MB           |

200 MB ≈ 25 cached overview images at the default. Bump if your app
keeps a working set bigger than that.

### When NOT to use the cache

- One-shot pipelines (cache won't fire twice on the same key)
- Privacy-sensitive flows where embeddings on disk are unacceptable
- Models where `imageMaxTokens` varies per call (cache key changes,
  every call misses) — pass `vtKeys = null` instead

---

## 6. Required event handling additions

If you're upgrading from a pre-VT-cache version, the host app needs:

**`StreamCallback` got a new method (default no-op, so existing
implementations compile unchanged):**

```kotlin
interface StreamCallback {
    fun onToken(token: String)
    fun onMetrics(...)
    fun onVlmStageMetrics(vlmEncodeMs: Float, vlmDecodeMs: Float, imageTokens: Int) {}
    fun onVlmCacheStatus(hit: Boolean, nTokens: Int, nEmbd: Int) {}        // ← new
    // …
}
```

**`GenerationEvent` got one new subclass:**

```kotlin
data class VtCacheStatus(val hit: Boolean, val nTokens: Int, val nEmbd: Int) : GenerationEvent()
```

If your `when (event)` blocks were exhaustive, add the new branch. The
SDK's own `streamCallback(...)` helper inside `GGMLEngine.kt` already
forwards it to the flow, so direct flow consumers just need the
`when` branch.

---

## 7. Recommended HuggingFace download pattern

The test app's `VlmModelDownloader` (under `app/src/main/java/com/dark/demon_system/data/`)
shows the canonical pattern:

```kotlin
// HF resolve URL — works for public repos without auth
"https://huggingface.co/$repoId/resolve/main/$filename?download=true"
```

Tested model: **`Qwen/Qwen3-VL-2B-Instruct-GGUF`**

| File | Purpose | Size |
|---|---|---:|
| `Qwen3-VL-2B-Instruct-Q8_0.gguf` | text model | 1.83 GB |
| `mmproj-Qwen3-VL-2B-Instruct-Q8_0.gguf` | vision projector (mmproj) | 445 MB |

Use `channelFlow` (NOT `flow {}`) when wrapping `withContext(IO)` write
loops — the plain `flow {}` builder rejects emissions from a different
context and crashes with a flow-invariant violation.

---

## 8. Performance reality (Snapdragon 7s Gen 3, Adreno 810)

CPU-only baseline at the time of writing:

| Stage | Cold | VT cache hit |
|---|---:|---:|
| ViT vision encoder | ~9.6 s | **0 ms** ⚡ |
| LLM image-prompt prefill | ~9.0 s | ~9.0 s |
| TTFT (image + 1-token prompt) | ~18.7 s | **~9.0 s** |
| Decode | ~21 tok/s | ~21 tok/s |

On hit, the VT cache halves time-to-first-token. On miss it's free
(it just writes the embeddings on the way through).

### What's NOT shipping yet

- **Per-op CPU/GPU routing.** Vulkan is compiled into the AAR but
  no op is dispatched to the GPU. The mtmd ViT path is hardcoded to
  the CPU backend in `clip.cpp`. The proper fix is `ggml_backend_sched`
  + an op-router callback (designed; ~80 lines of llama.cpp patches).
  Don't pass any `useGpu` flag — there isn't one, and the previous
  attempt to add one was rejected as the wrong abstraction (it offloads
  whole layers, doesn't route per op).
- **Image quality / resize enum** (LOW / MEDIUM / HIGH) — designed but
  not wired through JNI yet.

---

## 9. Putting it together — minimal ViewModel

Reference: `app/src/main/java/com/dark/demon_system/ui/vlm/VlmViewModel.kt`
in this repo. It's the canonical pattern: load order, key derivation,
event handling, and teardown order all match this guide.

---

## 10. Troubleshooting

- **`UnsatisfiedLinkError: nativeVtCache*`** → AAR is stale. Rebuild
  `:gguf_lib` (it ships these symbols since the May 2026 build) and
  re-sync the consuming module.
- **`vtCacheInit` returns `false`** → the directory is unwritable, or
  budget is non-positive. Check the path and `budgetBytes > 0`.
- **`generateVlmFlow` errors with "no projector loaded"** → call
  `loadVlmProjector(...)` after `load(...)`, before generation.
- **`VtCacheStatus` never fires** → you didn't pass `vtKeys`, or the
  list size doesn't match `imageData`. Check both. The native side
  treats a length-mismatched array as "no key for any image".
- **Cache always misses on the same image** → you're hashing different
  byte sequences. Two re-encoded JPEGs of the same picture *will* have
  different SHA256s. Decode + re-encode at a fixed resolution before
  hashing if you need pixel-level cache hits across recompressions.

---

## 11. JNI surface reference (cheat sheet)

All under `GGUFNativeLib` (internal to the AAR — go through `GGMLEngine`).

```
nativeVlmLoadProjector(path, nThreads, imageMinTokens, imageMaxTokens) : Boolean
nativeVlmLoadProjectorFromFd(fd, ...)                                  : Boolean
nativeVlmRelease()
nativeVlmIsLoaded()                                                    : Boolean
nativeVlmGetInfo()                                                     : String?      // {supports_vision, supports_audio, default_marker}
nativeVlmGetDefaultMarker()                                            : String

nativeVlmGenerateStream(messagesJson, imageData[], vtKeys[]?, maxTokens, callback) : Boolean

nativeVtCacheInit(dir, budgetBytes)        : Boolean
nativeVtCacheRelease()
nativeVtCacheClear()
nativeVtCacheSetBudget(bytes)
nativeVtCacheStatsJson()                   : String
nativeVtCacheListEntriesJson()             : String
nativeVtCacheRemove(hash[32])              : Boolean
```

Public Kotlin facade lives in `GGMLEngine.kt`. Use it.
