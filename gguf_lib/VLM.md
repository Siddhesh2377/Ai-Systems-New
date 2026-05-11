# VLM integration guide — gguf_lib

How to drive a vision-language model (text + images) through the SDK,
with two persistent caches that compound:

- **VT (Vision Token) cache** — skips the ~10s ViT pass on repeat queries
  against the same image.
- **VLM-KV cache** — strictly stronger: on hit, the LLM context state
  captured at the post-image boundary is restored, skipping BOTH the ViT
  pass AND the ~9s LLM image-prefill. TTFT drops from ~10s to ~hundreds
  of ms for "same image, different question" workflows.

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

        // 4) Open the VLM-KV cache (bigger entries, ~5–15 MB each)
        engine.vlmKvCacheInit(
            dir         = File(app.filesDir, "vlm_kv_cache").absolutePath,
            budgetBytes = 300L * 1024L * 1024L,
        )
    }

    fun release() {
        // Order matters: caches first, then projector, then text model
        engine.vlmKvCacheRelease()
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

    // VLM-KV key (32-byte SHA256). Stronger than vtKey: covers the *whole*
    // pre-question state (system prompt, chat template, image, projector,
    // model). On hit, both the ViT pass AND the ~9s LLM image-prefill are
    // skipped. ALL inputs to this hash must be stable for the lifetime of
    // the cached entry — change the system prompt and every entry is dead.
    val vlmKvKey: ByteArray = engine.computeVlmKvKey(
        imageBytes         = imageBytes,
        projectorPath      = projectorGgufPath,
        imageMaxTokens     = 256,
        modelFingerprint   = "$repoId:$textFilename",   // anything stable
        systemPrompt       = currentSystemPrompt,
        chatTemplatePrefix = "<__image__>\n",           // text between template + question
    )

    engine.generateVlmFlow(
        messagesJson = messagesJson,
        imageData    = listOf(imageBytes),
        maxTokens    = 512,
        vtKeys       = listOf(vtKey),                  // null to bypass VT cache
        vlmKvKey     = vlmKvKey,                       // null to bypass VLM-KV cache
    ).collect { event ->
        when (event) {
            is GenerationEvent.Token             -> append(event.text)
            is GenerationEvent.Progress          -> updatePrefillProgress(event.progress)
            is GenerationEvent.VtCacheStatus     -> showVtChip(event.hit)            // see §4
            is GenerationEvent.VlmKvCacheStatus  -> showVlmKvChip(event.hit)         // see §4
            is GenerationEvent.VlmStageMetrics   -> showEncodeDecode(event)
            is GenerationEvent.Metrics           -> showFinalMetrics(event.metrics)
            is GenerationEvent.Done              -> onDone()
            is GenerationEvent.Error             -> onError(event.message)
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
VlmKvCacheStatus(hit=…)           ← once, before any decode
VtCacheStatus(hit=…)              ← per image, only if VLM-KV missed
VlmStageMetrics(encMs, decMs, T)  ← once, after image+text prompt-eval
Progress(p)…                      ← repeated; 0..1 over prompt-eval
Token("text")…                    ← one per native batch
Metrics(...)                      ← once, terminal
Done                              ← terminal
```

State table:

| Outcome | `VlmKvCacheStatus.hit` | `VtCacheStatus.hit` | What ran |
|---|---|---|---|
| Cold | false | false (or absent) | ViT + LLM image-prefill + user-text decode |
| VT hit only | false | true | LLM image-prefill + user-text decode |
| VLM-KV hit | true | (skipped) | only the user-text decode after restore |

Drive UI chips off both events. See
`app/src/main/java/com/dark/demon_system/ui/vlm/VlmScreen.kt` for a
concrete example with a 6-cell metrics grid.

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

## 5b. VLM-KV cache management (the bigger TTFT win)

Stores the **LLM context state** captured at the post-image-chunk boundary
during VLM prompt-eval. On hit, both the vision encoder AND the ~9s LLM
image-prefill are skipped — TTFT drops from ~10s to ~hundreds of ms.

```kotlin
engine.vlmKvCacheInit(dir, budgetBytes = 300L * 1024L * 1024L)

engine.vlmKvCacheStatsJson()
//  {"initialized":true,"total_bytes":12300000,"budget_bytes":314572800,
//   "entry_count":1,"hits":2,"misses":1}

engine.vlmKvCacheListEntriesJson()
//  [{"hash":"7a4c…","n_tokens":238,
//    "size_bytes":12300000,"last_access_ms":1714060000000}]

engine.vlmKvCacheRemove(hashByteArray)
engine.vlmKvCacheClear()
engine.vlmKvCacheSetBudget(500L * 1024 * 1024)
engine.vlmKvCacheRelease()
```

### Key derivation

The cached state depends on every byte that was decoded into the LLM up
through the last image chunk. The key MUST cover:

- Image bytes
- Projector path (changes ⇒ different mtmd output geometry)
- `imageMaxTokens` (changes ⇒ different image-token count)
- Model fingerprint (changes ⇒ different KV layout, set_data fails)
- System prompt (changes ⇒ pre-image text is different)
- Chat-template prefix between system and the image marker

Use the canonical derivation:

```kotlin
val vlmKvKey = engine.computeVlmKvKey(
    imageBytes         = imageBytes,
    projectorPath      = projectorGgufPath,
    imageMaxTokens     = 256,
    modelFingerprint   = "$repoId:$textFilename",
    systemPrompt       = currentSystemPrompt,
    chatTemplatePrefix = "<__image__>\n",
)
```

### Storage budget

Per entry size = `n_pre_image_tokens + n_image_tokens` rows of KV cache,
serialized via `llama_state_seq_get_data`. Rough numbers:

| Model | Layers | KV head dim | KV dtype | Tokens (sys+img) | Bytes/entry |
|---|---:|---:|---|---:|---:|
| Qwen3-VL-2B | 24 | 1024 | q8_0 | ~240 | ~12 MB |
| Qwen3-VL-4B | 36 | 1024 | q8_0 | ~240 | ~18 MB |
| LFM2-VL    | varies | varies | q8_0 | ~240 | ~10–20 MB |

300 MB budget ≈ 25 cached scenarios. Bump to 1 GB if your app keeps a
larger working set.

### Geometry validation

If the cached blob doesn't match the current model's KV layout (e.g. the
host upgraded the model since the cache was written), `llama_state_seq_set_data`
returns 0 and we fall back to fresh decode + overwrite the entry. The host
shouldn't see a hard failure, just a `VlmKvCacheStatus(hit=false)` and a
log warning.

### Failure modes

- Geometry mismatch (model reload, different ctx params): cache returns
  miss, recovers automatically next call.
- Stale entry (system prompt changed but key didn't): the host's
  responsibility — make sure the system prompt is part of the key.
- Disk full: `vlm_kv_cache_store` returns false, generation completes
  normally. No host-side error visible.

### When NOT to use it

- One-shot prompts on each image (always misses, wastes disk write)
- Privacy: stored KV blobs *can* leak the chat template prefix and
  system prompt indirectly (via residuals). Don't ship to disk for
  PII-bearing system prompts.
- Frequent system-prompt rotation: every rotation invalidates all entries.

---

## 6. Required event handling additions

If you're upgrading from a pre-VT-cache version, the host app needs:

**`StreamCallback` got two new methods (both default no-op, so existing
implementations compile unchanged):**

```kotlin
interface StreamCallback {
    fun onToken(token: String)
    fun onMetrics(...)
    fun onVlmStageMetrics(vlmEncodeMs: Float, vlmDecodeMs: Float, imageTokens: Int) {}
    fun onVlmCacheStatus(hit: Boolean, nTokens: Int, nEmbd: Int) {}        // VT cache
    fun onVlmKvCacheStatus(hit: Boolean, nTokens: Int) {}                  // VLM-KV cache
    // …
}
```

**`GenerationEvent` got two new subclasses:**

```kotlin
data class VtCacheStatus(val hit: Boolean, val nTokens: Int, val nEmbd: Int) : GenerationEvent()
data class VlmKvCacheStatus(val hit: Boolean, val nTokens: Int) : GenerationEvent()
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

| Stage | Cold | VT hit only | VLM-KV hit |
|---|---:|---:|---:|
| ViT vision encoder | ~9.6 s | **0 ms** ⚡ | **0 ms** ⚡ |
| LLM image-prompt prefill | ~9.0 s | ~9.0 s | **~0 ms** ⚡ (state restored) |
| User-question prefill | ~50–500 ms | ~50–500 ms | ~50–500 ms |
| **TTFT** | **~18.7 s** | **~9.0 s** | **~hundreds of ms** |
| Decode | ~21 tok/s | ~21 tok/s | ~21 tok/s |

The two caches compose: VLM-KV is checked first, falls through to VT,
falls through to fresh encode + decode. Misses are free (just write on
the way out).

### Pre-warming the VT cache (`precomputeVisionEmbeddings`)

Run only the vision encoder for an image and store the embeddings in the
VT cache, without touching the LLM. The next `generateVlmFlow` call with
the same image hits the cache and skips the ~9s ViT pass — even on the
"first" user query, because you've already done the encode in the
background.

```kotlin
// Fire as soon as the user picks/imports an image
viewModelScope.launch {
    engine.precomputeVisionEmbeddings(
        imageBytes     = bytes,
        projectorPath  = projectorGgufPath,
        imageMaxTokens = 256,
    )
    // Or, if you've already derived a key:
    // engine.precomputeVisionEmbeddings(bytes, vtKey)
}
```

The `app/.../VlmScreen.kt` image-picker callback shows the canonical
fire-and-forget pattern. With pre-warm + VLM-KV cache stacked, the
"first" user query feels under a second:

| Scenario | TTFT |
|---|---:|
| Cold (no pre-warm, no caches) | ~18.7 s |
| Pre-warmed VT only (first prompt on a known image) | ~9.0 s |
| Both caches warm (any subsequent prompt on same image) | ~hundreds of ms |

The host pays the ViT cost once per image, in the background,
out-of-band from any user interaction.

### Per-op CPU/GPU routing — `opOffload` (shipped, opt-in)

> ⚠️ **Adreno 810 caveat**: opOffload=true currently triggers
> `vk::DeviceLostError` on Adreno 810's Vulkan driver during the
> 234-token image-prefill compute graph (kernel TDR). The mechanism
> works correctly (verified via `graph splits = 368 (with bs=512), 1
> (with bs=1)`); the failure is a driver-level GPU watchdog. Until
> we add per-device gating or split the image-prefill into smaller
> Vulkan dispatches, **leave opOffload=false on Adreno 810**. On
> hardware with a stable Vulkan driver (most desktop GPUs) it
> delivers the speedup described below.

Pass `opOffload = true` to `engine.load(...)` to enable per-op routing.
What this does:

1. Registers every available GPU/IGPU backend (e.g. Vulkan) with
   `ggml_backend_sched` as a *compute target* — without moving any
   layer weights off CPU memory.
2. Sets `cparams.op_offload = true`, which tells the sched to consult
   each backend's `ggml_backend_offload_op` heuristic per op.

The Vulkan backend's heuristic is **batch-size based**: ops with batch
size ≥ 32 (overridable via `GGML_OP_OFFLOAD_MIN_BATCH`) are dispatched
to GPU; everything below stays on CPU. This is the right boundary for
us:

| Workload | Batch | Where it runs | Why |
|---|---|---|---|
| Image prefill | 234 tokens × big GEMMs | **GPU** | Compute-bound; GPU GEMMs >> CPU |
| User-question prefill | typically 5–30 tokens | depends | borderline |
| Token decode | 1 token × small matvec | **CPU** | Dispatch overhead would kill tok/s |

GPU dispatch overhead on Adreno 810 is ~0.47 ms per shader call. With
~24 layers × ~8 ops/layer = 192 dispatches per token, that's 90 ms of
overhead per token before any compute — which is why we *don't* want
GPU for decode. The op_offload threshold keeps decode safely on CPU.

```kotlin
engine.load(
    path = textPath,
    flashAttn = true,
    cacheTypeK = "q8_0",
    cacheTypeV = "q8_0",
    opOffload = true,       // ← per-op CPU/GPU routing
)
```

When no GPU device is available (or libvulkan.so isn't loadable),
`opOffload = true` is a safe no-op — the sched just runs everything on
CPU as before.

#### Behavior with the caches

The two caches stack with op_offload as you'd hope:

| Path | ViT | LLM image-prefill | Decode |
|---|---|---|---|
| Cold, opOffload=false | CPU 9.6s | CPU ~9s | CPU |
| Cold, opOffload=true | CPU 9.6s | **GPU ~5s** | CPU |
| VT hit, opOffload=true | 0 | **GPU ~5s** | CPU |
| VLM-KV hit | 0 | 0 (state restored) | CPU |

So `opOffload = true` halves the *cold cached-VT* TTFT (the user's
"10s → 5s" target) while leaving decode untouched. VLM-KV hits
short-circuit everything regardless.

#### Limitations

- ViT (vision encoder) goes through mtmd's own `clip_ctx`, not
  llama_context. It's still CPU-only on cold; only the VT cache helps.
- ACCEL backends (BLAS, AMX) are unaffected.
- The Vulkan threshold of 32 is a heuristic that targets desktop GPUs
  with much higher dispatch latency than mobile UMA. On Adreno 810 the
  optimal threshold may be lower; tune via `GGML_OP_OFFLOAD_MIN_BATCH`
  if you have a way to set env vars before native init.

### Backend diagnostic

Enumerate what's registered without committing to use it:

```kotlin
val json = engine.listBackendsJson()
// {
//   "backends": [{"name": "CPU"}, {"name": "Vulkan"}],
//   "devices":  [{"name": "...", "type": "igpu", "memory_total": ..., ...}]
// }
```

Useful for surfacing "Vulkan available — see DEVICE.md for the trade-off"
in your settings UI.

### Cold-path knobs that DO work today

- **Drop `imageMaxTokens` 256 → 128**: halves prefill compute, cuts
  TTFT roughly in half on cold. Lossy.
- **Q4_K_M model + Q4_0 KV cache**: halves DRAM bandwidth → faster
  prefill *and* decode without changing backends.

### Other things NOT shipping yet

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
nativeVlmGetInfo()                                                     : String?      // {supports_vision, supports_audio, default_marker}
nativeVlmGetDefaultMarker()                                            : String

nativeVlmGenerateStream(messagesJson, imageData[], vtKeys[]?, vlmKvKey?, maxTokens, callback) : Boolean

nativeVlmPrecomputeVisionEmbeddings(imageData, vtKey[32]) : Boolean

nativeVtCacheInit(dir, budgetBytes)        : Boolean
nativeVtCacheRelease()
nativeVtCacheClear()
nativeVtCacheSetBudget(bytes)
nativeVtCacheStatsJson()                   : String
nativeVtCacheListEntriesJson()             : String
nativeVtCacheRemove(hash[32])              : Boolean

nativeVlmKvCacheInit(dir, budgetBytes)     : Boolean
nativeVlmKvCacheRelease()
nativeVlmKvCacheClear()
nativeVlmKvCacheSetBudget(bytes)
nativeVlmKvCacheStatsJson()                : String
nativeVlmKvCacheListEntriesJson()          : String
nativeVlmKvCacheRemove(hash[32])           : Boolean

nativeListBackendsJson()                   : String      // diagnostic only
```

Public Kotlin facade lives in `GGMLEngine.kt`. Use it.
