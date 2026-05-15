# gguf_lib runtime overhead — audit + remediation plan

Status: planning doc. None of the changes below have landed yet.
Last audit: 2026-05-15, against `gguf_lib.cpp` (4297 LOC) on branch `ai-rewrite`.

The binary-size discussion (drop VLM models, drop pdfium, etc.) is captured
in chat history but **out of scope here** — this doc is purely about runtime
overhead during inference: per-token CPU cost, per-call allocations, lock
contention, and steady-state memory.

## Global state map (3 parallel singletons)

| Singleton | Lines | Holds | Mutex |
|---|---|---|---|
| `g_state` | 314–372 | Main LLM: model, ctx, sampler, chat_template, sampling_params, system_prompt, 2× threadpool, session_tokens, prev_prompt_tokens, kv-policy ints, cancel_flag atomic, per-stage decode timings, auto-mode/thermal flags, mmap/mlock, thinking flag | `gen_mutex` |
| `g_embed` | 2228–2232 | Standalone embedding: model + ctx | `mutex` |
| `g_rag`   | 2549–2552 | RAG engine pointer (internally owns another model + ctx + BQ index) | `mutex` |

If chat + standalone embed + RAG are all loaded, **three independent
`llama_context` instances live in RAM**. Two of the three embedding contexts
are redundant in practice (see issue #13).

Scattered globals worth knowing:
- JNI method/class caches: `g_cb_class`, `g_onToken/Done/Error/Metrics/Progress/TokenBytes/VlmStageMetrics/VlmCacheStatus/VlmKvCacheStatus`, plus `g_embed_cb_class` + 7 prewarm callback IDs (lines 227–249).
- `g_prompt_batch` + `g_single_batch` reused across calls (lines 905–908). Good.
- `g_token_byte_buf` (4 KB jbyteArray, reused) for zero-copy token flushes.
- `g_utf8_buffer` (incomplete-multibyte carry).
- `g_token_batch_threshold` (256-byte flush trigger).
- `g_sampler_needs_rebuild`, `g_chat_templates_tried`, `g_backend_init_flag`.

`.bss` total ≈ 263 KB. Globals themselves are tiny until a model loads.

## The 16 findings

### High-cost per-token issues (decode loop, `gguf_lib.cpp:1539–1618`)

**1. Fresh `std::string unsent` allocation every token (line 1568)**
```cpp
std::string unsent(generated_text.data() + unsent_start, unsent_len);
size_t stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_FULL);
...
stop_pos = antiprompt.find_stop(unsent, (size_t)n, STOP_PARTIAL);
```
Comment at line 1566 claims "C++17 std::string_view not available in all NDKs" — stale; NDK 27 ships C++17 by default. One malloc/free pair per token whenever `unsent_len` > SSO (~22 bytes).
Fix: refactor `antiprompt_state::find_stop` to take `(const char* data, size_t len, size_t last_tok, stop_type)`.
Saves ~500 ns – 1 μs per token.

**2. `find_stop` runs twice per token, scans full stops list both times**
- STOP_FULL: `for (auto & word : stops) text.find(word, from)` per stop.
- STOP_PARTIAL: `find_partial` loops `len = word.size()-1 .. 1` with `text.compare` each.
- 4–6 stop strings × twice = 8–12 substring ops per token.
- Fix (conservative, semantic-preserving): precompute `max_stop_len` at `set_stops()`, only scan trailing `max_stop_len + last_tok_size` bytes.
- Fix (aggressive — skip): if `last_token_size == 0` and the new bytes contain no prefix of any stop, skip STOP_PARTIAL. Higher correctness risk; pick conservative form.

**3. 5× `clock_gettime` per token (lines 1542, 1545, 1555, 1598, 1616)**
vDSO-accelerated on ARM64 (~30–80 ns each), so 5× ≈ 200–400 ns/token.
For decode at 15 t/s this is noise (μs/s). For embedder at 500 t/s during RAG ingest it's 100–200 μs/s.
Fix: gate per-stage timing behind `g_state.profile_decode` flag (default OFF in release). `getLastDecodeBreakdown` returns zeros until enabled.

**4. `llama_n_ctx(g_state.ctx)` called per token (line 1602)**
Crosses TU boundary. Cache once at load: `g_state.n_ctx_cached`. Saves one call/token.

**5. `env->ExceptionCheck()` every token (line 1593)**
Tens of ns each. Needed for cancellation robustness. KEEP — note only.

### Per-flush issues (every ~256 bytes generated)

**6. `sanitize_utf8` rebuilds when not pure ASCII (line 864)**
Already has ASCII fast path. For non-ASCII scripts, every flush copies. No clean fix without complicating the Kotlin contract. KEEP — note only.

**7. `NewStringUTF` fallback when `g_onTokenBytes` is null (line 875)**
Modified-UTF-8 conversion + Java String alloc. The byte-buf fast path (`g_onTokenBytes`) is taken IF Kotlin's StreamCallback declares `onTokenBytes(byte[], int)`. Verify this is actually wired — otherwise every flush takes the slow path.

### Per-call issues (once per generate)

**8. `apply_chat_template` re-renders full history every multi-turn call (line 1666+)**
Single-turn already optimal (system prompt cached via tokenized prefix at 1481). Multi-turn re-renders ENTIRE history each turn. 20-turn chat = 20× rendering cost. A few ms of TTFT.
Fix: cache rendered template prefix for `messages[0..n-2]`, keyed by a content hash; invalidate on `setSystemPrompt` / `setChatTemplate` / `setThinkingEnabled`. Some templates embed turn-varying state (e.g., reasoning toggles), so check applicability per template.

**9. `tokenize_string` over-allocates (line 891)**
```cpp
int n_tokens = text.size() + 256;
std::vector<llama_token> tokens(n_tokens);
```
For 10K-char prompt → 40 KB allocation, then `resize(n)` shrinks (alloc already happened).
Fix: start with `text.size() / 3 + 16` (BPE averages 3–4 chars/token). The retry-on-undersize path (lines 894–898) catches the 5% case.

**10. `compute_memory_metrics` at end of every generate (line 1641)**
Reads `/proc/self/status` + `/proc/meminfo`. 100–500 μs/call. Inflates reported `total_ms`.
Fix: make optional via flag (default ON for backward compat); UI can opt out. Or at minimum cache `MemTotal` (never changes).

### Embedding-engine specific

**11. `llama_batch_init/free` per `encodeText` (lines 2317, 2330)**
Unlike `g_single_batch`, embed batch is alloc+freed per call. RAG ingest with 1000 chunks = 1000 alloc/free cycles.
Fix: reuse `g_embed.batch` like `g_single_batch`. Or drop the whole standalone engine — see #13.

### Steady-state memory (not binary — runtime VmSize)

**12. ggml threadpool default 1 MB stack × N threads × 2 pools ≈ 8 MB VmSize/model**
Most pages never faulted (RSS unaffected), but VmSize bloats and page-table overhead is real.
Fix: patch ggml-cpu's `ggml_threadpool_new` to call `pthread_attr_setstacksize(128 * 1024)`. Requires upstream-side validation that no ggml internal exceeds 128 KB stack. Stage at 256 KB first.

**13. Three `llama_context` instances in RAM if all engines used**
`g_embed` duplicates the embedding model that `g_rag` already holds internally.
Saves a full duplicate embed model (50–400 MB depending on model).
**Constraint**: `app/src/main/java/com/dark/tool_neuron/service/server/ServerEmbeddingEngine.kt:8` calls `EmbeddingEngine()`. Naive deletion breaks `:server`.
Fix path (a, recommended): keep Kotlin `EmbeddingEngine` API + 3 JNI fns, but route them to share `g_rag.engine`'s embedding model. Requires adding `rag_engine_encode(text, normalize, float*) -> int` to `engine/rag-engine.cpp`. `nativeEncodeText` becomes a thin wrapper that asserts `g_rag.engine != nullptr && rag_engine_is_loaded(...)`.
Fix path (b): migrate `ServerEmbeddingEngine` to use `RAGEngine` directly, then delete the standalone. Larger ToolNeuron diff.

**14. `g_state.prev_prompt_tokens` bounded at `n_ctx` (typical 16 KB).** Fine — note only.

**15. `g_state.session_tokens` — same bound.** Fine — note only.

### Lock contention

**16. `gen_mutex` held for the entire decode loop**
Anything reading state from another thread (`getContextUsage`, `getMemoryStatsJson`) blocks until generation finishes.
Decode itself isn't slowed, but UI poll-during-generate stalls.
Fix: add `std::atomic<int> n_past_shadow` field; decode loop updates it after each token; `getContextUsage` reads `n_past_shadow / n_ctx_cached` lock-free.

## Risk audit summary

| # | Risk class | Mitigation needed? |
|---|---|---|
| 1, 4, 9, 11, 16 | 🟢 Zero behavioral risk | None |
| 5, 6, 14, 15 | 🟢 No change | None |
| 7 | 🟢 Diagnostic | Verify Kotlin side, add fallback if missing |
| 3, 10 | 🟡 Mild behavior change | Default flag preserves old contract |
| 2 | 🟠 Could miss stops if wrong | Use conservative form, add unit tests |
| 8 | 🟠 Cache staleness | Content-hash key, invalidate on state change |
| 13 | 🔴 Breaks `:server` if naive | Route through `g_rag` (option a) — keeps Kotlin API |
| 12 | ⚙️ Upstream patch | Stage 256 KB → 128 KB; full validation |

## Execution batches

Each batch = one AAR rebuild cycle (`AiSystems/gguf_lib` → `ToolNeuron-New/libs/gguf_lib-release.aar`).

### Batch A — safe wins (no toggles, no API change)
Files: `gguf_lib/src/main/cpp/gguf_lib.cpp`
- #1: refactor `antiprompt_state::find_stop` to take `(const char*, size_t, size_t, stop_type)`. Drop the `std::string unsent` temp at line 1568.
- #4: add `int n_ctx_cached` to `g_state`; set at load (in `nativeLoadModel` after `llama_init_from_model`), clear at release; replace `llama_n_ctx(g_state.ctx)` at line 1602 with the cached value.
- #9: change `int n_tokens = text.size() + 256;` (line 890) to `int n_tokens = std::max((int)(text.size() / 3) + 16, 32);`. Retry path already covers undersize.
- #11: add `llama_batch batch{}; bool batch_init=false;` to `g_embed`. In `nativeEncodeText`: if not init, init to a chosen max (e.g. embed `n_ctx`); else `common_batch_clear(g_embed.batch)`. Remove the `llama_batch_free` lines. Free in `nativeReleaseEmbeddingModel`.
- #16: add `std::atomic<int> n_past_shadow{0}` to `g_state`; bump after `g_state.n_past++` in both single-turn (line ~1614) and multi-turn decode loops; `nativeGetContextUsage` reads `n_past_shadow.load() / (float)g_state.n_ctx_cached`.

### Batch B — toggles (default preserves old behavior)
Files: `gguf_lib.cpp`, `GGUFNativeLib.kt`, `GGMLEngine.kt`
- #3: add `bool profile_decode = false;` to `g_state`. Guard the 5× `clock_gettime` blocks with `if (g_state.profile_decode)`. Add JNI: `nativeSetProfileDecode(bool)`. Add Kotlin: `engine.setProfileDecode(true)`. Default OFF.
- #10: add `bool metrics_enabled = true;` to `g_state` (default ON, opt-out). Guard `compute_memory_metrics` call at line 1641. Add JNI + Kotlin toggle. Also cache `MemTotal` (re-read once at startup; never changes).

### Batch C — needs unit tests
Files: `gguf_lib.cpp` + a new C++ unit test file
- #2: at `antiprompt_state::set_stops`, compute `size_t max_stop_len = max(word.size() for word in stops)`. In `find_stop`: scan only `text.size() - (max_stop_len + last_tok_size) … text.size()`. Verify behavior on:
  - empty stops list
  - stops that overlap (e.g., `"<|im"`, `"<|im_end|>"`)
  - multi-byte UTF-8 inside a stop word
  - `last_tok_size == 0`
  - last_tok_size > max_stop_len
- #8: extract `apply_chat_template` cache:
  - Hash messages[0..n-2] content into `uint64_t prefix_hash` (FNV-1a on concatenated role+content strings).
  - On match: reuse cached `rendered_prefix` + cached `stops`; only render the trailing message.
  - Invalidate on: `setSystemPrompt`, `setChatTemplate`, `setThinkingEnabled`, model reload.
  - Test: 5-turn chat, edit turn 2, verify cache invalidated.

### Batch D — coordinated with `:server`
Files: `gguf_lib.cpp`, `engine/rag-engine.{h,cpp}`, optionally `ServerEmbeddingEngine.kt`
- #13 option (a): add `int rag_engine_encode(rag_engine_t*, const char* text, bool normalize, float* out, int out_cap)` returning embedding dim or negative error. In `gguf_lib.cpp`'s `nativeEncodeText`: route to `rag_engine_encode(g_rag.engine, ...)` when `g_rag.engine && rag_engine_is_loaded`. Fall back to `g_embed` only if RAG isn't loaded (back-compat). Eventually deprecate `g_embed` once all callers have RAG engine warm.
- Verify: `ServerEmbeddingEngine.kt` still compiles + works; `EmbeddingEngine` Kotlin API unchanged.

### Batch E — upstream validation (separate session)
Files: forked `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp` (find `ggml_threadpool_new`)
- #12: insert `pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setstacksize(&a, 256 * 1024);` before each `pthread_create`. Test full model battery (Qwen3, SmolLM3, etc.) for SIGSEGV. Drop to 128 KB if stable.

## Rebuild cycle

```
cd /home/home/AndroidStudioProjects/AiSystems
./gradlew :gguf_lib:assembleRelease    # produces gguf_lib/build/outputs/aar/gguf_lib-release.aar
cp gguf_lib/build/outputs/aar/gguf_lib-release.aar \
   /home/home/AndroidStudioProjects/ToolNeuron-New/libs/gguf_lib-release.aar
cd /home/home/AndroidStudioProjects/ToolNeuron-New
./gradlew :app:assembleRelease         # picks up new AAR via files("../libs/...")
```

Build pin reminders (from `gguf_lib/build.gradle.kts` + monorepo CLAUDE):
- NDK 27.3.13750724
- CMake 3.31.4
- C++17
- arm64-v8a primary; x86_64 for emulator only

## Order to land

Recommended order: A → B → D → C → E (defer E to its own session for validation).

A is small, all-additive, single rebuild. B adds toggles, ships defaults that match today's behavior — zero-impact rollout. D needs `:server` smoke-test. C is the biggest correctness risk (chat templating cache) and gets isolated. E is upstream and needs runtime validation across model families.

## Outcomes to expect

Per-token decode (15 t/s baseline on 7s Gen 3, 1B model):
- Saves ~1–1.5 μs/token from #1 + #4 + #3 + #2.
- At 15 t/s, ~22 μs/s — won't move tok/s perceptibly; matters at 100+ t/s.

Per-call:
- TTFT savings from #8 (multi-turn): scales with history length; tens of ms on 20-turn chats.
- Allocation pressure reduction from #9, #11: helps when malloc fragmenting matters (long sessions).

Steady-state RAM:
- #13 saves a full duplicate embedding model (50–400 MB).
- #12 saves ~7 MB VmSize (not RSS).

UI responsiveness:
- #16: `getContextUsage` becomes lock-free; UI poll-during-generate stops stalling.

Net: micro-optimizations + memory wins. No tok/s revolution — that's bounded by `llama_decode` itself, which is the floor we can't move without changing the model or backend.
