# dLLM: Pure ggml Diffusion Language Model Engine

**Date**: 2026-03-03
**Status**: Approved
**Location**: `/home/home/dev/include/dllm/`
**ggml source**: `/home/home/dev/include/llama.cpp/ggml/`

---

## Goal

Build a standalone C++ diffusion language model inference engine using raw ggml tensors.
No llama.cpp runtime dependency. Test via ADB executable on Android arm64-v8a.
Later integrate into AiSystems SDK.

## Target Model

**Qwen3-0.6B-diffusion-mdlm-v0.1** (from dllm-hub on HuggingFace)
- Architecture: Qwen3 transformer (same as gguf-forward-test.cpp)
- Vocab: ~151K tokens
- MASK token ID: stored in model metadata (tokenizer.mask_token_id)
- Params: ~0.6B
- Quantization target: Q4_0 / Q8_0

## Variant

**Phase 1: MDLM** (Masked Diffusion Language Modeling)
- Pure bidirectional attention (no causal mask)
- No KV cache needed
- Iterative confidence-based remasking

**Roadmap: BD3LM** (Block Diffusion)
- Block-causal attention (bidirectional within block, causal between blocks)
- KV cache reuse between blocks
- Fast-dLLM optimizations (prefix cache, confidence-aware parallel decoding)

---

## Architecture

### File Structure

```
/home/home/dev/include/dllm/
├── dllm.h              # Public C API
├── dllm_model.cpp      # GGUF reader, weight mapping, memory-mapped loading
├── dllm_graph.cpp      # ggml computation graph — bidirectional transformer
├── dllm_sampler.cpp    # MDLM diffusion loop: Gumbel noise + confidence remasking
├── dllm_tokenizer.cpp  # BPE tokenizer (Qwen3), MASK token support
├── dllm_main.cpp       # CLI test executable for ADB
├── CMakeLists.txt      # Build config (Android NDK arm64-v8a + host x86_64)
└── scripts/
    └── convert_to_gguf.py  # HuggingFace → GGUF conversion
```

### Dependencies

- **ggml**: Tensor math, quantization, SIMD backends
  - Source: `/home/home/dev/include/llama.cpp/ggml/`
  - Headers: `ggml.h`, `ggml-alloc.h`, `ggml-backend.h`, `ggml-cpu.h`
  - Backends: CPU (NEON on ARM), optionally Vulkan/OpenCL later
- **gguf**: Model file format
  - Source: `gguf.h` / `gguf.cpp` from ggml
- **No other dependencies** — pure C/C++, no Python at runtime

---

## Algorithm: MDLM Inference

### Core Loop (pseudocode)

```
Input: prompt tokens, gen_length, n_steps, temperature

1. canvas = [prompt_tok₀, prompt_tok₁, ..., MASK, MASK, ..., MASK]
                                              ← gen_length MASKs →

2. transfer_schedule = even_split(gen_length, n_steps)
   // e.g. 128 tokens / 32 steps = 4 tokens per step

3. for step = 0 to n_steps-1:
     a. logits = forward(canvas)              // bidirectional, all positions
        // logits shape: [seq_len, vocab_size]

     b. x0 = gumbel_argmax(logits, temp)      // sample predicted tokens
        // for each position: argmax(exp(logits) / gumbel_noise^temp)

     c. conf[i] = softmax(logits[i])[x0[i]]   // confidence = prob of prediction
        // use CLEAN logits (not noisy) for confidence

     d. For masked positions only:
        - Sort by confidence descending
        - Unmask top transfer_schedule[step] positions → commit x0 tokens
        - Everything else stays MASK

4. return canvas[prompt_len:]
```

### Forward Pass (single diffusion step)

Identical to Qwen3 transformer EXCEPT no causal mask:

```
embed = token_embedding[canvas]                    // [seq_len, d_model]

for layer in 0..n_layers:
    h = rms_norm(embed)
    Q = h @ Wq,  K = h @ Wk,  V = h @ Wv         // projections
    Q = rope(Q),  K = rope(K)                       // rotary pos encoding

    attn = (Q @ K^T) / sqrt(d_head)                // attention scores
    // NO ggml_diag_mask_inf here — full bidirectional
    attn = softmax(attn)
    out = attn @ V
    out = out @ Wo
    embed = embed + out                             // residual

    h2 = rms_norm(embed)
    ffn = silu(h2 @ W_gate) * (h2 @ W_up)          // SwiGLU
    ffn = ffn @ W_down
    embed = embed + ffn                             // residual

logits = rms_norm(embed) @ lm_head                  // [seq_len, vocab_size]
```

### Gumbel Noise Sampling

```cpp
// When temperature == 0: pure argmax (greedy)
// When temperature > 0: Gumbel-max trick
for (int v = 0; v < vocab_size; v++) {
    double u = uniform_random(0, 1);               // U(0,1)
    double noise = pow(-log(max(u, 1e-20)), temperature);
    score[v] = exp((double)logits[v]) / noise;
}
x0 = argmax(score);
```

**Critical**: Use float64 for Gumbel computation. Low-precision degrades generation quality.

### Confidence Scoring

```cpp
// softmax on CLEAN logits (not noisy)
float* probs = softmax(logits[pos], vocab_size);
float confidence = probs[x0[pos]];
```

### Transfer Schedule (linear)

```cpp
int base = n_masked / n_steps;
int remainder = n_masked % n_steps;
for (int i = 0; i < n_steps; i++)
    schedule[i] = base + (i < remainder ? 1 : 0);
```

---

## GGUF Model Format

### Weight Names (Qwen3 architecture)

```
token_embd.weight                         // [vocab_size, d_model]
blk.{i}.attn_norm.weight                  // [d_model]
blk.{i}.attn_q.weight                     // [d_model, d_model]
blk.{i}.attn_k.weight                     // [d_model, n_kv_heads * d_head]
blk.{i}.attn_v.weight                     // [d_model, n_kv_heads * d_head]
blk.{i}.attn_output.weight                // [d_model, d_model]
blk.{i}.ffn_norm.weight                   // [d_model]
blk.{i}.ffn_gate.weight                   // [d_model, ffn_dim]
blk.{i}.ffn_up.weight                     // [d_model, ffn_dim]
blk.{i}.ffn_down.weight                   // [ffn_dim, d_model]
output_norm.weight                         // [d_model]
output.weight                              // [vocab_size, d_model] (lm_head)
```

### Custom Metadata

```
dllm.mask_token_id    = <int>     // MASK special token ID
dllm.variant          = "mdlm"   // or "bd3lm"
dllm.is_diffusion     = true     // flag for diffusion model
```

### Conversion Script

Python script to convert HuggingFace Qwen3-0.6B-diffusion to GGUF:
1. Load safetensors weights
2. Map HF names → GGUF names
3. Quantize (Q4_0 or Q8_0)
4. Write GGUF with custom metadata

---

## Public API

```c
// dllm.h — Clean C API

typedef struct dllm_context dllm_context;

typedef struct {
    int32_t  n_threads;        // CPU threads (0 = auto)
    int32_t  n_ctx;            // max sequence length
    bool     use_mmap;         // memory-map model file
} dllm_params;

typedef struct {
    int32_t  n_steps;          // diffusion steps (default: 64)
    int32_t  gen_length;       // tokens to generate
    float    temperature;      // Gumbel temperature (0 = greedy)
    int32_t  remasking;        // 0=low_confidence, 1=random
} dllm_sampling;

// Lifecycle
dllm_context * dllm_create(dllm_params params);
int            dllm_load_model(dllm_context * ctx, const char * gguf_path);
void           dllm_free(dllm_context * ctx);

// Generation
typedef bool (*dllm_step_callback)(int step, int total_steps, const char * current_text, void * user_data);

int  dllm_generate(
    dllm_context * ctx,
    const char * prompt,
    dllm_sampling sampling,
    dllm_step_callback callback,    // called after each diffusion step
    void * user_data
);

char * dllm_get_result(const dllm_context * ctx);
void   dllm_free_string(char * str);

// Infilling (native capability of diffusion models)
int  dllm_infill(
    dllm_context * ctx,
    const char * prefix,
    const char * suffix,
    int32_t max_fill_tokens,
    dllm_sampling sampling,
    dllm_step_callback callback,
    void * user_data
);

// Cancel (thread-safe)
void dllm_cancel(dllm_context * ctx);
```

Key difference from autoregressive API: the callback fires per **diffusion step** (not per token), and the text progressively reveals from masks.

---

## Performance Characteristics

### Memory (Qwen3-0.6B Q4_0)
- Model weights: ~350MB
- Activation memory per forward pass: ~50MB (no KV cache!)
- Total: ~400MB peak
- Fixed regardless of generation length (unlike AR where KV cache grows)

### Compute
- Each diffusion step = 1 full forward pass over entire sequence
- N steps × full sequence = total compute
- 64 steps × 256 tokens ≈ 64 forward passes
- At ~47ms/token for Qwen3-0.6B on ARM: ~47ms × 256 positions = ~12s per step
- 64 steps × 12s = ~768s total (too slow — need to optimize)
- **Optimization**: 16-32 steps with confidence-aware parallel decoding → viable

### Advantages Over Autoregressive
- No KV cache: fixed memory, no growing allocations
- All positions computed at once: natural for SIMD/GPU parallelism
- Native infilling: just mask the positions you want filled
- Interruptible: natural checkpoint at every diffusion step

---

## Build System

```cmake
# CMakeLists.txt for dllm

cmake_minimum_required(VERSION 3.18)
project(dllm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# ggml source
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp/ggml)
add_subdirectory(${GGML_DIR} ggml_build)

# dllm library
add_library(dllm STATIC
    dllm_model.cpp
    dllm_graph.cpp
    dllm_sampler.cpp
    dllm_tokenizer.cpp
)
target_include_directories(dllm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${GGML_DIR}/include)
target_link_libraries(dllm PRIVATE ggml)

# CLI test executable
add_executable(dllm_cli dllm_main.cpp)
target_link_libraries(dllm_cli PRIVATE dllm ggml)
```

### Android NDK Build

```bash
# Build for arm64-v8a
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=28 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android --target dllm_cli

# Push and test
adb push build-android/dllm_cli /data/local/tmp/
adb push model.gguf /data/local/tmp/
adb shell /data/local/tmp/dllm_cli --model /data/local/tmp/model.gguf --prompt "Hello" --steps 32 --gen-length 64
```

---

## Roadmap

### Phase 1: MDLM Core (current)
- GGUF reader + Qwen3 weight mapping
- Bidirectional transformer forward pass
- MDLM sampling loop (Gumbel + confidence remasking)
- BPE tokenizer with MASK token
- CLI executable, ADB testing
- HF → GGUF conversion script

### Phase 2: BD3LM
- Block-causal attention mask
- KV cache for prefix blocks
- Block-level semi-autoregressive generation

### Phase 3: Optimizations
- Fast-dLLM prefix cache
- Confidence-aware parallel decoding
- Vulkan GPU backend
- ARM NEON-optimized Gumbel sampling

### Phase 4: Android SDK Integration
- JNI bridge in AiSystems
- Kotlin API (DLLMEngine)
- Integration with existing ToolNeuron UI

### Phase 5: Advanced Features
- Classifier-Free Guidance (CFG)
- EditFlow (insertion/deletion/substitution)
- Infilling mode
- Multiple remasking strategies (entropy, margin)
