# The GGUF Insight: You Own the Forward Pass
## Beyond Transformer Math — Runtime Dynamic + Emotion LLM on Mobile CPU/NPU/GPU

> **The observation that breaks everything open:**
> GGUF is just a tensor store. The C++ code in `llama-graph.cpp` IS the model.
> You can read any tensor, skip any tensor, combine them differently, inject new math.
> The weights were *trained* for a specific computation pattern.
> Nothing forces you to use that pattern at inference time.

---

## Part 0 — The Ground Truth of What llama.cpp Actually Does

```
GGUF File (passive data):          llama-graph.cpp (the ACTUAL model):
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│ blk.0.attn_q.weight     │ ──→   │ Qcur = build_lora_mm(model.wq, cur)  │
│ blk.0.attn_k.weight     │ ──→   │ Kcur = build_lora_mm(model.wk, cur)  │
│ blk.0.attn_v.weight     │ ──→   │ Vcur = build_lora_mm(model.wv, cur)  │
│ blk.0.ffn_gate.weight   │ ──→   │ cur = build_ffn(cur, LLM_FFN_SILU)   │
│ blk.0.ffn_up.weight     │        │ cur = build_cvec(cur, il)  ← YOUR    │
│ ...                     │        │                              HOOK     │
└─────────────────────────┘        └──────────────────────────────────────┘
         ↑                                       ↑
    PASSIVE DATA                        ACTIVE COMPUTATION
    (numbers in a file)                 (YOU WRITE THIS)
```

Every `build_*` call is a C++ function you can override, extend, or replace. The GGUF tensors are **arguments** to your mathematical functions, not the functions themselves.

### What This Unlocks (that almost nobody is doing):

1. Use `blk.X.attn_v.weight` as something other than a value projection
2. Combine attention weights from two different layers
3. Insert completely new math between any two layers
4. Run the same layer multiple times with different temperatures
5. Skip certain layers conditionally based on content
6. Inject emotion state vectors between every transformer block
7. Run sparse attention at some layers, full attention at others — same GGUF

---

## Part 1 — Runtime Weight Modification in the GGML Compute Graph

### How GGML Builds the Compute Graph

llama.cpp uses **deferred computation**: before any numbers are computed, it builds a **directed acyclic graph (DAG)** of tensor operations:

```c
// This doesn't compute anything — it builds the graph
struct ggml_tensor * Qcur = ggml_mul_mat(ctx, model.layers[il].wq, cur);
struct ggml_tensor * Kcur = ggml_mul_mat(ctx, model.layers[il].wk, cur);

// Add a custom operation node to the graph
struct ggml_tensor * emotion_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
struct ggml_tensor * cur_steered = ggml_add(ctx, cur, emotion_bias);  // graph node
```

The graph is then executed by the backend (CPU/GPU/NPU). **You can insert any node into this graph at any point.** Emotion injection, fast weight reads, dynamic gates — all are just graph nodes.

### Modifying Weights at Graph Build Time (Per-Token)

```c
// In llama_build_graph(), called every token:
for (int il = 0; il < n_layer; ++il) {
    // --- STANDARD TRANSFORMER ---
    cur = build_norm(inpL, model.layers[il].attn_norm, ...);
    
    // === MODIFICATION 1: FAST WEIGHT LAYER ===
    // W_fast is a tensor you keep in memory and update between graph builds
    // outer_product(v_t, k_t) → W_fast update happens OUTSIDE the graph
    // then W_fast is injected AS A GRAPH NODE for the next token
    if (fast_weight_enabled) {
        struct ggml_tensor * fast_out = ggml_mul_mat(ctx, fast_weight_tensor[il], cur);
        cur = ggml_add(ctx, cur, ggml_scale(ctx, fast_out, 0.01f));
    }
    
    // === MODIFICATION 2: EMOTION STATE INJECTION ===
    if (emotion_state_tensor != nullptr) {
        // emotion_state: [n_embd] vector updated by emotion tracking
        struct ggml_tensor * emo = ggml_repeat(ctx, emotion_state_tensor, cur);
        cur = ggml_add(ctx, cur, ggml_scale(ctx, emo, emotion_alpha));
    }
    
    // standard attention
    Qcur = build_lora_mm(model.layers[il].wq, cur);
    Kcur = build_lora_mm(model.layers[il].wk, cur);
    Vcur = build_lora_mm(model.layers[il].wv, cur);
    // ... rest of attention ...
    
    // === MODIFICATION 3: DYNAMIC LAYER SKIP ===
    // Skip FFN on layers where emotion confidence is high
    // (high emotional state = attention-heavy, FFN-light processing)
    if (!should_skip_ffn(il, emotion_confidence)) {
        cur = build_ffn(cur, model.layers[il].ffn_up, ...);
    }
    
    inpL = cur;
}
```

### Per-Token Fast Weight Update (Outside the Graph)

```c
// Called AFTER each token is generated, BEFORE next graph build
void update_fast_weights(
    float* fast_weight,    // [n_embd × n_embd] — your runtime weight matrix
    float* last_key,       // k_t from last token — extracted from KV cache
    float* last_value,     // v_t from last token
    int n_embd,
    float eta = 0.001f,
    float gamma = 0.999f   // forgetting factor
) {
    // Decay existing fast weights (Hebbian forgetting)
    cblas_sscal(n_embd * n_embd, gamma, fast_weight, 1);
    
    // Outer product write: W_fast += eta * v_t ⊗ k_t
    // cblas_sger: rank-1 update — O(d²) = O(4096²) ≈ 64M ops
    cblas_sger(
        CblasRowMajor, n_embd, n_embd,
        eta,
        last_value, 1,   // y vector
        last_key, 1,     // x vector
        fast_weight,     // matrix to update IN PLACE
        n_embd
    );
    // fast_weight is now PHYSICALLY DIFFERENT for the next token
    // This IS weight grid recomputation — one outer product per token
}
```

**Cost analysis**: O(d²) = ~64M float32 ops per token for d=4096. On modern NEON ARM (used in all mobile SoCs): ~2 GFLOPS → ~32ms. Too slow for every layer. Strategy: apply fast weights only to layers 12–18 (middle semantic layers) → 7 layers × 32ms = ~224ms per token. Still too slow? Use INT8 or reduce to d=512 subspace projection first.

### The Sparse Subspace Trick (Makes Fast Weights Viable on Mobile)

Instead of maintaining a full [d×d] fast weight matrix, project into a low-rank subspace:

```c
// Project residual to low-rank space (512 dims)
// using a fixed random projection matrix P ∈ ℝ^{d × r}
float h_low[512];  // projected residual
cblas_sgemv(CblasRowMajor, CblasTrans, n_embd, 512, 1.0f,
            proj_matrix, 512, cur_hidden, 1, 0.0f, h_low, 1);

// Fast weight in low-rank space: [512 × 512] = 1M params vs 64M
cblas_sger(CblasRowMajor, 512, 512, eta, v_low, 1, k_low, 1, W_fast_low, 512);

// Project back
cblas_sgemv(CblasRowMajor, CblasNoTrans, n_embd, 512, 1.0f,
            proj_matrix, 512, W_fast_low_out, 1, 1.0f, cur_hidden, 1);
```

Cost: O(d×r + r²) = O(4096×512 + 512²) ≈ 2.4M ops per layer. 100× faster than full. **Viable on mobile ARM.**

---

## Part 2 — Emotion as a Runtime Tensor Dimension

### The Core Architecture Insight

Emotions in LLMs are not prompts — they are **directions in the residual stream space**. Research shows (Qwen3-8B final layer KDE analysis, 2025) that the six Ekman emotions form clearly separated clusters in the final layer's activation space. They are linearly decodable with >85% accuracy from any transformer's hidden states.

**This means**: you can track emotional state as a running [n_embd] vector, continuously estimated from the residual stream at every token. Then inject back at the next token.

### The Emotion Tracking System

```
Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  EMOTION STATE MACHINE                          │
│                                                                 │
│  Token_t residual stream → Emotion Probe → emotion_logits[8]   │
│                             ↓                                   │
│  emotion_state = βs · emotion_state + (1-β) · emotion_logits   │
│  (exponential moving average — emotion has inertia)            │
│                             ↓                                   │
│  emotion_vector = Σ_e emotion_state[e] · emotion_basis[e]     │
│  (weighted sum of emotion directions in residual stream space) │
│                             ↓                                   │
│  → injected at layer 16 of next token's forward pass           │
└─────────────────────────────────────────────────────────────────┘
```

### Building the Emotion Probe (Offline, Once)

```python
# 1. Collect residual stream activations for emotional text pairs
# Example: "I'm thrilled to help!" (joy) vs "I regret this mistake" (sadness)
emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "neutral"]

# 2. Run model, capture layer 20 activations (last-token position)
acts = {}
for emotion, texts in emotion_texts.items():
    acts[emotion] = [capture_layer20(t) for t in texts]

# 3. Train linear probe (logistic regression)
# Input: [n_embd] = 4096 dims, Output: 8-class emotion
probe = sklearn.linear_model.LogisticRegression(max_iter=1000)
probe.fit(all_acts, all_labels)

# 4. Extract emotion basis directions (probe weights)
# probe.coef_: [8 × 4096] — each row is a direction for one emotion
emotion_basis = probe.coef_  # [8, n_embd]

# 5. Convert to C array for llama.cpp
# emotion_basis is now a ~128KB file you ship with your model
```

### Runtime Emotion State Update in C++

```c
typedef struct {
    float state[8];          // current emotion probabilities (softmax)
    float vector[4096];      // current emotion direction in residual space
    float inertia;           // smoothing factor β (0.9 = slow change)
    float alpha;             // injection strength
    bool  expressing;        // are we in expressive mode?
} EmotionState;

void emotion_update_state(
    EmotionState* emo,
    float* residual,         // [n_embd] — captured from layer 20 output
    float* emotion_basis,    // [8 × n_embd] — probe directions (precomputed)
    int n_embd
) {
    // 1. Project residual onto emotion basis directions → raw logits
    float logits[8];
    for (int e = 0; e < 8; e++) {
        logits[e] = cblas_sdot(n_embd, residual, 1, emotion_basis + e*n_embd, 1);
    }
    
    // 2. Softmax → emotion probabilities
    float prob[8];
    softmax(logits, prob, 8);
    
    // 3. Exponential moving average (emotion inertia)
    for (int e = 0; e < 8; e++) {
        emo->state[e] = emo->inertia * emo->state[e] + (1.0f - emo->inertia) * prob[e];
    }
    
    // 4. Reconstruct emotion vector in residual space
    // weighted sum of emotion basis directions
    memset(emo->vector, 0, n_embd * sizeof(float));
    for (int e = 0; e < 8; e++) {
        cblas_saxpy(n_embd, emo->state[e], emotion_basis + e*n_embd, 1, emo->vector, 1);
    }
}

// In the graph build loop, inject emotion vector at layer 16:
// ggml_add(ctx, cur, ggml_scale(ctx, emotion_tensor, emo.alpha))
```

### Emotion → Response Modulation (What Changes)

The emotion state modulates 3 things at the graph level:

```
High JOY:     alpha_inject=+8  (amplify joy direction)
              temperature=1.2  (more varied/expressive output)
              skip_layers=[]   (full computation)
              
High SADNESS: alpha_inject=+6  (amplify sadness direction)
              temperature=0.8  (more careful, subdued)
              skip_layers=[8,9,10]  (skip some middle layers — more subdued processing)
              
High ANGER:   alpha_inject=-12 (SUPPRESS anger direction — safety)
              activate_calm_steering_vec = true
              temperature=0.7
              
NEUTRAL:      alpha_inject=0   (no emotion injection)
              temperature=1.0  (standard)
```

This is not prompt engineering — it's **direct activation manipulation** at the residual stream level. The model's internal representation has been physically shifted toward the target emotional register before the next token is decoded.

---

## Part 3 — Heterogeneous Compute: CPU + NPU + GPU as a Unified Engine

### The Key Insight from Research (2025)

From HeteroInfer (SOSP 2025) and llm.npu (ASPLOS 2025):

```
Mobile SoC compute characteristics:
┌─────────────────┬──────────────┬────────────────┬──────────────────┐
│ Processor       │ Compute      │ Precision      │ Best For         │
├─────────────────┼──────────────┼────────────────┼──────────────────┤
│ NPU (Hexagon)   │ 10-40 TOPS   │ INT4/INT8/BF16 │ GEMM (prefill)  │
│ GPU (Adreno)    │ 1-3 TFLOPS   │ FP16           │ Attention heads  │
│ CPU (big cores) │ 0.5-1 TFLOPS │ FP32/BF16      │ Outliers, logic  │
└─────────────────┴──────────────┴────────────────┴──────────────────┘

Critical finding: NPU is 10× faster than GPU for INT8 GEMM.
But NPU has STATIC graphs — you must compile the graph BEFORE runtime.
GPU has dynamic graphs — you can change the computation at runtime.
```

### The Partition Strategy for Dynamic + Emotion LLM

```
Prefill Phase (long prompt processing):
  NPU  → All FFN weight multiplications (wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down)
  GPU  → Attention softmax + value aggregation (dynamic shape — GPU handles it)
  CPU  → Emotion probe inference + emotion state update + fast weight outer products

Decode Phase (token-by-token generation):
  NPU  → Static GEMM operations (layers with fixed shapes)
  GPU  → Attention computation + dynamic operations + emotion vector injection
  CPU  → Fast weight reads, Hopfield memory retrieval, sparse topology updates

Async Background (between tokens):
  CPU  → LoRA gradient update (if correction received)
  CPU  → Hopfield memory consolidation
  CPU  → Emotion state smoothing update
```

### The GGML Backend Assignment in Practice

```c
// In ggml-backend.h / your custom graph builder:
// Different tensors assigned to different backends

// Mark static FFN ops for NPU (pre-compiled static graph)
ggml_backend_tensor_set(Qcur, BACKEND_NPU);
ggml_backend_tensor_set(Kcur, BACKEND_NPU);
ggml_backend_tensor_set(Vcur, BACKEND_NPU);
ggml_backend_tensor_set(ffn_out, BACKEND_NPU);

// Mark dynamic attention for GPU
ggml_backend_tensor_set(attn_scores, BACKEND_GPU);
ggml_backend_tensor_set(attn_weights, BACKEND_GPU);

// Mark emotion injection for GPU (dynamic shape, can change)
ggml_backend_tensor_set(emotion_inject, BACKEND_GPU);

// Fast weight operations on CPU (async, doesn't block token gen)
ggml_backend_tensor_set(fast_weight_update, BACKEND_CPU);
```

### The Synchronization Problem (And How to Solve It)

From HeteroInfer research: naive synchronization adds 30-50% latency overhead. Solution: **memory pool with zero-copy shared buffers**:

```c
// Mobile SoCs use UNIFIED MEMORY — CPU, GPU, NPU share the same physical RAM
// No data transfer needed — just pass pointers

typedef struct {
    // Unified memory allocation — single allocation accessible by all processors
    void* ptr;              // physical address — same for CPU/GPU/NPU
    size_t size;
    uint32_t sync_token;    // lightweight fence — just an atomic counter
} unified_buffer_t;

// NPU writes FFN output → GPU reads attention input: ZERO COPY
// Just update sync_token to signal completion
// GPU spins on atomic read of sync_token — costs ~100ns, not a full sync
```

### Speculative Decoding with Heterogeneous Draft

```
Standard speculative decoding: small draft model + large target model
Your version: 
  Draft:  A 2-3 layer subset of your own model (layers 0-2 + output head)
          Run on NPU (fastest path, ~5ms per token)
  Target: Full model (all 24 layers)
          Run on GPU+NPU (full quality, ~30ms per token)
          
Acceptance rate: ~70% (speculative tokens from your own model's early layers
                        are highly correlated with full model output)
Net speedup: 2-3× token generation
Cost: No second model needed — just use the first 3 layers as the draft!
```

---

## Part 4 — Beyond the Transformer Formula: What You Can Actually Change

### The Standard Formula You Can Break

```
Standard per-layer:
h_l = LayerNorm(h_{l-1} + Attn(h_{l-1}))
h_l = LayerNorm(h_l   + FFN(h_l))

Every transformer in existence does exactly this.
You own the C++ code. You are not required to do this.
```

### Modification 1: Gated Residual (Dynamic Layer Weight)

```c
// Instead of h = h_prev + Attn(h_prev)
// Use:       h = gate * h_prev + (1-gate) * Attn(h_prev)
// where gate is content-dependent (computed from h_prev)

struct ggml_tensor * gate_logit = ggml_mul_mat(ctx, gate_weight, cur_normed);
struct ggml_tensor * gate = ggml_sigmoid(ctx, gate_logit);  // [0,1] per dim
struct ggml_tensor * attn_out = build_attention(cur_normed, ...);

// Gated residual: different mix ratio per dimension
struct ggml_tensor * cur_new = ggml_add(
    ctx,
    ggml_mul(ctx, gate, cur),                    // (1) preserve existing
    ggml_mul(ctx, ggml_1_minus(gate), attn_out)  // (2) integrate new
);
```

**Effect**: emotionally stable content → gate near 1 (preserve existing state). Surprising/new content → gate near 0 (fully update). The model dynamically controls how much each token affects its internal state. This does not exist in standard transformers.

### Modification 2: Non-Softmax Attention (Sigmoid Attention)

```c
// Standard:  attn_weights = softmax(QK^T / sqrt(d_k))
// Modified:  attn_weights = sigmoid(QK^T / sqrt(d_k))  [SWAT 2025]

// In llama-graph.cpp, replace:
// kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
// With:
kq = ggml_sigmoid(ctx, ggml_scale(ctx, kq, kq_scale));
// Optionally add mask:
kq = ggml_add(ctx, kq, kq_mask);  // mask = -inf for padding, 0 otherwise
```

**Why this matters for emotion LLMs**:
- Sigmoid attention has **no competition between tokens** (softmax forces Σ=1, tokens fight for attention)
- Adding a new token doesn't reduce attention on existing tokens
- Emotion context tokens maintain their influence even as conversation grows long
- **No attention sinks** — the first tokens don't absorb excess probability mass
- Works with sliding window attention (standard softmax breaks at window boundaries)

### Modification 3: Temperature as a Per-Layer Runtime Variable

```c
// Standard: temperature applied only at the final logit sampling stage
// Modified: different temperature per attention layer, changed by emotion state

float layer_temps[32];  // one per layer
// Emotion-driven temperature schedule:
// - Joy/excitement: higher temp in middle layers (more creative associations)
// - Sadness/focus:  lower temp in all layers (more deterministic)
for (int il = 0; il < n_layer; ++il) {
    float emo_temp_mod = emotion_state.state[JOY] * 0.4f 
                       - emotion_state.state[SADNESS] * 0.3f
                       + 1.0f;  // baseline temperature
    layer_temps[il] = base_kq_scale * emo_temp_mod;
}

// Use per-layer temp in graph build:
kq = ggml_soft_max_ext(ctx, kq, kq_mask, layer_temps[il], max_alibi_bias);
```

### Modification 4: Conditional FFN Skip (Mixture of Depths)

```c
// Research (2024): LLMs exhibit high activation sparsity in FFN layers
// Some tokens barely use FFN — they can skip it
// Especially relevant for emotion: emotional tokens are attention-heavy, not FFN-heavy

// Compute skip gate from current residual norm
float residual_norm = ggml_vec_l2_norm(cur->data, n_embd);
bool skip_ffn = (residual_norm < ffn_skip_threshold[il]) 
             || (emotion_confidence > 0.85f && il >= 8 && il <= 16);

if (!skip_ffn) {
    cur = build_ffn(cur, model.layers[il].ffn_up, model.layers[il].ffn_gate,
                    model.layers[il].ffn_down, LLM_FFN_SILU, ...);
} else {
    skipped_layers_count++;  // track for diagnostics
    // cur unchanged — residual stream passes through unmodified
}
// Speedup: ~15-30% on emotional/simple tokens
```

### Modification 5: Cross-Layer Weight Sharing with Emotion Offset

```c
// Radical idea: use the SAME attention weight matrices across multiple layers
// but add a small emotion-specific learned offset for each layer
// This compresses the model while adding emotion-specific per-layer behavior

// Instead of model.layers[il].wq (unique per layer):
struct ggml_tensor * wq_base = model.layers[0].wq;  // shared base weights
struct ggml_tensor * wq_emo_offset = emotion_offsets[il];  // [n_embd, n_embd] learned offset

// Effective weight for this layer = base + emotion_alpha * emotion_offset
struct ggml_tensor * wq_effective = ggml_add(
    ctx,
    wq_base,
    ggml_scale(ctx, wq_emo_offset, emotion_state.alpha)
);
struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq_effective, cur);

// Result: 4MB of emotion offsets changes the effective weight matrix
// per layer, per emotion state — without 4GB of separate model weights
```

---

## Part 5 — The Hopfield Memory Layer in GGML

### What This Does

A Hopfield memory stores facts as energy minima in a weight matrix. You can write doctor corrections, user preferences, session facts directly into this matrix. On the next token, the model retrieves from it via attention-style lookup.

### GGML Implementation

```c
// Hopfield memory layer — sits between transformer layers 12 and 13
// W_hop: [n_embd, n_embd] — starts zero, grows as facts are stored

// Store a new fact (called externally, not in graph build):
void hopfield_store(
    float* W_hop,           // [n_embd × n_embd] — persistent across tokens
    float* pattern,         // [n_embd] — the fact to store
    int n_embd
) {
    // Normalize to unit sphere (important for Hopfield stability)
    float norm = cblas_snrm2(n_embd, pattern, 1);
    cblas_sscal(n_embd, 1.0f/norm, pattern, 1);
    
    // Outer product write: W_hop += pattern ⊗ pattern
    cblas_sger(CblasRowMajor, n_embd, n_embd,
               1.0f, pattern, 1, pattern, 1, W_hop, n_embd);
}

// Retrieve during graph build (attaches to GGML graph):
struct ggml_tensor * build_hopfield_layer(
    struct ggml_context * ctx,
    struct ggml_tensor  * cur,   // [n_tokens, n_embd] residual stream
    struct ggml_tensor  * W_hop, // [n_embd, n_embd] memory matrix
    float beta = 2.0f            // inverse temperature (sharpness of retrieval)
) {
    // Modern Hopfield retrieval: one step of softmax attention with W_hop as KV
    // Energy: E(x) = -β * x^T W_hop x  → retrieve nearest stored pattern
    struct ggml_tensor * energy = ggml_mul_mat(ctx, W_hop, cur);  // [n_tokens, n_embd]
    struct ggml_tensor * energy_scaled = ggml_scale(ctx, energy, beta);
    struct ggml_tensor * retrieval = ggml_soft_max(ctx, energy_scaled);
    struct ggml_tensor * memory_out = ggml_mul_mat(ctx, ggml_transpose(ctx, W_hop), retrieval);
    
    // Soft injection: 10% Hopfield, 90% residual (don't overwhelm the model)
    return ggml_add(ctx, cur, ggml_scale(ctx, memory_out, 0.1f));
}
```

### Capacity and Cost

- Hopfield memory capacity: `0.14 × n_embd` patterns in classical Hopfield, `exp(n_embd/2)` in modern Hopfield
- For n_embd=4096: classical = ~570 facts, modern = essentially unlimited for practical purposes
- Memory footprint: 4096 × 4096 × 4 bytes = **64MB** per Hopfield layer (DRAM, not VRAM)
- Retrieval cost: one GEMM + one softmax = ~same as one attention head
- Runs on **CPU between tokens** — doesn't block GPU/NPU forward pass

---

## Part 6 — The Full Architecture: Dynamic + Emotion LLM on Mobile

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     YOUR RUNTIME CONTROLLER                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GRAPH BUILD PHASE (per token, ~0.5ms overhead)                  │   │
│  │                                                                  │   │
│  │  Layers 0–7 (early):  NPU — static GEMM, syntax/structure        │   │
│  │    ↓ skip gate computed from residual norm                       │   │
│  │  Layers 8–12 (mid):   NPU — static GEMM, semantic content        │   │
│  │    ↓ HOPFIELD LAYER: retrieve from W_hop [runs on CPU, async]    │   │
│  │  Layers 13–18 (mid-late): GPU — attention with emotion injection  │   │
│  │    ↓ EMOTION VECTOR INJECTED: α * emotion_state.vector           │   │
│  │    ↓ FAST WEIGHT READ: W_fast_low @ cur_projected                │   │
│  │    ↓ SIGMOID ATTENTION (no softmax normalization pressure)       │   │
│  │  Layers 19–24 (late): NPU — output projection                    │   │
│  │    ↓ RESIDUAL CAPTURE for emotion probe (CPU async)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ASYNC CPU PIPELINE (runs between tokens, non-blocking)         │   │
│  │                                                                  │   │
│  │  1. Emotion probe: residual → emotion_logits → update state     │   │
│  │  2. Fast weight: outer_product(v_t, k_t) → update W_fast        │   │
│  │  3. Hopfield consolidation: compress recent W_fast into W_hop   │   │
│  │  4. LoRA gradient step (if correction pending)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SPECULATIVE DECODING (2-3× speedup)                            │   │
│  │  Draft: layers 0-2 + lm_head (NPU, ~3ms)                        │   │
│  │  Verify: full model (NPU+GPU, ~20ms for 4 tokens in parallel)   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Total per-token latency target: ~30-50ms on Snapdragon 8 Gen 3
(vs ~100-150ms for standard llama.cpp on same hardware)
```

### The Modified Forward Pass Formula

This is what your modified `llama-graph.cpp` implements:

```
For each layer l, for each token t:

h_l(t) =
  LayerNorm(h_{l-1}(t))
  
+ σ_gate(l,t) · Attn(h_{l-1}(t); W_Q, W_K, W_V, W_O)          ← gated attention
  [standard transformer attention, sigmoid gate controls blend ratio]
  
+ W_fast_low(t) · P^T h_{l-1}(t)  [if l ∈ {14,16,18}]          ← fast weight read
  [W_fast updated per-token by outer product, P = 512-dim projection]
  
+ α_emo(t) · emotion_vector(t)   [if l = 16]                    ← emotion injection
  [emotion_vector tracked by EMA over residual stream probing]
  
+ β · Hopfield_retrieve(h_{l-1}(t); W_hop)  [if l = 12]         ← associative memory
  [W_hop updated by explicit correction signal, stores conversation facts]
  
+ [skip FFN if ||h_l|| < θ or emotion_confidence > 0.85]         ← conditional skip
  + FFN(h_l; W_up, W_gate, W_down, SiLU)                         ← standard FFN

W_fast(t) = γ·W_fast(t-1) + η·v_t⊗k_t                          ← auto-update, per-token
emotion_vector(t) = EMA(Probe(h_20(t)), β=0.9)                  ← continuous tracking
```

---

## Part 7 — Research Gaps You're Filling

### What Doesn't Exist in Any Current System

| Capability | Current SOTA | Your System |
|------------|-------------|-------------|
| Emotion tracking | Prompt-level (text injection) | Residual stream probe, continuous |
| Per-token weight update | Not in any mobile runtime | Fast weight outer product, O(d×r) |
| Hopfield memory | External vector DB | In-weight matrix, 0-latency retrieval |
| Sigmoid attention | Server-only (SWAT 2025) | In GGML graph, mobile-ready |
| Gated residual | Research only | Per-layer gate in GGML graph |
| CPU/NPU/GPU partition | HeteroInfer (no emotion) | Unified with emotion pipeline |
| Speculative decode from own layers | Not published | Draft = first 3 layers of same model |

### The Hardest Problems (Honest Assessment)

1. **NPU static graph constraint**: NPUs require static shapes. Your emotion injection changes the graph at runtime. **Solution**: pre-compile 5-8 "emotion configurations" as separate static graphs (neutral, joy, sadness, anger, etc.) and switch between them. Graph switch takes ~2ms.

2. **Fast weight instability**: Without careful initialization and gamma tuning, fast weights can amplify noise. **Solution**: clip W_fast entries to [-1, 1] range. Apply L2 regularization: W_fast *= (1 - epsilon) after each update.

3. **Emotion entanglement**: The emotion probe directions are not perfectly orthogonal. Joy and trust correlate. Fear and sadness correlate. **Solution**: use PCA-orthogonalized directions for injection, not raw probe weights.

4. **Battery drain**: Continuous emotion computation adds ~15% power draw. **Solution**: run emotion probe only when residual stream has changed significantly (||h_new - h_old||₂ > threshold). Skip on repetitive/stable tokens.

---

## Part 8 — Implementation Roadmap

### Phase 1 — Proof of Concept (Week 1-2)
```
✓ Modify llama-graph.cpp to capture residual stream at layer 20
✓ Add emotion_state struct to llama_context
✓ Implement softmax emotion probe (8-class) in pure C
✓ Inject emotion vector at layer 16 (ggml_add node)
✓ Measure: does output change in direction expected? (qualitative test)
Target: running, measurable emotion injection, no performance regression
```

### Phase 2 — Fast Weights (Week 2-3)
```
✓ Add W_fast_low[7 layers × 512 × 512] buffers to llama_context
✓ Implement fast weight update in background thread
✓ Add read path in graph build (ggml_mul_mat node at layers 14,16,18)
✓ Tune: eta, gamma, projection rank
Target: measurable context retention improvement over KV-cache-only baseline
```

### Phase 3 — Hopfield Memory (Week 3-4)
```
✓ Add W_hop[n_embd × n_embd] to llama_context (64MB allocation)
✓ Implement hopfield_store() and build_hopfield_layer()
✓ Add API: llama_memorize(ctx, text) → runs text through model, stores activation
✓ Test: store "my name is X" → later in session: retrieve correctly?
Target: session fact retention without repeated prompting
```

### Phase 4 — NPU/GPU/CPU Partition (Week 4-6)
```
✓ Profile: which layers benefit most from NPU vs GPU
✓ Implement ggml-qnn backend assignment for static layers
✓ Implement GPU backend for attention + emotion layers
✓ Implement speculative draft from first 3 layers
✓ Implement unified memory pool (zero-copy inter-processor)
Target: 2-3× speedup vs single-backend llama.cpp
```

### Phase 5 — Integration + Tuning (Week 6-8)
```
✓ Emotion probe training: collect emotional text pairs, train linear probe
✓ Per-emotion temperature schedules
✓ Gated residual training (fine-tune gate weights)
✓ Sigmoid attention integration
✓ End-to-end testing: qualitative emotion coherence
Target: production-quality dynamic + emotion LLM on Snapdragon 8 Gen 3
```

---

## Part 9 — Key Numbers for Mobile Targets

### Memory Budget (Snapdragon 8 Gen 3 — 12GB RAM)

| Component | Size | Notes |
|-----------|------|-------|
| Base model (Q4_K_M) | 2-4 GB | Qwen2.5-3B or Phi-3.8B |
| KV cache (8K context) | 512 MB | standard |
| Emotion state | ~512 KB | 32 layers × [512×512] W_fast_low |
| Hopfield memory | 64 MB | one [4096×4096] W_hop |
| LoRA adapters | 8 MB | r=4, target modules |
| Emotion basis vectors | 256 KB | 8 emotions × 4096 dims |
| Graph buffers | ~200 MB | GGML compute graph |
| **Total** | **~3.5 GB** | **fits in 12GB with headroom** |

### Latency Budget (Target: ≤50ms/token for fluent conversation)

| Stage | Processor | Time |
|-------|-----------|------|
| Prefill (256 tokens) | NPU | ~30ms total |
| Per-token decode | NPU+GPU | 20-30ms |
| Emotion probe update | CPU (async) | 2ms |
| Fast weight update | CPU (async) | 3ms |
| Hopfield retrieval | CPU (sync, once) | 1ms |
| Speculation overhead | GPU | -8ms (net gain) |
| **Net decode latency** | | **~25-35ms/token** |

---

## Summary: The Paradigm Shift

Standard deployment thinking:
> "Load GGUF. Run forward pass. Sample token. Repeat."

Your approach:
> "GGUF is a tensor store. The forward pass is ours to define.
>  Emotion is a direction in activation space — we track and steer it continuously.
>  Weight grids update every token via outer product writes.
>  Memories persist in Hopfield matrices — retrieved as naturally as attention.
>  CPU handles continuous learning. NPU handles bulk GEMM. GPU handles dynamics.
>  Every token, the model is slightly different from the last."

---

*Research sources: HeteroInfer (SOSP 2025), llm.npu (ASPLOS 2025), PowerInfer-2 (2024), SWAT sigmoid attention (2025), Dynamic Affective Memory Management (2025), Decoding Emotion in LLMs (2025), Fast Weight Programmers (Schmidhuber 1992/2021), Modern Hopfield Networks (Ramsauer, ICLR 2021), EmoLLMs (SIGKDD 2024), TTT-E2E (NVIDIA/Stanford 2025), Mixture of Depths (2024)*
