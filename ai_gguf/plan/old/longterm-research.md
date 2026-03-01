# Long-Term Architecture Research — Findings & Implementation Decisions

## 1. Gated Residual (Dynamic Layer Weight) — IMPLEMENT

**Theory**: Modify `x_{l+1} = x_l + F_l(x_l)` to `x_{l+1} = x_l + alpha_l * F_l(x_l)`.
Per-layer scalar gate on BOTH attention and FFN residual connections.

**Key paper**: GateSkip (arxiv 2510.13876, Oct 2025) — 37.3% compute savings with 91.5% accuracy on Llama-3.2-3B.

**Why implement**:
- Memory: 256 bytes for 32 layers (2 floats per layer — attn_gate + ffn_gate)
- Latency: ~0.064ms/token for 32 layers — unmeasurable
- Flash attention: fully compatible (gate is outside the attention kernel)
- Personality: different layers encode different behaviors. Gate down middle layers
  (semantics/knowledge) and gate up late layers (style/personality) for stronger persona.
- ShortGPT (2024) proved up to 55% of layers are redundant in LLaMA.

**Where in forward pass**:
```
  for each layer:
    cur = attn(norm(inpL))
    cur = cur * attn_gate[il]        ← NEW: scale attention output
    inpL = inpL + cur                 (residual)
    cur = ffn(norm(inpL))
    cur = cur * ffn_gate[il]         ← NEW: scale FFN output
    inpL = inpL + cur                 (residual)
```

Gate applied AFTER build_attn/build_ffn returns, BEFORE the ggml_add residual.
NOT inside those functions. NOT after build_cvec.

**Safe gate ranges**: [0.0, 2.0]. Default 1.0 = no change.
0.0 = skip layer entirely. 2.0 = amplify. Typical steering range: 0.5–1.5.

**Implementation**: Same pattern as head_scales — per-layer float arrays in context,
passed via graph_params, no graph input class needed (just ggml_scale on existing tensor).
BUT: requires modifying each model's build function, not just build_ffn/build_attn.
Alternative: inject in the graph callback (cb_func) which sees every named tensor.

---

## 2. Sigmoid Attention — DO NOT IMPLEMENT (incompatible with pretrained models)

**Finding**: Sigmoid attention works well when TRAINED from scratch, but switching to sigmoid
at inference on a pretrained softmax model causes severe quality degradation.

**Key paper**: Apple "Sigmoid Attention is All You Need" (ICLR 2025, arxiv 2409.04431).
Formula: `sigma(QK^T/sqrt(d) - log(n)) @ V`. Matches softmax quality when trained from scratch.

**Why NOT implement**:
- Our models are pretrained with softmax — their weights expect competitive attention dynamics
- Switching at inference fundamentally changes what the weights compute
- Flash attention kernel is incompatible (online softmax has no sigmoid analog)
- No paper shows sigmoid working as a drop-in replacement on pretrained softmax models

**What could work instead** (but low priority):
- Post-attention sigmoid gate (Qwen's Gated Attention, G1-style): `Y' = softmax_attn(Q,K,V) * sigma(XW_gate)`
- This is essentially a more principled version of our existing head rescaling (Part D)
- Since we already have Part D, this adds little value

**Decision**: Skip. Our existing attention bias (Part C) + head rescaling (Part D) already
provide good attention control for pretrained models.

---

## 3. Cross-Layer Weight Sharing with Emotion Offset — PARTIALLY IMPLEMENT

**Finding**: Largely overlaps with existing Systems A (Control Vectors) and G (LayerNorm Offsets).

Key papers:
- ALBERT (Lan et al., 2020): Cross-layer parameter sharing
- Relaxed Recursive Transformers (2024, ICLR 2025): Shared + per-layer LoRA
- Transformer-Squared (Sakana AI, ICLR 2025): SVD + singular value scaling

**Overlap with existing systems**:
- Control vectors (A) = per-layer activation offsets — already IS the `Delta_l` concept
- LayerNorm offsets (G) = per-layer norm bias shifts — already IS a weight offset
- Hypernetwork (P4) = per-layer LoRA — already IS context-dependent weight modification

**What's genuinely NEW (implement these)**:

### 3a. Emotion-Conditioned Dimensional Gating — IMPLEMENT
Currently: `offset = alpha * base_direction` (scalar strength)
New: `offset = sigmoid(W_gate * emotion_vec) * base_direction` (dimensional gating)

This modulates WHICH dimensions of the offset are active based on the current
emotional state, not just the overall magnitude. "Joy" activates certain embedding
dimensions while "sadness" activates different ones.

Cost: one 6×n_embd matmul + sigmoid + elementwise multiply per layer.
For n_embd=896, 24 layers: ~142K FLOPs per token — negligible.

The W_gate matrix can be derived from existing contrastive activation data:
stack each emotion axis's direction vector as a row.

### 3b. Zone-Based Offset Regularization — IMPLEMENT (Kotlin-side only)
Average direction vectors within functional zones (early/mid/late) before applying.
Provides implicit regularization on small models where per-layer vectors are noisy.
Zero cost at inference (preprocessing step in ControlVectorManager).

---

## 4. Speculative Decoding from Own Layers — DONE

**Approach: Self-Speculative with Early Exit (SWIFT-inspired)**

Based on SWIFT (ICLR 2025) and LayerSkip (Meta, ACL 2024). Uses the model's
own early layers as draft model — zero extra memory, no retraining.

**How it works:**
1. `llama_set_early_exit_layer(ctx, E)` → graph builders loop over only E layers
2. Draft K tokens autoregressively through truncated model (fast — E/L fraction of compute)
3. `llama_reset_early_exit_layer(ctx)` → remove draft KV entries (partial layer data)
4. Batch-verify all K tokens through full model in one forward pass
5. Greedy argmax comparison: accept matching tokens, reject at first mismatch
6. Rejected position: use full model's argmax, single decode for fresh KV + logits
7. All accepted: bonus token from last full model logits

**Key insight:** `output_norm + lm_head` are applied AFTER the layer loop in all
model builders. So early exit naturally produces logits even from partial layers.

**Implementation:**
- `llama-graph.h`: `early_exit_layer` in `llm_graph_params`, checked in `allow_reuse`
- `llama-graph.cpp`: constructor uses `min(hparams.n_layer, early_exit_layer)` for `n_layer`
- `llama-context.h/cpp`: field + API (`llama_set_early_exit_layer`, `llama_reset_early_exit_layer`)
- `ai_gguf.cpp`: `speculative_generate()` — full loop with draft/verify/accept/stream
  - `nativeEnableSpeculativeDecode(exitLayer, numDraft)` / `nativeDisableSpeculativeDecode()`
  - Auto-integrates with `nativeGenerateStreamMultiTurn` (speculative path when enabled)
- `GGUFNativeLib.kt`: Kotlin external declarations

**Expected performance:**
- Draft at E=L/4: ~25% compute per draft token, 4-6 draft tokens per iteration
- Acceptance rate: ~60-80% (depends on model depth and task)
- Effective speedup: ~1.3-1.8x on mobile CPU (bandwidth-bound, batch amortizes weight loading)
- Memory overhead: zero (same model, same KV cache)

**Caveats:**
- Output is greedy (argmax from full model) — differs from sampled output when temperature > 0
- Stochastic acceptance (for temperature > 0) deferred to future enhancement
- Optimal exit_layer depends on model — recommend 25% of total layers as starting point

---

## 5. NPU/GPU/CPU Heterogeneous Scheduling — RESEARCH IN PROGRESS

**Reference**: `ai_gguf/plan/snapdragon-7s-gen3-hetero-arch.html` — full interactive guide.

**Key findings from Snapdragon 7s Gen 3 architecture analysis:**

### Hardware capabilities:
- **CPU** (Kryo, 1+3+4 cores, ARMv9/A720+A520): Branching logic, softmax, sampling,
  async learning (P7 SPSA), outlier ops. 128-bit NEON SIMD. ~50 GOPS INT8.
- **GPU** (Adreno 810, ~1 TFLOPS FP32): Dynamic-shape ops — attention scores,
  softmax over variable seq lengths, emotion vector injection, KV cache ops.
  Vulkan/OpenCL API. Supports dynamic shapes (unlike NPU).
- **NPU** (Hexagon, HVX+HMX): Fixed-shape GEMM — FFN weight×activation, Q/K/V
  projections, embedding lookup, LayerNorm. INT4/INT8/FP16. ~15 TOPS INT8.
  **Requires static computation graphs** compiled via QNN SDK.
- **Unified memory** (LPDDR5, 25.6 GB/s): Zero-copy between all processors.
  ~100ns sync (atomic fence), no PCIe overhead. This is the key enabler.

### Bandwidth bottleneck:
- Decode latency = model_size / bandwidth (hard physical limit)
- 1B INT4: ~20ms → 50 tok/s, 3B INT4: ~59ms → 17 tok/s, 7B INT4: ~137ms → 7 tok/s
- Quantization is the ONLY lever for decode speed

### Optimal per-operation assignment:
| Operation | Processor | Why |
|-----------|-----------|-----|
| FFN GEMM (up/gate/down) | NPU | Fixed shape, HMX INT4 = 10x GPU |
| Q/K/V projections | NPU | Fixed shape, bulk matmul |
| Attention scores | GPU | Dynamic [seq×seq], variable length |
| Softmax | GPU or CPU | Dynamic, branching |
| Embedding lookup | NPU | Static table lookup |
| LayerNorm / RMSNorm | NPU (HVX) | Fixed d_model, vector ops |
| Control vector injection | GPU | Dynamic runtime, additive |
| Emotion gating | GPU | Dynamic sigmoid + elementwise |
| Sampling (top-p/k) | CPU | Branching, small tensor |
| Fast weight update | CPU (async) | Outer product on efficiency cores |
| SPSA learning (P7) | CPU (async) | Async background on A520 cores |

### HeteroInfer strategy (SOSP 2025):
- GPU does attention for layer N while NPU does GEMM for layer N+1 (pipelined)
- 1.34-6x speedup over single-processor inference
- Sync cost: ~100ns per layer boundary (atomic write in shared DRAM)

### Speculative decoding synergy:
- Draft model (layers 0-3 of main model) on NPU: ~4ms for 4 draft tokens
- Verify all in one pass on full model: ~20ms
- ~70% accept rate → 2.8 tok / 24ms = ~116 tok/s effective (vs ~35 tok/s baseline)

### Implementation path (for our engine):
1. Already have: `GGML_BACKEND_DL=ON` → runtime CPU variant selection
2. Need: QNN SDK integration for NPU static graph compilation
3. Need: Vulkan backend activation for attention layers (already in llama.cpp)
4. Key: use `ggml_backend_tensor_set_backend()` to tag ops per processor
5. The GGML scheduler already handles multi-backend graph execution

### Cross-SoC portability challenge:
The Snapdragon 7s Gen 3 analysis above is just ONE SoC. Real deployment targets:
- **Qualcomm Snapdragon** (6xx, 7xx, 8xx series): Hexagon NPU + Adreno GPU + Kryo CPU
  - QNN SDK for NPU (proprietary, Qualcomm-specific)
  - INT4 support varies by generation (7s Gen 3+ only)
- **MediaTek Dimensity** (800, 9000 series): APU (AI Processing Unit) + Mali GPU + Cortex CPU
  - NeuroPilot SDK for NPU (MediaTek-specific)
  - Different GEMM tile sizes, different INT4 capabilities
- **Samsung Exynos** (2200, 2400): NPU + Xclipse GPU + Cortex CPU
  - Samsung ONE SDK for NPU
  - AMD RDNA2 GPU (different from Adreno/Mali)
- **Google Tensor** (G3, G4): Edge TPU + Mali GPU + Cortex CPU
  - TFLite delegate for TPU

**If we optimize per-SoC, we have to do it for ALL SoCs** — not just Snapdragon.
This means the architecture MUST be backend-agnostic:
1. Use ggml's backend abstraction (already handles multi-backend scheduling)
2. Tag operations by TYPE (fixed-shape GEMM, dynamic attention, scalar logic)
3. Let the runtime backend scheduler map types to available processors
4. NNAPI serves as universal fallback for NPU across all Android SoCs
5. Vulkan serves as universal GPU compute path

**Practical approach**: Don't target specific NPU SDKs initially. Instead:
- Phase 1: CPU + Vulkan GPU (current llama.cpp backends) — works everywhere
- Phase 2: NNAPI delegate for NPU ops — Android's universal NPU API
- Phase 3: QNN SDK for Qualcomm-specific optimizations (optional, highest perf)
This way we get broad device support first, then optimize for specific hardware.

**Decision**: HIGH PRIORITY for future — the unified memory architecture makes
this much more practical than desktop heterogeneous compute. But must be done
in a backend-agnostic way that works across ALL Android SoCs, not just Snapdragon.
Current CPU-only path works well for models under 3B. Heterogeneous becomes
essential for 7B+ models and for competing with native NPU inference engines.

---

## 6. Emotion State Machine — DONE

Full pipeline: probe model internals → blend with other signals → feedback → re-apply interventions.

### C++ Layer (ai_gguf.cpp):
- `nativeSetCaptureEnabled(bool)` — toggles per-layer activation storage (~86KB/token for 24-layer model)
- `nativeProbeEmotionAxes(cacheDir)` — reads cached direction vectors, computes
  `dot(activation, direction_vector) / n_embd` at layers [40%, 60%, 80%] depth with
  weights [0.2, 0.6, 0.2]. Tanh squashed to [-1, +1]. Returns JSON per axis.

### EmotionalStateTracker.kt (ToolNeuron):
- **4-tier detection**: Probe (50%) + Keyword (10%) + Embedding (25%) + LLM (15%)
- **EmotionRegime state machine**: NEUTRAL, WARMING, COOLING, EXCITED, VULNERABLE,
  PLAYFUL, TENSE, TRANSITIONING. Transition costs prevent oscillation (e.g.,
  WARMING→COOLING requires 3 consecutive signals).
- **Dual-timescale EMA**: Turn-level α=0.30, decay=0.90, velocity/momentum tracking
- **Feedback correction**: `error = target - actual`, correction = error × 0.3, clamped ±0.15.
  Negative feedback loop for stability.

### ControlVectorManager.kt — Orchestration:
- `enableEmotionProbing()` called from `applyPersonality()` — activates capture
- `onGenerationTurnComplete(response, userMessage)` — the full post-generation loop:
  1. Probe residual stream → Tier 0 EmotionalState
  2. Keyword analysis → Tier 1 EmotionalState
  3. `tracker.update(tier0, tier1)` → weighted blend + regime transition
  4. `tracker.computeFeedbackCorrection(persona)` → correction map
  5. `updateEmotionState(emotionMap + correction)` → re-apply with gating
  6. `learnFromResponse()` → P7 forward-only learning
- ChatViewModel hooks into both `simpleFlow()` and `agentFlow()` post-generation

---

## Implementation Priority

1. **Gated Residual** — DONE. Per-layer attn/FFN output scaling, 256 bytes, 0.064ms overhead.
2. **Emotion-Conditioned Dimensional Gating** — DONE. sigmoid(W_gate × emotion) × base_direction.
3. **Zone Regularization** — DONE. 50/50 blend toward zone mean, smooths small model noise.

Skip: Sigmoid attention (incompatible with pretrained models)

4. **Self-Speculative Decoding** — DONE. Early exit draft + full model verify, ~1.3-1.8x speedup.

5. **Emotion State Machine (Residual Probing)** — DONE. Full probe→update→feedback→apply pipeline.

### Next priorities:
6. **Heterogeneous NPU/GPU/CPU scheduling** — HIGH VALUE but requires QNN SDK integration.
   See snapdragon-7s-gen3-hetero-arch.html for full analysis. Key insight: NPU handles
   FFN GEMM (fixed shape, 10x faster), GPU handles attention (dynamic shape), CPU handles
   sampling/learning. Unified memory enables zero-copy between processors.
   Speculative decoding synergizes: draft on NPU, verify on full model.
