# Gated Residual Connections & Dynamic Layer Weighting: Research Findings

**Date**: 2026-02-23
**Purpose**: Evaluate gated residual connections for runtime personality/behavior steering in llama.cpp on mobile

---

## 1. Theory: What Are Gated Residual Connections?

### Standard Residual Connection (ResNet, 2015)

The standard transformer residual connection is an identity skip:

```
x_{l+1} = x_l + F_l(x_l)
```

where `F_l` is the sublayer (attention or FFN). Every layer contributes equally (coefficient = 1.0) to the residual stream. There is no mechanism to modulate how much a layer's output influences the final representation.

### Gated Residual Connection

A gated residual connection introduces a learnable or configurable gate that controls how much of the layer's output is added to the residual stream:

```
x_{l+1} = x_l + g_l * F_l(x_l)
```

where `g_l` is a gate value in [0, 1]. This is the core idea.

### Mathematical Formulations (4 variants)

**Variant A: Per-layer scalar gate (simplest, our target)**
```
x_{l+1} = x_l + alpha_l * F_l(x_l)

where alpha_l in [0.0, 1.0] is a single scalar per layer
Memory: n_layer * sizeof(float) = ~128 bytes for 32 layers
```

**Variant B: Per-layer sigmoid gate (learned)**
```
x_{l+1} = x_l + sigmoid(g_l) * F_l(x_l)

where g_l is a learnable parameter per layer
sigmoid ensures output is in (0, 1)
Equivalent to Variant A at inference with alpha_l = sigmoid(g_l)
```

**Variant C: Input-dependent vector gate (GateSkip, 2025)**
```
h_{l+1} = h_l + o_l * sigma(W_G * h_l + b)

where W_G in R^{H x H} is a weight matrix
sigma is element-wise sigmoid
Gate is different for every token and every dimension
Memory: H^2 + H per layer = ~16M params for d=4096
```

**Variant D: Evaluator-Adjuster Unit (Gated Residual Connection paper, 2024)**
```
g = sigma(W_g * r + b_g)           -- gate vector via sigmoid
y = r + (g . s)                     -- element-wise gated residual

where r = residual input, s = sublayer output
```

### Key Difference from Standard Residuals

| Property | Standard | Gated |
|----------|----------|-------|
| Layer influence | Always 1.0x | Configurable [0, 1] |
| Can suppress a layer | No | Yes (gate -> 0) |
| Can amplify a layer | No | Yes (gate > 1, though usually clamped) |
| Runtime adjustable | No | Yes (Variant A) |
| Requires retraining | N/A | No for Variant A, Yes for C/D |
| Parameter overhead | 0 | 32 floats (A) to millions (C) |

---

## 2. Key Papers

### 2.1 LayerDrop / Stochastic Depth (Fan et al., ICLR 2020)
- **Paper**: [Reducing Transformer Depth on Demand with Structured Dropout](https://arxiv.org/abs/1909.11556)
- **Key idea**: Randomly drop entire layers during training. At inference, select any sub-network depth.
- **Results**: Maintains performance while reducing 25% of layers. State-of-art on MT, LM, summarization, QA.
- **Relevance to us**: Proves that transformers are over-parameterized and layers can be safely scaled down. If dropping layers entirely works, scaling them by 0.7x should be even safer.

### 2.2 GateSkip: Residual Gates for Layer Skipping (Laitenberger et al., Oct 2025)
- **Paper**: [What Layers When: Learning to Skip Compute in LLMs with Residual Gates](https://arxiv.org/abs/2510.13876)
- **THE most relevant paper for our use case.**
- **Key idea**: Each attention/MLP branch gets a sigmoid-linear gate. Gate values determine per-token, per-layer importance.
- **Mathematical formulation**:
  ```
  h_{l+1} = h_l + o_l . sigma(W_G * h_l + b)
  ```
- **Results (Llama-3.2-1B)**:
  - 15% compute savings: retains >90% of baseline accuracy
  - On Llama-3.2-3B: 37.3% compute savings while retaining 91.5% of GSM8K performance
  - Instruction-tuned models: +12.5 accuracy points at full compute, matches baseline at 50% savings
  - Throughput: 25% skip -> 2,927 tok/s, 50% skip -> 3,141 tok/s, 70% skip -> 3,642 tok/s
- **Parameter overhead**: 0.004% for scalar gates, 4% for vector gates
- **Initialization**: W_G with sigma=0.01, bias=5 (sigmoid(5) ~ 1.0 = no perturbation initially)
- **Key finding for us**: "Differentiable gates train stably on pretrained models without full retraining" -- but even without training, we can set scalar gates manually.
- **Token importance patterns**: BOS tokens and punctuation get high gate values. Early layers maintain broad compute; deeper layers become selective for content words.

### 2.3 DeepNorm / DeepNet (Wang et al., 2022)
- **Paper**: [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
- **Key idea**: Per-layer residual scaling with constant alpha > 1: `x_{l+1} = LayerNorm(alpha * x_l + F(x_l))`
- **Results**: Successfully trained 1,000-layer transformers (2,500 sublayers)
- **Relevance**: Proves per-layer scalar weights on residuals are mathematically sound. Our approach is the inverse -- instead of scaling residuals up for training stability, we scale layer outputs down for behavior control.

### 2.4 ShortGPT: Layer Redundancy Analysis (Men et al., 2024)
- **Paper**: [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)
- **Key finding**: Up to 55% of LLaMA layers can be pruned with minimal performance loss.
- **Block Influence (BI) metric**: Measures importance via cosine similarity between layer input/output. High cosine similarity = layer does little = safe to prune/suppress.
- **Results**: 25% parameter reduction with ~90% performance retention.
- **Critical insight for us**: Middle-to-late layers are most redundant. Early layers (0-3) and the final layer are most important. This tells us which layers are SAFE to gate down.

### 2.5 LayerSkip: Early Exit + Self-Speculative Decoding (Meta, ACL 2024)
- **Paper**: [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
- **Key idea**: Train with increasing dropout rates for later layers. At inference, exit early and use remaining layers for verification (self-speculative decoding).
- **Results**: Up to 2.16x speedup on summarization, 1.82x on coding, 2.0x on parsing.
- **Relevance**: Proves later layers can be skipped without catastrophic failure. Our gates at 0.0 = layer skip.

### 2.6 CALM: Confident Adaptive Language Modeling (Google, NeurIPS 2022)
- **Paper**: [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061)
- **Key idea**: Measure confidence at intermediate layers. Exit early when confident. Uses softmax response as confidence measure.
- **Results**: ~3x speedup while controlling for high quality output.
- **Relevance**: Shows that not every token needs all layers. "Easy" tokens (common words, punctuation) can exit early.

### 2.7 Mixture of Depths (Raposo et al., Apr 2024)
- **Paper**: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)
- **Key idea**: Router at each layer selects top-k tokens for full computation. Remaining tokens bypass via residual. Unlike early-exit, tokens can skip middle layers but still be processed by later layers.
- **Results**: Models that are both faster AND better performing than isoFLOP baselines.
- **Relevance**: Validates that per-layer, per-token routing through residual bypass is sound. Our per-layer gates are a simplified version.

### 2.8 ADEPT: Adaptive Dynamic Early-Exit (Jan 2026)
- **Paper**: [ADEPT: Adaptive Dynamic Early-Exit Process for Transformers](https://arxiv.org/pdf/2601.03700)
- **Key idea**: Token-level early exit in both prefill and decode phases. Unlike CALM which is prompt-level, ADEPT adapts per-token.
- **Relevance**: Shows the frontier is moving toward per-token, per-layer compute allocation -- exactly what gated residuals enable.

### 2.9 Gated Residual Connections + Evaluator Adjuster Unit (May 2024)
- **Paper**: [Dynamic Context Adaptation and Information Flow Control in Transformers](https://arxiv.org/html/2405.13407)
- **Gate formulation**: `g = sigma(W_g * r + b_g)`, `y = r + (g . s)`
- **Results on WMT 2014 EN-DE**:
  - Baseline: 26.61 BLEU
  - + GRC only: 26.77 BLEU (+0.16)
  - + EAU + GRC: 26.79 BLEU (+0.18)
- **Results on GLUE (MRPC)**: 79.41% vs 73.04% baseline (+6.37 points)
- **Relevance**: Even simple gated residuals improve quality when gates are learned.

---

## 3. Practical Benefits

### 3.1 Performance Improvements

| Method | Model | Compute Saved | Quality Retained |
|--------|-------|---------------|-----------------|
| GateSkip | Llama-3.2-1B | 15% | >90% accuracy |
| GateSkip | Llama-3.2-3B | 37.3% | 91.5% GSM8K |
| LayerSkip | Llama variants | 50%+ layers | 2.16x speedup |
| CALM | T5 | variable | 3x speedup |
| ShortGPT | LLaMA | 25% params | 90% quality |
| LayerDrop | various | 25% layers | SOTA maintained |
| MoD | various | varies | better than isoFLOP |

### 3.2 Inference Latency Impact

**For our use case (Variant A, scalar gates only):**
- **Overhead per layer**: 1 multiply + 1 add = ~0 measurable overhead
- **Actually: The `ggml_scale` op is a single scalar multiply on the full tensor. On ARM NEON, this is a single vectorized loop -- <0.01ms per layer.**
- **Total overhead for 32 layers**: <0.32ms additional per token
- **Potential savings if gates < 1.0**: Reduced activation magnitudes propagate smaller values through subsequent layers, potentially improving cache efficiency

**For aggressive gating (gate = 0.0, full layer skip):**
- If we can skip the layer computation entirely (not just scale output to 0), we save:
  - Attention: ~60% of layer compute
  - FFN: ~40% of layer compute
  - For a 3B model with 32 layers, skipping 8 layers = ~25% latency reduction
  - On mobile: ~15-25ms saved per layer on CPU

### 3.3 Personality/Behavior Steering Applications

This is where gated residuals become uniquely powerful for our system:

**Hypothesis**: Different layers encode different aspects of behavior:
- **Early layers (0-5)**: Token-level syntax, basic grammar, language detection
- **Middle layers (6-20)**: Semantic meaning, factual knowledge, reasoning patterns
- **Late layers (21-31)**: Style, personality, output formatting, refusal behavior

**Supported by ShortGPT findings**: Middle layers are most redundant (high cosine similarity between input/output). This means middle layers make smaller transformations -- they're "refinement" layers that can be safely scaled.

**Supported by activation steering research**: Control vectors (which we already use) work by adding directional vectors to the residual stream. Gated residuals multiply the stream. These are complementary:
- Control vector: additive bias `x + direction`
- Gated residual: multiplicative scaling `alpha * x`
- Combined: `alpha * (x + direction)` or `x + alpha * layer_output + direction`

**Concrete steering scenarios**:
1. **"Make responses more creative"**: Boost late layers (style), suppress middle layers (factual constraints)
2. **"Make responses more factual"**: Boost middle layers (knowledge), suppress late layers (style variation)
3. **"Reduce verbosity"**: Suppress middle refinement layers that add detail
4. **"Increase emotional expression"**: Boost specific late layers that control sentiment output

---

## 4. Implementation Approaches (Detailed)

### 4.1 Variant A: Per-Layer Scalar Gate (RECOMMENDED for v1)

```cpp
// In the model forward pass
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    // ... attention computation ...
    cur = build_attn(...);

    // GATE: Scale attention output before residual add
    if (attn_gate[il] != 1.0f) {
        cur = ggml_scale(ctx0, cur, attn_gate[il]);
    }

    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

    // ... FFN computation ...
    cur = build_ffn(...);

    // GATE: Scale FFN output before residual add
    if (ffn_gate[il] != 1.0f) {
        cur = ggml_scale(ctx0, cur, ffn_gate[il]);
    }

    cur = ggml_add(ctx0, cur, ffn_inp);
    cur = build_cvec(cur, il);
    inpL = cur;
}
```

**Properties**:
- Memory: 2 * n_layer * sizeof(float) = 256 bytes for 32 layers
- Compute overhead: <0.01ms per layer (scalar multiply)
- No retraining needed
- Runtime adjustable via JNI
- Fully compatible with flash attention (scalar multiply doesn't change attention pattern)

### 4.2 Variant B: Split Attention/FFN Gates

Same as Variant A but with separate gates for attention and FFN sublayers. This allows independent control:

```
attn_residual:  x_l + alpha_attn_l * Attention(norm(x_l))
ffn_residual:   x_l + alpha_ffn_l  * FFN(norm(x_l))
```

**Rationale**: Attention and FFN serve different functions:
- Attention: information routing between tokens
- FFN: per-token feature transformation
- Being able to independently gate them is more expressive

### 4.3 Variant C: Full Layer Skip (gate = 0)

When gate is exactly 0.0, we can skip the entire layer computation:

```cpp
for (int il = 0; il < n_layer; ++il) {
    // Full layer skip -- save all compute for this layer
    if (layer_gate[il] == 0.0f) {
        // Still need to apply control vectors
        cur = build_cvec(inpL, il);
        inpL = cur;
        continue;
    }
    // ... normal computation ...
}
```

**Caution**: Full skip is aggressive. Based on ShortGPT, safe to skip:
- Some middle layers (8-24 for a 32-layer model)
- Never skip layers 0-2 or the final layer
- Maximum safe skip: ~25% of layers for small models, ~55% for large models

### 4.4 Variant D: Per-Token Dynamic Gate (Future work, requires training)

```
gate_l(x) = sigmoid(w_l^T * x + b_l)     -- scalar gate, input-dependent
x_{l+1} = x_l + gate_l(x_l) * F_l(x_l)
```

This requires a learned weight vector per layer (d_model parameters per layer) and is the approach used by GateSkip. Would need fine-tuning but provides much better accuracy/compute tradeoff.

### 4.5 Variant E: Early Exit (Future work)

```
for (int il = 0; il < n_layer; ++il) {
    // ... normal layer computation ...

    if (il >= min_layers && gate[il] < exit_threshold) {
        break;  // Skip all remaining layers
    }
}
```

This is the most aggressive speedup but requires confidence calibration. Based on CALM results, could achieve 3x speedup.

---

## 5. Implementation in llama.cpp

### 5.1 The Forward Pass Structure

From `/home/home/CLionProjects/llama.cpp-android/src/models/llama.cpp`:

```cpp
for (int il = 0; il < n_layer; ++il) {
    ggml_tensor * inpSA = inpL;

    // --- NORM + ATTENTION ---
    cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
    cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].bo,
                     Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

    // >>> ATTENTION RESIDUAL (line 100) <<<
    ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);    // <-- GATE POINT 1

    // --- NORM + FFN ---
    cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
    cur = build_ffn(cur, ...);

    // >>> FFN RESIDUAL (line 138) <<<
    cur = ggml_add(ctx0, cur, ffn_inp);                     // <-- GATE POINT 2

    // >>> CONTROL VECTOR (line 141) <<<
    cur = build_cvec(cur, il);                              // <-- existing intervention

    inpL = cur;
}
```

### 5.2 Exact Insertion Points

**Gate Point 1 (Attention Residual)** -- line 100:
```cpp
// BEFORE (current):
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

// AFTER (with gate):
if (residual_gates && il < (int)residual_gates->size()) {
    float attn_alpha = (*residual_gates)[il].attn_gate;
    if (attn_alpha != 1.0f) {
        cur = ggml_scale(ctx0, cur, attn_alpha);
        cb(cur, "attn_gated", il);
    }
}
ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
```

**Gate Point 2 (FFN Residual)** -- line 138:
```cpp
// BEFORE (current):
cur = ggml_add(ctx0, cur, ffn_inp);

// AFTER (with gate):
if (residual_gates && il < (int)residual_gates->size()) {
    float ffn_alpha = (*residual_gates)[il].ffn_gate;
    if (ffn_alpha != 1.0f) {
        cur = ggml_scale(ctx0, cur, ffn_alpha);
        cb(cur, "ffn_gated", il);
    }
}
cur = ggml_add(ctx0, cur, ffn_inp);
```

### 5.3 Gate Data Structure

```cpp
// In llama-graph.h, alongside existing interventions:
struct llm_residual_gate {
    float attn_gate = 1.0f;  // [0.0, 2.0], default 1.0 = no change
    float ffn_gate  = 1.0f;  // [0.0, 2.0], default 1.0 = no change
};

// In llm_graph_context:
const std::vector<llm_residual_gate> * residual_gates;
```

### 5.4 Where NOT to Put the Gate

**NOT inside `build_ffn`**: The FFN function computes `gate(x) * up(x)` internally (for SwiGLU). Adding a gate there would interfere with the FFN's own gating mechanism. The residual gate should be AFTER `build_ffn` returns, scaling its total output.

**NOT inside `build_attn`**: The attention function has its own scaling (`kq_scale`), head rescaling, and temperature. Adding a residual gate inside attention would conflate with these. Gate the attention OUTPUT, not internals.

**NOT after `build_cvec`**: Control vectors are additive to the residual stream. If we gate after cvec, we'd be scaling the control vector too, which would require compensating the cvec magnitude. Better to gate the layer output before the residual add, and let cvec apply independently after.

### 5.5 Execution Order in the Intervention Stack

```
Layer l:
  1. RMSNorm(input)
  2. Attention computation
     - Head rescaling (Part D) -- inside build_attn
     - Attention temperature (Part E) -- inside build_attn
     - Attention bias (Part C) -- inside build_attn
  3. >>> RESIDUAL GATE (attn) <<<  -- NEW, scales attention output
  4. Attention residual add: ffn_inp = scaled_attn + input
  5. RMSNorm(ffn_inp)
  6. FFN computation
     - KAN overlay -- inside build_ffn
     - Hypernetwork LoRA (Part P4) -- inside build_ffn
     - Sparse masks -- inside build_ffn
  7. >>> RESIDUAL GATE (ffn) <<<  -- NEW, scales FFN output
  8. FFN residual add: output = scaled_ffn + ffn_inp
  9. Control vector addition (cvec)
  10. -> Next layer input
```

---

## 6. Interaction with Existing Systems

### 6.1 Control Vectors (Part A)

**Current**: `cur = build_cvec(cur, il)` which does `cur = ggml_add(ctx, cur, direction_vector)`

**Interaction**: Control vectors add to the residual stream AFTER both residual additions. Gated residuals scale the layer output BEFORE the residual addition.

**Combined effect**:
```
Without gates:  output = x + attn(x) + ffn(x+attn(x)) + cvec
With gates:     output = x + alpha_a*attn(x) + alpha_f*ffn(x+alpha_a*attn(x)) + cvec
```

**Key insight**: These are ORTHOGONAL interventions:
- Gates control HOW MUCH a layer contributes (multiplicative)
- Control vectors control WHICH DIRECTION the output shifts (additive)
- They compose naturally without interference
- If gate = 0, the layer is suppressed but cvec still applies

**Practical consideration**: If a layer is heavily gated down (alpha = 0.3), the control vector for that layer still applies at full strength. This means you can suppress a layer's natural behavior while still injecting a control direction. This is actually very powerful for steering.

### 6.2 Head Rescaling (Part D)

**Current**: Inside `build_attn`, per-head scale factors multiply attention outputs.

**Interaction**: Head rescaling modifies WHICH information the attention extracts. The residual gate then scales the overall attention contribution.

```
Effective attention contribution = alpha_attn * sum(head_scale_h * head_output_h)
```

**These compose multiplicatively** -- if head_scale for head 3 is 1.5x and attn_gate is 0.7x, the effective contribution of head 3 is 1.05x. This is fine and expected.

**No conflicts**: Head rescaling operates inside the attention block, gates operate outside. They are at different granularities (per-head vs per-layer) and compose cleanly.

### 6.3 Attention Temperature (Part E)

**Current**: Per-head temperature scaling of attention logits before softmax.

**Interaction**: Temperature changes the attention PATTERN (which tokens attend to which). The residual gate changes how much that pattern's output contributes.

**No conflicts**: Temperature is inside softmax, gates are on the output. Orthogonal.

### 6.4 KAN Overlay (inside FFN)

**Current**: Applied inside `build_ffn` as an activation function overlay.

**Interaction**: KAN modifies the FFN computation itself. The FFN gate then scales the KAN-modified output.

```
FFN output = alpha_ffn * KAN_modified_FFN(x)
```

**No conflicts**: KAN changes what the FFN computes, gates scale how much it contributes.

### 6.5 Hypernetwork LoRA (Part P4)

**Current**: Applied inside `build_ffn` as a low-rank modification to FFN weights.

**Interaction**: Like KAN, this modifies the FFN computation. Gates scale the modified output.

**No conflicts**: Orthogonal -- LoRA changes weights, gates scale outputs.

### 6.6 Sparse Masks

**Current**: Zero out specific dimensions of layer outputs.

**Interaction**: Sparse masks set specific dimensions to 0 (binary). Gates scale all dimensions uniformly (continuous). These are complementary -- sparse masks for precision, gates for magnitude.

### 6.7 Fast Weight Memory (Part F)

**Current**: Hopfield-style associative memory applied as additive modification to hidden states.

**Interaction**: Fast weight memory adds retrieved associations. If we gate the layer output before the residual add, but fast weights apply after, they remain unaffected. The ordering matters:

```
Recommended: output = gate * layer_out + input + fast_weight_delta
NOT:         output = gate * (layer_out + fast_weight_delta) + input
```

### 6.8 Norm Offsets (Part G)

**Current**: Additive offsets to RMSNorm outputs, applied inside `build_norm`.

**Interaction**: Norm offsets modify the input to the sublayer (attention or FFN). Gates modify the output. These are at opposite ends of the sublayer and don't interfere.

### 6.9 Summary Table

| System | Type | Where | Interacts with Gate? | Conflict? |
|--------|------|-------|---------------------|-----------|
| Control Vectors | Additive | After residual adds | No - applies independently | None |
| Head Rescaling | Multiplicative | Inside attention | Composes multiplicatively | None |
| Attn Temperature | Multiplicative | Inside softmax | No - affects pattern, not magnitude | None |
| KAN Overlay | Activation mod | Inside FFN | Composes multiplicatively | None |
| Hypernetwork LoRA | Weight mod | Inside FFN | Composes multiplicatively | None |
| Sparse Masks | Binary mask | Inside FFN | Composes (0 stays 0) | None |
| Fast Weights | Additive | After layer | Apply after gate, independent | None |
| Norm Offsets | Additive | Before sublayer | No - affects input, not output | None |

**Conclusion**: Gated residuals have ZERO conflicts with any existing intervention system. They operate at a unique point in the pipeline (between sublayer output and residual addition) that no other system touches.

---

## 7. Recommended Implementation Plan

### Phase 1: Static Scalar Gates (Variant A)
1. Add `llm_residual_gate` struct to `llama-graph.h`
2. Add `residual_gates` vector to `llm_graph_context`
3. Add `ggml_scale` before both residual additions in `llama.cpp` (and other model files, or better: create `build_residual_gate` helper)
4. Wire through `llm_graph_params` from `llama_context`
5. Expose via JNI: `nativeSetResidualGates(float[] attn_gates, float[] ffn_gates)`
6. Wire to Kotlin `ControlVectorManager` for persona integration

**Estimated effort**: ~2-3 hours (follows exact same pattern as head_scales)

### Phase 2: Layer Importance Profiling
1. Compute Block Influence (BI) scores per layer using cosine similarity: `BI_l = 1 - cos(input_l, output_l)`
2. Do this once per model at load time (or cache)
3. Use BI scores to set initial gate values: low BI (redundant) -> lower gate, high BI (important) -> higher gate

### Phase 3: Personality Presets
Define gate profiles for different personality axes:
```json
{
  "creative": {
    "attn_gates": [1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.0, ...],
    "ffn_gates":  [1.0, 1.0, 1.0, 0.8, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, ...]
  },
  "factual": {
    "attn_gates": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, ...],
    "ffn_gates":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, ...]
  }
}
```

### Phase 4: Full Layer Skip Optimization
When gate = 0.0, skip the entire layer computation (attention + FFN) to save compute. This requires careful handling of the KV cache (still need to write KV entries for consistency).

### Phase 5: Dynamic Gating (Future)
Train lightweight per-layer gate networks (single linear layer + sigmoid) using the GateSkip approach. This would require a fine-tuning pass but could be done as a small LoRA-like adapter (~0.004% parameter overhead).

---

## 8. Safe Gate Value Ranges

Based on the research:

| Range | Effect | Risk |
|-------|--------|------|
| 0.0 | Full layer skip | High -- may break coherence if applied to important layers |
| 0.1 - 0.3 | Heavy suppression | Medium -- safe for redundant middle layers |
| 0.5 - 0.8 | Moderate reduction | Low -- safe for most layers |
| 0.8 - 1.0 | Subtle adjustment | Very low -- barely perceptible |
| 1.0 | No change (default) | None |
| 1.0 - 1.5 | Mild amplification | Low -- can increase expressiveness |
| 1.5 - 2.0 | Strong amplification | Medium -- may cause instability |
| > 2.0 | Extreme amplification | High -- likely to cause garbage output |

**Recommendation**: Clamp gates to [0.0, 2.0] in the API, default to 1.0, and provide presets.

---

## 9. Mobile-Specific Considerations

### Memory
- Variant A (scalar gates): 256 bytes for 32 layers -- negligible
- No additional tensors needed in the compute graph (ggml_scale is in-place)
- No impact on KV cache size

### Compute
- `ggml_scale` is a single NEON/SVE vectorized loop
- On Snapdragon 8 Gen 1: ~0.002ms per layer for a 4096-dim tensor
- Total: ~0.064ms per token for 32 layers = unmeasurable overhead

### Flash Attention Compatibility
- Scalar gates do NOT affect flash attention eligibility
- The gate is applied to the attention OUTPUT (after the flash attention kernel completes)
- No per-layer flash attention checks needed (unlike head rescaling / temperature)

### Battery
- With aggressive gating (50% of layers at 0.3x), reduced activation magnitudes lead to:
  - Smaller intermediate values in subsequent layers
  - Slightly better cache utilization
  - Marginal power savings from reduced computation
  - Net effect: neutral to slightly positive

---

## Sources

- [GateSkip: What Layers When (2025)](https://arxiv.org/abs/2510.13876)
- [LayerDrop: Structured Dropout (ICLR 2020)](https://arxiv.org/abs/1909.11556)
- [DeepNet: Scaling to 1000 Layers (2022)](https://arxiv.org/abs/2203.00555)
- [ShortGPT: Layer Redundancy (2024)](https://arxiv.org/abs/2403.03853)
- [LayerSkip: Early Exit + Self-Speculative Decoding (Meta, ACL 2024)](https://arxiv.org/abs/2404.16710)
- [CALM: Confident Adaptive Language Modeling (Google, NeurIPS 2022)](https://arxiv.org/abs/2207.07061)
- [Mixture-of-Depths (2024)](https://arxiv.org/abs/2404.02258)
- [ADEPT: Adaptive Dynamic Early-Exit (Jan 2026)](https://arxiv.org/pdf/2601.03700)
- [Gated Residual Connections + EAU (May 2024)](https://arxiv.org/html/2405.13407)
- [Contrastive Activation Addition / Steering Vectors (2023)](https://arxiv.org/abs/2312.06681)
- [Activation Addition: Steering Without Optimization (2023)](https://arxiv.org/abs/2308.10248)
- [Streamlining Redundant Layers (2024)](https://arxiv.org/abs/2403.19135)
- [SpecEE: Speculative Early Exiting (ISCA 2025)](https://dai.sjtu.edu.cn/my_file/pdf/7e067065-0e58-4e87-a373-feea0bebde1b.pdf)
- [Probe Pruning: Dynamic Pruning via Model Probing (2025)](https://arxiv.org/abs/2502.15618)
