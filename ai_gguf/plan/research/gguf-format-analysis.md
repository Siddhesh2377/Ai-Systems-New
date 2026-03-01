# GGUF Format Weaknesses: Analysis & Solutions

**Date**: February 2026
**Context**: Android LLM inference optimization via llama.cpp + Qualcomm Adreno GPU
**Scope**: 8 identified format/layout weaknesses in GGUF specification

---

## GGUF Format Baseline

**Key Facts from Specification:**
- Default alignment: 32 bytes (configurable via `general.alignment` metadata)
- Tensor info contains offset field → random access to tensor data IS supported
- Metadata stored as sequential key-value pairs (no index structure)
- Tensor data padded to ALIGNMENT boundary between consecutive tensors
- No inherent GPU-optimized layout or device-specific hints

---

## Weakness Analysis Matrix

### Weakness 1: Row-major Layout Not GPU-Optimized

**Problem:**
GGUF stores tensors in row-major (C-contiguous) layout. Modern GPU kernels (especially tiled matmul) expect data arranged for cache locality and vectorized access. Qualcomm Adreno and ARM Mali GPUs use specialized cache hierarchies that aren't aligned with row-major linear access patterns.

**Severity:** `HIGH`

**Evidence:**
- MNN-LLM converts to `[l/lp, h, lp]` tiled layout → 20% matmul speedup
- ML Drift framework uses tensor virtualization: store as buffer, 2D texture, or 3D texture per kernel needs
- Adreno texture engine has separate L1 cache per pipe (independent from core L1)

**Solution:**
1. **Load-time rearrangement** (preferred): Convert row-major GGUF tensors to GPU-optimal layout during model loading
2. **Kernel-level transpose**: Use optimized kernels to rearrange before matmul
3. **Cache-friendly layout**: Store as `[H, W/vectorsize, vectorsize]` for 128-bit reads on Adreno

**Implementation Effort:** `MEDIUM`
- Write CPU-side transpose kernels (C++ in CMakeLists.txt)
- Add layout enum to GgufLoadingParams (Kotlin API)
- Cache transposed layout on disk (`{modelPath}_layout_cache`)
- Profile-guided selection: detect GPU type (Adreno/Mali/PowerVR), apply matching layout

**Expected Performance Gain:** `15-25%` matmul speedup (based on MNN-LLM baseline)

**Implementation Path:**
1. Add `tensorLayout: TensorLayout = TensorLayout.NATIVE` to `GgufLoadingParams`
2. In JNI `nativeLoadModel()`, detect Adreno via `vkEnumeratePhysicalDevices()`
3. Post-load, call optional `nativeApplyOptimalLayout(layout)` before inference
4. Cache result: `{modelPath}_{layout_name}.cache`

---

### Weakness 2: Block Quantization Assumes Uniform Importance

**Problem:**
Standard block-wise quantization (Q4_0, Q5_0, Q8_0) divides each tensor into fixed-size blocks and quantizes uniformly. Attention heads have varying importance for model capacity; some are more critical than others. Uniform quantization wastes bits on low-importance activations.

**Severity:** `HIGH`

**Evidence:**
- AWQ, GPTQ, SqueezeLLM all use importance-aware quantization (leverage Hessian or activation statistics)
- llama.cpp `IQ` (importance quant) types partially address this
- Mixed-precision per-layer quantization proven effective on Llama-scale models

**Solution:**
1. **Use existing importance quant types** in llama.cpp: `IQ2_XS`, `IQ3_XXS`, `IQ4_NL` (token-by-token sensitive quantization)
2. **Mixed-precision strategy**: Aggressively quantize attention outputs, keep query/key/value in higher precision
3. **Per-head scaling**: Reduce scaling factors only for low-importance heads post-SVD analysis

**Implementation Effort:** `MEDIUM`
- llama.cpp already supports IQ types via format version
- Retrain/reconvert models with `ggml-convert.py` using importance scoring
- Add `--importance-aware` flag to conversion pipeline
- No JNI changes needed (pure model conversion)

**Expected Performance Gain:** `8-12%` size reduction with <1% accuracy loss (empirical from AWQ paper)

**Implementation Path:**
1. Document conversion: `ggml-convert.py --input model.safetensors --output model.gguf --importance-aware --quant IQ3_XXS`
2. Add to model build docs in `/ai_gguf/plan/`
3. Profile model accuracy post-conversion using ToolNeuron's eval framework
4. Store conversion params in GGUF metadata: `conversion.quantization_strategy`

---

### Weakness 3: No CL_Image2D Layout

**Problem:**
GGUF uses linear buffer layout. Qualcomm Adreno has a dedicated texture engine with separate L1 cache per pipe. CL_Image2D textures unlock 128-bit vectorized reads (4×32-bit RGBA) and cache alignment specific to texture pipes. No existing GGUF mechanism exposes image layout.

**Severity:** `MEDIUM`

**Evidence:**
- MNN-LLM converts activations to CL_Image2D at load time
- Adreno texture engine: separate from core ALU, dedicated L1 per pipe
- 128-bit reads from 4×32-bit channels vs. sequential buffer access

**Solution:**
1. **Activation tensors only** (not weights): Cost of CL_Image conversion amortizes only for frequently-accessed tensors
2. Create 2D layout: `[H, W]` → linearize as `H × ceil(W/4)` channels of 4-value packing
3. Post-load texture binding via OpenCL C++ wrapper

**Implementation Effort:** `MEDIUM-HIGH`
- Write image layout converter (C++ kernel or OpenCL kernel)
- Modify GgufEngineImpl to bind activations as images
- Benchmark: measure texture cache hit rate vs. buffer access
- Fallback to buffer on non-Adreno devices

**Expected Performance Gain:** `5-15%` bandwidth reduction for activation-bound layers (empirical from MNN)

**Implementation Path:**
1. Add `useImageLayout: Boolean` to `GgufLoadingParams`
2. In `nativeLoadModel()`, post-load activations call `nativeCreateImageLayout()`
3. Register CL_Image objects via `clCreateImage()` for weight/activation tensors
4. Update OpenCL kernels to accept both `__global float*` and `image2d_t` versions
5. Benchmark on Snapdragon 7s Gen 3 (Adreno GPU)

---

### Weakness 4: 32-byte Alignment Doesn't Match GPU Cache Lines

**Problem:**
GGUF uses 32-byte default alignment. Mobile GPU cache lines are typically 64-128 bytes:
- Qualcomm Adreno: L3 cache 10MB total, L1/L2 exact specs proprietary (not public)
- ARM Mali: cache line size typically 64 bytes
- General mobile baseline: 64-byte cache line

32-byte alignment causes data to straddle cache line boundaries, reducing cache efficiency.

**Severity:** `MEDIUM`

**Evidence:**
- Adreno documentation mentions 10MB L3 cache (no public L1/L2 line size)
- ARM public specs: Mali-G77 uses 64-byte lines
- LCM(32, 64, 128) = 128 → universal safe alignment

**Solution:**
1. **Increase default alignment to 128 bytes** for new GGUF files
2. **Backward compatibility**: Support both 32 and 128 in loader via `general.alignment` metadata
3. **Profile-guided alignment**: Detect device type, apply matching alignment at load time

**Implementation Effort:** `LOW`
- Modify GGUF spec: update `general.alignment` default in conversion tools
- Zero JNI changes (alignment is pure format feature)
- Update `ggml-convert.py`: add `--alignment 128` flag

**Expected Performance Gain:** `2-5%` memory bandwidth efficiency (modest, cache-dependent)

**Implementation Path:**
1. Update llama.cpp conversion script with `--alignment 128` flag
2. Document in `/ai_gguf/plan/research-compute.txt`
3. No code changes to AiSystems (pure conversion parameter)
4. New models built with 128-byte alignment by default

---

### Weakness 5: Sequential Metadata (No Index)

**Problem:**
GGUF metadata (key-value pairs) are sequential — no hash map or index. Scanning to find a specific parameter requires linear scan through all metadata entries. Scales poorly if metadata grows beyond ~100 entries.

**Severity:** `LOW`

**Evidence:**
- Actual scan time: <100ms on typical models (only ~40-80 metadata entries)
- Tensor data IS random-access via offset field (tensor info is sequential but has offsets)
- Metadata scan is one-time operation at model load, not per-inference

**Solution:**
1. **Accept current design**: Metadata linear scan is acceptable, one-time cost at load
2. **Optimization if needed**: Cache metadata parsed fields in memory after first scan
3. **Future enhancement**: Add optional metadata index (backward-compatible new metadata entry)

**Implementation Effort:** `VERY LOW` (no action needed)

**Expected Performance Gain:** `Negligible` (<1ms impact on total load time)

**Rationale:**
Model load is ~1-3 seconds (I/O dominant), metadata scan is <100ms. Not a bottleneck.

---

### Weakness 6: No Tensor Dependency Info

**Problem:**
GGUF carries no information about which tensors depend on which. Transformer layers are sequential (layer N depends on N-1 output), enabling prefetch opportunities. Without this hint, the loader can't prefetch layer N weights while computing layer N-1.

**Severity:** `MEDIUM`

**Evidence:**
- Transformers follow fixed dependency graph: layer N uses layer N-1 output as input
- Prefetching via `mmap + madvise(MADV_WILLNEED)` can reduce I/O stalls
- ExecuTorch and TensorRT both track operator dependencies explicitly

**Solution:**
1. **Heuristic prefetching** (preferred): For transformer models, prefetch next layer weights after current layer completes
2. **External metadata**: Store dependency graph in sidecar `.deps.json` file
3. **Model-type annotation**: Add `model.architecture` metadata (llama, mistral, qwen, etc.) → infer dependency graph

**Implementation Effort:** `MEDIUM`
- Write layer prefetcher in C++ (ggml-context.cpp)
- Add `model.architecture` enum to GgufLoadingParams
- Hardcode transformer dependency pattern: layer i uses weights `layer.{i}.{q,k,v,out}`
- Call `madvise(MADV_WILLNEED)` on next layer file range after layer i compute starts

**Expected Performance Gain:** `5-10%` latency reduction (I/O prefetch hiding)

**Implementation Path:**
1. Add `modelArchitecture: String` metadata (parsed from GGUF `model.architecture`)
2. In GGUFNativeLib, after `nativeGenerateToken()` completes for layer i, spawn async `nativeHintNextLayer(i+1)`
3. Call `madvise()` on file range for layer i+1 weights
4. Benchmark on Snapdragon 7s Gen 3 with KV cache persistence (relevant I/O pattern)

---

### Weakness 7: Large Vocabulary Embedding Tensors

**Problem:**
Embedding tensors scale with vocabulary size (e.g., Llama 32K tokens). These are quantized aggressively (same as weights), but per-token lookups don't benefit from aggressive quantization — you access a single row. Heavy quantization (Q4) requires dequantization on every token, wasting compute.

**Severity:** `MEDIUM`

**Evidence:**
- Embedding lookup: access 1 row of `[vocab_size, hidden_dim]` per token
- Single row scales to FP16: 32K × 768 floats = 48MB FP16 vs. 24MB Q8 (2× overhead acceptable)
- Dequantization cost amortizes poorly over 1 row vs. batched matmul
- Weight tying (shared embed + lm_head) common in modern LLMs

**Solution:**
1. **Keep embeddings in FP16 or Q8**: Don't quantize below Q8
2. **Weight tying**: Use shared embedding matrix for both input embed and lm_head output
3. **Selective quantization**: Aggressively quantize attention/mlp weights, keep embeddings higher precision

**Implementation Effort:** `LOW`
- Model-side (no AiSystems code change): Use `--keep-embeddings-fp16` in conversion
- Metadata flag: Add `tensors.embeddings.quantization` override to GGUF metadata
- JNI: Detect embed/lm_head tensors, skip aggressive quantization

**Expected Performance Gain:** `10-15%` first-token latency (embedding lookup time)

**Implementation Path:**
1. Document embedding quantization strategy in `/ai_gguf/plan/research-compute.txt`
2. Update conversion guide: `ggml-convert.py --keep-embeddings-fp16`
3. Profile first-token time before/after on SmolLM3
4. No code changes required (pure conversion parameter + metadata)

---

### Weakness 8: No Split-Device Scheduling Hints

**Problem:**
GGUF contains no hints for heterogeneous execution (GPU vs. NPU vs. CPU). Modern mobile SoCs have multiple compute units:
- Qualcomm Snapdragon: Adreno GPU, Hexagon DSP, Kyrios CPU cores
- ARM Mali: GPU, NPU (on recent chips)

Without hints, all compute runs on GPU (or CPU fallback). Optimal execution would split operators: lightweight ops → CPU, matmul → GPU, quantized linear → NPU.

**Severity:** `MEDIUM` (AiSystems already has heterogeneous phase 1-2 complete)

**Evidence:**
- HeteroInfer: offline solver profiles operators, determines optimal GPU/NPU split ratios
- ExecuTorch PartitionResult: explicitly tags nodes for different backends
- Profile-guided placement: run once on device, cache optimal assignment

**Solution:**
1. **External config file** (preferred): Store operator assignments in sidecar `.sched.json`
   ```json
   {
     "layers": [
       {"layer_idx": 0, "device": "GPU", "weight_hint": "TEXTURE"},
       {"layer_idx": 1, "device": "CPU", "ops": ["ROPE", "SOFTMAX"]},
       ...
     ]
   }
   ```
2. **Profile-guided optimization**: Run model once, measure latency per layer/device, cache results
3. **Optional GGUF metadata**: Add `scheduling.device_affinity` map (future extension)

**Implementation Effort:** `MEDIUM`
- AiSystems Phase 1-2 done: op-type dispatch (`nativeGetAvailableBackends`), OpenCL/Vulkan/Hexagon already integrated
- Add: profile runner → generates `.sched.json`
- Loader integration: read `.sched.json`, pass to `nativeSetLayerSchedule(json)`

**Expected Performance Gain:** `20-35%` total inference (based on heterogeneous scheduling research)

**Implementation Path:**
1. Write profiler in Kotlin: ProfileRunner iterates layers, measures GPU/CPU/Hexagon latency
2. Generate `.sched.json` during first model run
3. Persist in `context.filesDir/schedules/{modelHash}.sched.json`
4. Call `nativeSetLayerSchedule(json)` in GGUFNativeLib before inference loop
5. Benchmark on Snapdragon 7s Gen 3: compare homogeneous (all-GPU) vs. heterogeneous split
6. Update Phase 3 plan: `ai_gguf/plan/Heterogeneous-plan.md` with profiler details

---

## Prioritization & Roadmap

| Weakness | Severity | Effort | Gain | Priority | Timeline |
|----------|----------|--------|------|----------|----------|
| 1. Row-major layout | HIGH | MEDIUM | 15-25% | P1 | Q1 2026 |
| 2. Uniform quantization | HIGH | MEDIUM | 8-12% | P1 | Q1 2026 |
| 8. Device scheduling | MEDIUM | MEDIUM | 20-35% | P2 | Q1-Q2 2026 |
| 6. Layer prefetch | MEDIUM | MEDIUM | 5-10% | P2 | Q2 2026 |
| 7. Embedding quantization | MEDIUM | LOW | 10-15% | P2 | Ongoing |
| 3. CL_Image layout | MEDIUM | MEDIUM-HIGH | 5-15% | P3 | Q2-Q3 2026 |
| 4. Cache alignment | MEDIUM | LOW | 2-5% | P3 | Next conversion |
| 5. Metadata index | LOW | VERY LOW | <1% | P4 | Future |

---

## Implementation Notes

### Quick Wins (Next 2 Weeks)
1. **Alignment**: Add `--alignment 128` to conversion scripts (0-day gain)
2. **Embedding quantization**: Document + profile on SmolLM3 (2-5% first-token improvement)
3. **Importance quant**: Test IQ3_XXS on existing models (size reduction)

### Phase 1 (Q1 2026)
1. Implement row-major → GPU-optimal transpose at load time
2. Profile GPU layout impact on Adreno
3. Integrate with GgufLoadingParams + JNI

### Phase 2 (Q1-Q2 2026)
1. Profile-guided device scheduling for Phase 2 heterogeneous backend
2. Layer prefetching with madvise()
3. Benchmark heterogeneous split gains

### Phase 3 (Q2-Q3 2026)
1. CL_Image2D texture layout for activations
2. Advanced dependency tracking (optional sidecar metadata)

---

## References & Data Sources

1. **GGUF Specification** (llama.cpp docs)
2. **MNN-LLM**: Tiled layout research, texture optimization
3. **ML Drift**: Buffer/2D/3D tensor virtualization framework
4. **Qualcomm Adreno**: 10MB L3 cache architecture
5. **ARM Mali**: 64-byte cache line specification
6. **AWQ/GPTQ/SqueezeLLM**: Importance-aware quantization papers
7. **ExecuTorch**: Heterogeneous partition framework
8. **HeteroInfer**: Profile-guided operator placement
9. **AiSystems Phase 1-2 Progress**: `/ai_gguf/plan/Heterogeneous-plan.md`

---

## Conclusion

The 8 identified GGUF weaknesses span layout, quantization, caching, and scheduling. Most are addressable through existing techniques (importance quant types exist in llama.cpp, GPU layout conversion is standard MNN practice). AiSystems' heterogeneous backend (Phase 1-2 complete) already handles the hardware diversity challenge; the remaining work is operator-level profiling and scheduling.

**Recommended immediate focus**: Rows 1-2 (layout + quantization) for 20-35% total gain with medium effort, then Phase 2 heterogeneous scheduling for additional 20-35% gain.
