================================================================================
UNIFIED COMPUTE GRAPH RESEARCH — COMPLETE FINDINGS
================================================================================

This directory contains comprehensive research and findings on building unified
compute graphs for mobile LLM inference. All findings synthesized from research
on: Megakernel (Hazy Research), FlashInfer (MLSys 2025), ML Drift (Google CVPR
2025), HeteroInfer (Tsinghua SOSP 2025), and production frameworks.

================================================================================
FILES IN THIS RESEARCH
================================================================================

EXECUTIVE SUMMARY (START HERE):
  RESEARCH-SYNTHESIS-SUMMARY.txt (247 lines)
    Quick synthesis of key findings, recommendations, and 12-week roadmap
    Perfect for: Decision-makers, project leads, quick reference

COMPREHENSIVE TECHNICAL REPORT:
  unified-compute-graph-report.md (414 lines)
    In-depth technical analysis with architecture details, code examples,
    implementation guidance, risk assessment
    Perfect for: Engineers, architects, those implementing recommendations

SUPPORTING RESEARCH:
  research-compute.txt (previous comprehensive research on mobile optimization)
  old/Heterogeneous-plan.md (Phase 1-2 heterogeneous GPU/NPU scheduling work)

================================================================================
QUICK ANSWERS TO KEY QUESTIONS
================================================================================

Q: Can we build a unified compute graph?
A: YES. GGUF proves it — multiple projects load same weights with different
   engines (vLLM→CUDA, llama.cpp→ggml, Cactus→ARM SIMD).

Q: What should it look like?
A: 3-layer abstraction:
   • IR Layer: Model-agnostic DAG (12-15 op types)
   • Lowering: Per-backend translation (OpenCL, Vulkan, NPU, CPU)
   • Execution: Graph replay + tensor virtualization + hetero dispatch

Q: What to steal from each framework?
A: • Megakernel: NOT for mobile (wrong hardware), steal fusion concept
   • FlashInfer: JIT + binary caching + composable formats
   • ML Drift: Tensor virtualization + stage-aware kernels (RECOMMENDED)
   • HeteroInfer: GPU+NPU concurrent dispatch + unified memory
   • llama.cpp: All of it (GGUF loading, KV cache, sampling)

Q: Which option should AiSystems pursue?
A: OPTION B (Tensor Virtualization + HeteroInfer)
   • Effort: 8-12 weeks
   • Performance: 5-11x speedup (6 tok/s → 30-55 tok/s decode)
   • ROI: Excellent (best bang for buck)
   
   Quick alternative: Option A (Graph Replay) = +30-50% in 4-6 weeks
   Not recommended: Option C (Full custom) = 10-14 weeks, diminishing returns

================================================================================
CRITICAL TECHNICAL INSIGHTS
================================================================================

1. MUL_MAT is 90% of compute time
   → Optimize ONE kernel perfectly = solve 90% of problem
   → Best target: Adreno CL_Image layout (5-7x decode speedup)

2. No single processor saturates memory bandwidth
   GPU alone: 43 GB/s (60% utilization)
   NPU alone: 43 GB/s (60% utilization)
   GPU + NPU: 59.5 GB/s (88% utilization!)
   → GPU+NPU concurrent dispatch is killer optimization

3. Prefill and decode need different kernels
   Prefill: batch GEMM (compute-bound)
   Decode: GEMV (memory-bound)
   → Stage-aware kernel selection = 2-3x speedup

4. OpenCL > Vulkan for Adreno on mobile
   Qualcomm officially recommends this
   Vulkan has known performance issues on Adreno mobile
   → Use OpenCL for Snapdragon, Vulkan for Mali

5. Graph replay (decode only) eliminates launch overhead
   Record identical decode graph once, replay every token
   → +30% from removing per-op CPU overhead

6. Tensor virtualization is portable
   Same logical tensor, different physical layouts:
   • Adreno: TEXTURE_2D (exploit texture L1 cache)
   • Mali: NHWC (subgroup operations)
   • CPU: Linear NEON
   → One abstraction, many implementations

================================================================================
RECOMMENDED 12-WEEK IMPLEMENTATION PLAN (OPTION B)
================================================================================

WEEK 1-2: Graph IR + Logical Layer
  Task: Build model-agnostic compute graph abstraction
  • ComputeGraph class (Op, Tensor, DAG)
  • GGUF → Graph converter (architecture-independent)
  • Graph caching to disk (skip rebuild on restart)
  Files: ai_gguf/src/main/cpp/src/graph/{ComputeGraph, GraphBuilder}.*

WEEK 3-4: Tensor Virtualization + Kernel Registry
  Task: Decouple logical tensors from physical layouts
  • TensorVirtualizer (choose layout per device/stage)
  • StageAwareKernelRegistry (prefill vs decode kernels)
  • Per-device kernel binary caching
  Files: ai_gguf/src/main/cpp/src/graph/{TensorVirtualizer, KernelRegistry}.*

WEEK 5-8: Backend Lowering + Kernel Compilation
  Task: Write device-optimized kernels
  
  ADRENO (OpenCL):
    • CL_Image weight loading (5-7x decode boost)
    • GEMV specialization (compile for any M/N/K)
    • Binary kernel cache (skip recompilation)
  
  MALI (Vulkan):
    • Warptile tuning (medium tiles for mobile)
    • VkPipelineCache (skip shader recompilation)
  
  CPU (ARM NEON):
    • NEON vectorization fallback
  
  Files: ai_gguf/src/main/cpp/src/backends/{Adreno, Mali, CPU}.h/cpp

WEEK 9-10: Heterogeneous Dispatch (GPU+NPU)
  Task: Enable concurrent GPU+NPU execution
  • Partition matmul rows (75% GPU, 25% NPU)
  • Unified memory mapping (zero-copy LPDDR5)
  • Predictive polling (no blocking)
  Files: ai_gguf/src/main/cpp/src/graph/HeterogeneousDispatcher.*

WEEK 11: Performance Profiling & Tuning
  Task: Measure bottlenecks and optimize
  • Per-op latency characterization
  • Memory bandwidth measurement
  • Kernel tuning based on device capabilities
  Files: ai_gguf/src/main/cpp/src/profiler/GraphProfiler.*

WEEK 12: Testing + Documentation
  Task: Validate correctness and document architecture
  • Correctness vs llama.cpp baseline
  • Device-specific regression tests
  • Architecture guide for future contributors
  • API documentation

EXPECTED OUTCOME:
  Device: Snapdragon 8 Gen 3 (Adreno 750)
  Model: Llama2-7B Q4_0
  Before: ~6 tok/s decode (llama.cpp baseline)
  After:  ~30-55 tok/s decode (5-11x speedup)

================================================================================
WHAT NOT TO DO (ANTI-PATTERNS)
================================================================================

✗ MEGAKERNEL FUSION
  Why not: Requires 240KB shared memory (mobile has 32-64KB only)
  Impact: Won't work on tile-based GPUs (Adreno, Mali)

✗ CUSTOM PROPRIETARY FORMAT (like Cactus .cact)
  Why not: Breaks GGUF ecosystem compatibility
  Impact: Users can't use Ollama models, llama.cpp models, etc.

✗ REPLACE ggml CORE OPS WHILE KEEPING GGUF
  Why not: Violates separation of concerns
  Impact: Tight coupling, hard to maintain

✗ FULL CUSTOM ENGINE (Option C)
  Why not: 10-14 weeks for 800% speedup vs 8-12 weeks for 500% (Option B)
  Impact: Diminishing ROI, high maintenance burden

✗ DELAY DECISION
  Why not: Clear recommendation is Option B
  Impact: Wasted time, unclear project direction

================================================================================
RISK ASSESSMENT & MITIGATION STRATEGIES
================================================================================

HIGH RISK: Kernel Compilation Bugs
  Likelihood: High
  Impact: Build failure, runtime crashes
  Mitigation:
    1. Comprehensive unit tests for each kernel variant
    2. Incremental rollout: Adreno first, then Mali, then CPU
    3. Feature detection (only enable if device supports)
    4. Fallback to llama.cpp if custom backend fails

MEDIUM RISK: Memory Layout Mismatches
  Likelihood: Medium
  Impact: Numerical errors, wrong outputs
  Mitigation:
    1. Validate every layout transformation against reference
    2. Numerical correctness tests (compare against llama.cpp)
    3. Layout transformation unit tests
    4. Per-device golden output verification

MEDIUM RISK: Hardware Variance
  Likelihood: Medium (Adreno driver versions vary)
  Impact: Crashes on older chips
  Mitigation:
    1. Feature detection (query device capabilities)
    2. Conservative defaults (disable optimizations if unsupported)
    3. Graceful degradation to llama.cpp
    4. Hardware matrix testing (old/new Adreno, Mali variants)

LOW RISK: NPU Sync Overhead
  Likelihood: Low
  Impact: No speedup from GPU+NPU if overhead dominates
  Mitigation:
    1. Profile first (measure GPU+NPU overhead)
    2. Enable GPU+NPU only if >10% gain
    3. Dynamic enable/disable (user can opt out)

MEDIUM RISK: Regression vs llama.cpp
  Likelihood: Medium (common in optimization)
  Impact: Users revert, project loses momentum
  Mitigation:
    1. CI/CD with automatic perf regression tests
    2. Keep llama.cpp fallback (gpuLayers=0 always works)
    3. A/B testing on ToolNeuron (compare side-by-side)
    4. Clear communication about known issues

================================================================================
NEXT STEPS FOR AISYSTEMS TEAM
================================================================================

IMMEDIATE (Today):
  1. Read RESEARCH-SYNTHESIS-SUMMARY.txt (5 min)
  2. Read unified-compute-graph-report.md (15 min)
  3. Decision: Option A, B, or C?

SHORT-TERM (This week):
  4. If Option B approved:
     - Schedule 12-week sprint
     - Allocate engineer time (1 FTE or 2 part-time)
     - Set up multi-backend build/test environment
  
  5. If Option A (quick wins):
     - Scope graph replay (4-6 weeks)
     - Implement OpenCL binary caching first

  6. If Option C (full custom):
     - Clarify: Which SoC variants to target?
     - Risk: 10-14 weeks for diminishing returns

MEDIUM-TERM (Next 2 weeks):
  7. Week 1 prep:
     - Set up ci/cd for performance regression tests
     - Prepare GGUF test suite (small models for fast iteration)
     - Design ComputeGraph class (interfaces, data structures)

  8. Week 1 start:
     - Implement ComputeGraph IR
     - Build GGUF → Graph converter
     - Cache graph to disk

================================================================================
TECHNICAL REFERENCES (PAPERS & PROJECTS)
================================================================================

Megakernel LLM Inference (Hazy Research / Stanford)
  • Whole-model fusion into single GPU kernel
  • 2.5x faster than vLLM, 1.5x faster than SGLang on H100
  • NOT applicable to mobile (shared memory constraints)

FlashInfer (MLSys 2025 Best Paper)
  • JIT kernel compilation + binary caching
  • Composable attention kernel formats
  • Integrated: vLLM, SGLang, MLC-LLM

ML Drift (Google, CVPR 2025) ← RECOMMENDED PATTERN
  • Tensor virtualization (logical ≠ physical)
  • Stage-aware kernel selection (prefill vs decode)
  • 5-11x speedup on Adreno 750 (Gemma 2B)
  • Application: Mobile LLM inference

HeteroInfer (Tsinghua, SOSP 2025) ← RECOMMENDED PATTERN
  • GPU+NPU concurrent execution
  • Weight partitioning (75% GPU, 25% NPU)
  • 51.1 tok/s decode on Snapdragon 8 Gen 3 (InternLM-1.8B)
  • Bandwidth: 43 GB/s single → 59.5 GB/s parallel (88% utilization)

Cactus Compute (YC S25 startup)
  • Reference implementation of custom mobile engine
  • 91 tok/s Galaxy S25 Ultra, 136 tok/s iPhone 17 Pro
  • ARM SIMD kernels + vendor-specific NPU scheduling
  • Note: Full custom engine approach (10-14 week effort)

llama.cpp (Production Baseline)
  • GGUF loading + multi-backend scheduler + KV cache
  • Already integrated into AiSystems
  • Fallback for all new optimizations

================================================================================
CONTACT & QUESTIONS
================================================================================

For questions about this research:
  • See RESEARCH-SYNTHESIS-SUMMARY.txt for quick answers
  • See unified-compute-graph-report.md for technical details
  • Existing heterogeneous work: old/Heterogeneous-plan.md

For implementation questions:
  • Reference llama.cpp source (ggml backends)
  • Reference ML Drift paper (tensor virtualization patterns)
  • Reference HeteroInfer paper (GPU+NPU dispatch patterns)

For decision-making:
  • Option B (recommended) = ML Drift + HeteroInfer + llama.cpp
  • Option A (quick wins) = Graph replay + kernel caching
  • Option C (ambitious) = Full custom engine (diminishing ROI)

================================================================================
