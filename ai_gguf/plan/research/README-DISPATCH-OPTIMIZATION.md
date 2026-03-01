# Vulkan/OpenCL Dispatch Optimization Research — Complete Compilation

**Date:** February 24, 2026
**Status:** Research compilation complete — NO additional web searches
**Total Output:** 828 lines across 2 comprehensive documents

## Documents in This Directory

### 1. Full Technical Report (RECOMMENDED FOR DETAILED REVIEW)
**File:** `vulkan-opencl-dispatch-optimization-report.md`
**Size:** 24KB, 566 lines
**Purpose:** Academic-grade comprehensive analysis

**Sections:**
- Executive summary (96x performance gap analysis)
- 8 optimization strategies with detailed technical explanations
- Expected performance gains and risk assessments
- Complete implementation roadmap (4 phases, 14+ weeks)
- Files to modify with absolute paths
- Risk mitigation strategies for each approach
- Performance projection scenarios

**Best for:** Understanding the complete technical context, long-term planning, stakeholder presentations

**Key Finding:** OpenCL CL_Image weight storage offers 3.0x speedup (2.7 → 8.1 tok/s) with MEDIUM difficulty, LOW risk in 2-3 weeks.

---

### 2. Quick Reference Guide (RECOMMENDED FOR IMMEDIATE ACTION)
**File:** `dispatch-optimization-summary.txt`
**Size:** 12KB, 262 lines
**Purpose:** Actionable checklist and executive brief

**Sections:**
- One-page problem statement
- Priority-ranked solutions (#1-7) with effort estimates
- Performance roadmap: 2.7 → 8.1 → 10.1 → 25.5 tok/s
- Critical findings summary
- Risk mitigation checklist
- Implementation next steps (immediate, short-term, medium-term, long-term)
- File references with absolute paths

**Best for:** Teams starting implementation, weekly standup briefs, resource planning

---

## Research Summary

### The Problem
Current llama.cpp Vulkan/OpenCL:
- **Observed:** 2.7 tok/s (3B Q4 model, Snapdragon 8 Gen 3)
- **Theoretical:** 263 tok/s (GPU+NPU combined bandwidth)
- **Gap:** 96x slower
- **Root cause:** CPU-side dispatch overhead + GPU scheduling inefficiency (NOT hardware limitation)

### The Solution (7 Strategies, Ranked by ROI)

| Priority | Strategy | Expected Gain | Effort | Risk | Status |
|----------|----------|---------------|--------|------|--------|
| **#1** | OpenCL CL_Image Weights | 3.0x (→8.1 t/s) | 2-3 wks | LOW | READY NOW |
| **#2** | UMA Backend Scheduler | +25% (→10.1 t/s) | 4-5 wks | MED | Foundation |
| **#3** | Vulkan Push Descriptors | +15% (→11.6 t/s) | 2-3 wks | MED | Vulkan-only |
| **#4** | Command Buffer Replay | +8-14% | 4-6 wks | HIGH | Advanced |
| **#5** | Barrier Optimization | +5-10% | 3-4 wks | MED-HIGH | Advanced |
| **#6** | Mega-Kernel Fusion | +8-15% | 4-6 wks | HIGH | Future |
| **#7** | GPU+NPU Heterogeneous | 2.2x (→25.5 t/s) | 6-8 wks | HIGH | Q2 2026 |

### Critical Strategic Recommendation
**OpenCL > Vulkan for Adreno (3-4x performance difference)**
- Qualcomm officially recommends OpenCL for mobile LLM
- Vulkan has device-lost issues above batch 32
- CL_Image texture cache unavailable in Vulkan on Adreno architecture
- **Implication:** Prioritize OpenCL optimization; Vulkan enhancements for portability only

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
Implement OpenCL CL_Image weight storage
- **Target:** 8.1 tok/s (+200% from baseline)
- **Achievable:** YES (MNN-LLM proven, directly applicable)
- **Key file:** `/home/home/CLionProjects/llama.cpp-android/ggml/src/ggml-opencl.cpp`

### Phase 2: Scheduler Optimization (Weeks 3-4)
Add UMA backend scheduler with predictive polling
- **Target:** 10.1 tok/s (+275% from baseline, Transformer-Lite parity)
- **Prerequisite:** Phase 1 complete
- **Key files:** `llama-context.cpp`, `ggml-backend.h`

### Phase 3: Portability (Weeks 5-6)
Vulkan push descriptors for non-Adreno devices
- **Target:** 11.6 tok/s (full OpenCL + Vulkan parity)
- **Optional:** Skip if Adreno-only deployment

### Phase 4: Heterogeneous (Weeks 9-14)
GPU+NPU concurrent tensor split (if needed after Phase 2)
- **Target:** 25.5 tok/s (HeteroInfer-class performance)
- **Prerequisite:** Hexagon HTP backend stability

---

## Key Technical Insights

### 1. Adreno Has Dual Memory Hierarchies
- **Compute L1:** 48KB/CU, global memory cache, 30-40% hit rate
- **Texture L1:** 16-32KB/block, optimized prefetcher, 70-85% hit rate for 2D layouts
- **Current llama.cpp:** Uses neither (generic global buffer access)
- **Solution:** Store weights as CL_RGBA images → 3-4x speedup from cache hits

### 2. Synchronous Tensor Copies Waste 40-60% of Time
- llama.cpp blocks after every GPU kernel for tensor readback
- Unified Memory Architecture (UMA) makes this unnecessary
- HeteroInfer solution: predictive polling instead of blocking
- **Benefit:** Eliminates microsecond-scale stalls per 40-50 tokens

### 3. GPU+NPU Bandwidth Not Saturated
- Single GPU: 40-45 GB/s (59-66% of LPDDR5X 68 GB/s)
- Single NPU: 40-45 GB/s
- GPU+NPU parallel: 60 GB/s (88% saturation)
- **Implication:** After single-GPU optimization, must do heterogeneous dispatch

### 4. Descriptor Overhead Dominates CPU Time
- 200+ descriptor updates per token
- Current: ~20-30ms per token in descriptor bookkeeping
- VK_KHR_push_descriptor: reduces to single CPU call
- **Vulkan users:** 8-15% speedup available

### 5. Decode Graph Rebuilt Every Token (Wasteful)
- Identical graph shape every token
- CUDA Graphs concept: record once, replay 128+ times
- Vulkan/OpenCL can emulate via command buffer recording
- **Benefit:** 8-14% for decode-only workloads

---

## File References

### Research Output
- Full report: `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/plan/vulkan-opencl-dispatch-optimization-report.md`
- Quick ref: `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/plan/dispatch-optimization-summary.txt`
- This file: `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/plan/README-DISPATCH-OPTIMIZATION.md`

### Source Code to Modify
**llama.cpp OpenCL Backend:**
- `/home/home/CLionProjects/llama.cpp-android/ggml/src/ggml-opencl.cpp`
- New kernels: `kernel_mul_mv_q8_0_image()`, `kernel_mul_mv_q4_0_image()`

**llama.cpp Scheduler:**
- `/home/home/CLionProjects/llama.cpp-android/src/llama-context.cpp`
- `/home/home/CLionProjects/llama.cpp-android/include/ggml-backend.h`

**llama.cpp Vulkan:**
- `/home/home/CLionProjects/llama.cpp-android/ggml/src/ggml-vulkan.cpp`

**AiSystems Build Integration:**
- `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/src/main/cpp/CMakeLists.txt`
- `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/src/main/cpp/src/ai_gguf.cpp`

### Related Research
- Prior work: `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/plan/research-compute.txt`
- Heterogeneous plan: `/home/home/AndroidStudioProjects/AiSystems/ai_gguf/plan/old/Heterogeneous-plan.md`

---

## Next Steps

### This Week
1. Read full report (`vulkan-opencl-dispatch-optimization-report.md`)
2. Baseline benchmark with profiling tools
3. Prototype CL_Image kernel for Q8_0 weights

### Week 1-2
1. Implement OpenCL CL_Image in ggml-opencl.cpp
2. Device capability detection
3. Multi-device testing
4. Measure 3.0x speedup expectation

### Week 3-4
1. Design UMA memory mapping layer
2. Implement predictive polling
3. Profile overhead vs synchronous waits
4. Target 1.25x additional speedup

### Week 5+
1. Vulkan push descriptors (if needed)
2. Advanced optimizations (barrier reduction, command replay)
3. Heterogeneous GPU+NPU (if needed for further gains)

---

## Success Criteria

- [ ] Phase 1: 8.1 tok/s achieved (CL_Image)
- [ ] Phase 2: 10.1 tok/s achieved (UMA scheduler)
- [ ] Phase 3: 11.6 tok/s achieved (Vulkan push descriptors, optional)
- [ ] Phase 4: 25.5 tok/s achieved (GPU+NPU heterogeneous, future)

**Primary success:** Reach 10+ tok/s (Transformer-Lite parity) in 4-5 weeks using Phases 1-2.

---

## Questions?

Refer to the full technical report for:
- Detailed technical explanations of each strategy
- Risk analysis and mitigation approaches
- Performance projection scenarios
- Complete implementation roadmap with code file references
- Detailed breakdowns of each optimization technique

---

**Compilation Date:** February 24, 2026
**Next Milestone:** CL_Image kernel prototype
**Team:** AiSystems Heterogeneous GPU/NPU Optimization
