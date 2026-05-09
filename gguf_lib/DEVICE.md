# Device profile — Snapdragon 7s Gen 3 (SM7635)

Measured on the connected reference device with a standalone benchmark
(`/tmp/mobile_bench/bench_mobile.cpp`, NDK r27d, `-O3 -march=armv8.4-a+fp16`).

## Hardware

```
SoC:    SM7635 (Snapdragon 7s Gen 3, Volcano)
CPU:    8 cores
        4× Cortex-A78 P-cores  @ 2.40 – 2.50 GHz  (cpu4..cpu7)
        4× Cortex-A55 E-cores  @ 1.80 GHz         (cpu0..cpu3)
GPU:    Adreno 810  (2 compute units, 32 KB local mem, 3.7 GB visible)
RAM:    7.28 GB total, ~3 GB available
NPU:    Hexagon HTP V73  (not exercised by this benchmark)
```

## DRAM is the ceiling

This is the single most important number on this device. All batch=1 LLM
decode is bound by it.

| Workload | Bandwidth |
|---|---|
| `memcpy` 1 thread, 64 MB blocks | 6.75 GB/s |
| `memcpy` 2 threads | 6.89 GB/s |
| `memcpy` 4 threads | 6.54 GB/s |
| `memcpy` 8 threads (P+E) | 6.68 GB/s |
| GPU 16 MB transfer | 6.88 GB/s |

**Saturates at ~6.9 GB/s with 2 threads.** Adding more threads cannot help.
The CPU and the GPU hit the same ceiling for large transfers — that's UMA
(unified memory architecture) proving itself. The GPU has **zero memory-
bandwidth advantage** for streaming weights through. It only "wins" on data
already cached internally:

| GPU transfer size | Bandwidth |
|---|---|
| 4 KB | 3.74 GB/s (dispatch overhead dominates) |
| 64 KB | 14.50 GB/s (internal coherent cache) |
| 1 MB | 11.33 GB/s (still partly cached) |
| 16 MB | **6.88 GB/s (DRAM-bound, matches CPU)** |

## CPU GEMV (the actual transformer op)

FP16 NEON dot-product, weights laid out `[N, K]` row-major so each output
row reads a contiguous K-vector. Times are per call.

### 2048 × 2048 (~1B model hidden dim)

| Threads | ms | GFLOPS | GB/s | Notes |
|---|---|---|---|---|
| 1 | 1.45 | 5.8 | **5.78** | already at DRAM ceiling |
| 2 | 1.70 | 4.9 | 4.92 | thread spawn cost > gain |
| 4 | 1.66 | 5.1 | 5.06 | no further gain |

### 4096 × 4096 (~7B model hidden dim)

| Threads | ms | GFLOPS | GB/s | Notes |
|---|---|---|---|---|
| 1 | 4.68 | 7.2 | 7.17 | partial L2 reuse |
| 2 | 3.27 | 10.3 | **10.27** | best |
| 4 | 3.33 | 10.1 | 10.08 | flat |

### 2048 × 32K (LM head / vocab projection)

| Threads | ms | GFLOPS | GB/s | Notes |
|---|---|---|---|---|
| 1 | 15.8 | 8.3 | 8.27 | weight is read once for 32K outputs |
| 2 | 9.80 | 13.4 | **13.37** | compute-bound, scales |
| 4 | 9.80 | 13.4 | 13.37 | flat |

**Interpretation**: small square GEMVs (the per-layer ops) are DRAM-bound on
a single thread. Wide-output GEMVs (LM head) become compute-bound and scale
to two threads. Past two threads, nothing helps a memory-bound op on this
SoC.

## GPU (Adreno 810) — the disappointing reality

```
Dispatch overhead (no-op kernel + sync): 0.47 ms per call
```

Every kernel launch + clFinish costs ~0.5 ms. A transformer has ~169 ops
per decoded token (24 layers × 7 GEMVs + LM head for a 1B model). At
0.47 ms/op that's **84 ms/token of pure scheduling overhead**, before any
compute happens. That alone makes per-op GPU offload infeasible.

### Naive GEMV on GPU vs CPU

Same workloads, naive OpenCL kernel (one work-item per output row, no
shared-memory tiling):

| Size | GPU GB/s | CPU best GB/s | Verdict |
|---|---|---|---|
| 2048 × 2048 | 2.05 | 5.78 | **GPU 2.8× slower** |
| 4096 × 4096 | 2.19 | 10.27 | **GPU 4.7× slower** |
| 2048 × 32K | 2.27 | 13.37 | **GPU 5.9× slower** |

Adreno 810 only has 2 CUs. With proper tiling (shared memory cache, vec4
loads, subgroup reduction) llama.cpp's real OpenCL kernel does ~2-3×
better — but even at peak it would **not exceed** CPU + NEON + KleidiAI on
this SoC.

This contradicts the "GPU for VLM encode, CPU for decode" hybrid strategy
*on this device*. On a phone with Adreno 730+ / 740+ / 8 Gen 1-tier GPU
the math flips, but on 7s Gen 3 the GPU is a worse decoder and a worse
encoder than the CPU. The GPU earns its keep only for genuinely parallel
compute that fits in 32 KB local memory (image preprocess kernels,
softmax with high arithmetic intensity, depth/style pre/post-processing).

## Batched GEMM — arithmetic intensity scaling

`Y[M, N] = X[M, K] @ W[N, K]^T` at K=N=2048, varying batch dim M.
CPU: single thread, 4×4 NEON microkernel, weight-stationary.
GPU: 16×16 tiled OpenCL kernel with local-memory tile cache.

| M | CPU GFLOPS | CPU GB/s | GPU GFLOPS | GPU GB/s | Winner | Intensity (FLOPs/byte) |
|---|---|---|---|---|---|---|
| 1 | 2.4 | 2.36 | 1.1 | 1.11 | **CPU 2.2× faster** | 1.0 |
| 4 | 9.4 | 2.36 | 4.4 | 1.11 | CPU 2.1× faster | 4.0 |
| 16 | 9.5 | 0.60 | **17.6** | 1.11 | **GPU 1.9× faster** | 15.9 |
| 64 | 9.6 | 0.15 | 17.8 | 0.29 | GPU 1.9× faster | 62.1 |
| 256 | 9.6 | 0.04 | 17.9 | 0.08 | GPU 1.9× faster | 227.6 |
| 512 | 9.5 | 0.02 | 17.9 | 0.04 | GPU 1.9× faster | 409.6 |

### What this confirms

- **CPU 1-thread compute ceiling: ~9.5 GFLOPS** (NEON FP16 peak on one A78 @ 2.4 GHz)
- **GPU sustained compute ceiling: ~18 GFLOPS** (Adreno 810 with 2 CUs)
- **GPU has 1.9× more compute** than one CPU core, but only when arithmetic
  intensity exceeds ~16 FLOPs/byte
- **Crossover point: M ≈ 4**. Below it, the GPU's dispatch tax + workgroup
  underutilization dominate. Above it, the GPU's parallelism wins.
- **Workgroup utilization matters**: at M=1 the tiled GEMM kernel uses only
  1 of 16 rows of its workgroup → 6% utilization → 1.45 GB/s, *worse* than
  the naive M=1 GEMV kernel (2.05 GB/s) which doesn't waste rows.

### Routing implications

| Workload | Effective M | Backend |
|---|---|---|
| Decode at batch=1 | 1 | **CPU** (GPU 2× slower) |
| Speculative verify (K=4-8 drafts) | 5-9 | **CPU** (still loses on Adreno 810) |
| Prefill chunked at 32 tokens | 32 | **GPU** (~2× faster) |
| Prefill chunked at 128 tokens | 128 | **GPU** (~2× faster) |
| Vision encoder ViT patches | 64-256 | **GPU** (~2× faster) |
| Embedding batched at 16+ docs | 16+ | **GPU** (~2× faster) |
| Single embedding query | 1 | **CPU** |

The absolute compute ceiling on this device is ~18 GFLOPS (GPU-bound on
M ≥ 16 ops) or ~13 GFLOPS (2-thread CPU-bound for ops with large output
dim like the LM head). At batch=1 decode, the workload is memory-bound at
~3.5 GFLOPS effective regardless of backend — the bandwidth ceiling does
not move.

## Realistic decode simulation (1B model)

24 layers × 7 GEMVs of [2048, 2048] + 1 LM head [2048, 32K] = 169 ops/token.
This is a *naive* sim — fresh threads spawned per GEMV — but useful as
a lower bound:

| Threads | ms/token | tok/s |
|---|---|---|
| 1 | 349 | 2.9 |
| 2 | 450 | 2.2 |
| 4 | 497 | 2.0 |

More threads = slower. Two causes:
1. **DRAM contention** — threads fight for the 6.9 GB/s bus
2. **Thread spawn cost** per op (840 spawns for 5 tokens) > op work

Real llama.cpp on this device runs ~21 tok/s for Q8_0 1B (per `MEMORY.md`),
~7× this benchmark. The gap is:

- **Q8_0 weights** halve memory traffic vs fp16 (0.5 vs 1.0 GB read per op)
- **KleidiAI fused dequant + matmul** kernels (no separate dequant pass)
- **Persistent thread pool** (no per-op spawn overhead)
- **Cache-aware blocking** (real kernels keep tiles warm in L1/L2)

So the 2.9 → 21 tok/s improvement comes from quantization, fused kernels,
and a thread pool — *not* from parallelism beyond 2 threads.

## Memory footprint (process)

```
Baseline RSS:       3.1 MB
After all benches: 42.1 MB
VmPeak (virtual):  11.7 GB  (alloc/free churn — not actual residency)
```

## What this changes for gguf_lib design

Three concrete consequences worth recording:

### 1. GPU offload on Adreno 810 is not worth it for decode

Dispatch overhead (0.47 ms × 169 ops/token = 84 ms) alone exceeds any
plausible gain. Per-op offload is dead on arrival. **Keep decode on CPU.**

GPU might still be worth it for:
- Vision preprocess (image resize, normalize) where work is bulk and
  parallel-friendly
- Image-only ops that run *once* per call (no per-token dispatch tax)
- Long-context attention (>4K) where N² matters and the kernel runs
  long enough to amortize the 0.5 ms launch

For ≤2K context, single-token decode, this device: **CPU-only is the right
default**. That's what `gguf_lib` already does — the data confirms it.

### 2. The thread-mode knob hits a hard wall at 2 threads

DRAM is saturated by 2 threads. The current modes:

| Mode | gen threads | reality |
|---|---|---|
| power_saving (0) | 1 | optimal — fewest threads, less DRAM contention |
| balanced (1) | 2 | optimal for memory-bound ops, scales LM head |
| performance (2) | 4 | **wasted** — 4 threads slower than 2 for square GEMVs |

`performance` mode should drop to 2 generation threads, not 4. Keep 4 for
prompt-eval (compute-bound, batch matmul scales differently).

### 3. The biggest wins are quantization + thread pool, not parallelism

Two changes that would give 5-7× decode speedup, both already in llama.cpp
and already used by `gguf_lib`:

- **Q8_0 KV / Q4_0 weights** — halves or quarters DRAM traffic
- **KleidiAI fused kernels** — eliminates dequant pass + uses int8mm

Things that look promising but **won't help** on this device:

- More CPU threads beyond 2 for decode
- GPU offload via OpenCL for square GEMVs
- VLM encode on GPU (Adreno 810 is too small)
- Layer skipping by an external GPU predictor (dispatch tax kills it)

## Methodology

| Variable | Value |
|---|---|
| Compiler | NDK r27d clang, `-O3 -march=armv8.4-a+fp16 -static-libstdc++` |
| Threads | `pthread`, pinned via `sched_setaffinity` |
| Bandwidth test | 64 MB/thread `memcpy`, ≥3 iters, ~256 MB total work |
| GEMV | FP16 NEON dot-product, `vfmla` fused multiply-add, vec8 loads |
| GPU GEMV | OpenCL 3.0, fp16 vload8, no tiling, profiling queue |
| Dispatch test | `clEnqueueNDRangeKernel` (1 work-item) + `clFinish`, 50 iters |
| Run env | `adb shell` foreground, no model loaded, no other workload |
| Timing | `std::chrono::steady_clock`, ms resolution |

Source: `bench_mobile.cpp` (kept under `/tmp/mobile_bench/` on the dev box,
push to `/data/local/tmp/bench_mobile` on device).

## Re-running

```sh
NDK=/home/home/Android/Sdk/ndk/27.3.13750724
TC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin
$TC/aarch64-linux-android29-clang++ -O3 -std=c++17 -march=armv8.4-a+fp16 \
    -static-libstdc++ /tmp/mobile_bench/bench_mobile.cpp \
    -o /tmp/mobile_bench/bench_mobile -ldl
adb push /tmp/mobile_bench/bench_mobile /data/local/tmp/
adb shell chmod +x /data/local/tmp/bench_mobile
adb shell /data/local/tmp/bench_mobile
```
