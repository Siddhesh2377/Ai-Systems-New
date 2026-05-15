# Ai-Systems

On-device AI SDK monorepo for Android — LLM/VLM inference, image generation, speech (ASR/TTS), and a shared diagnostic/crash layer. No cloud, no internet, runs entirely on-device.

Built for **Android** (ARMv8 via NDK) with JNI + native C++ backends.

> This repo is developed strictly for **[ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)**. If you want to use these SDKs in your own app, fork or clone and integrate the modules you need.

---

## Modules

| Module | What it does | Backend | Package |
|--------|-------------|---------|---------|
| **[gguf_lib](gguf_lib/)** | LLM / VLM inference, RAG, embeddings, extractive summarization | llama.cpp custom fork + mtmd | `com.dark.gguf_lib` |
| **[ai_sd](ai_sd/)** | Image generation (txt2img, img2img) + 5 image-processing modules (upscale, segment, inpaint, depth, style) | QNN (Hexagon DSP) + MNN | `com.dark.ai_sd` |
| **[ai_sherpa](ai_sherpa/)** | Offline speech-to-text and text-to-speech | sherpa-onnx + ONNX Runtime | `com.dark.ai_sherpa` |
| **[tn_security](tn_security/)** | Unified error / crash / log capture used by every other SDK | C JNI + Kotlin SharedFlow | `com.dark.tn_security` |
| **app** | In-repo test app: SD + VLM demos | — | `com.dark.demon_system` |

**Module list lives in [`settings.gradle.kts`](settings.gradle.kts).** All other docs derive from it.

---

## Quick setup

```kotlin
// settings.gradle.kts
include(":tn_security")
include(":gguf_lib")
include(":ai_sd")
include(":ai_sherpa")

// app/build.gradle.kts
dependencies {
    implementation(project(":gguf_lib"))    // LLM / VLM / RAG
    implementation(project(":ai_sd"))       // Image gen + processing
    implementation(project(":ai_sherpa"))   // ASR + TTS
    // :tn_security is pulled in transitively (api scope) by the SDKs above
}
```

### Requirements
- Min SDK: 29 (Android 10)
- Target SDK: 36
- ABI: `arm64-v8a` (all modules). `ai_sherpa` also ships `armeabi-v7a`.
- NDK: 27.3.13750724 (pinned — default selection often picks the wrong one)
- CMake: 3.31.4 for `gguf_lib` / `ai_sd` / `tn_security`; 3.22.1 OK for `ai_sherpa`
- JDK 17, Gradle 9.3.1, AGP 9.0.1

---

## Module details

### gguf_lib — LLM / VLM / RAG

On-device LLM and VLM inference via a [custom llama.cpp fork](https://github.com/Siddhesh2377/llama.cpp-custom) optimized for ARM CPU with KleidiAI micro-kernels.

Features:
- Multi-turn chat with `Flow`-based streaming tokens; thinking-mode passthrough for reasoning models
- Vision-language models (Qwen-VL, LFM2-VL, MiniCPM, InternVL) via mtmd projector
- **Two disk-backed caches** that compose to drop VLM TTFT from ~18.7 s to hundreds of ms:
  - **VT cache** — vision-token embeddings (skips ViT pass)
  - **VLM-KV cache** — LLM state at post-image boundary (skips ViT AND image-prefill)
- Standalone embedding engine + RAG engine with late chunking + binary quantization index
- Extractive summarization (no model required) via `TextDigest`
- KV cache prefix reuse, StreamingLLM eviction, disk-backed system-prompt cache
- Thermal-adaptive thread scheduling, CPU affinity to P-cores, zero-copy token delivery
- Multi-variant ARM CPU backends (armv8.0 → armv9.2+SME, runtime-selected via `getauxval`)

> **Removed in May 2026:** tool calling, control vectors, personality / mood / uncensored mode. These symbols no longer exist; host code referencing them must be deleted before upgrading the AAR.

See [`gguf_lib/CLAUDE.md`](gguf_lib/CLAUDE.md), [`gguf_lib/README.md`](gguf_lib/README.md), [`gguf_lib/VLM.md`](gguf_lib/VLM.md), [`gguf_lib/DEVICE.md`](gguf_lib/DEVICE.md).

### ai_sd — Image generation + processing

On-device Stable Diffusion via Qualcomm QNN (Hexagon HTP, W8A16) or MNN (CPU/GPU fallback).

Features:
- txt2img, img2img
- LoRA (CPU/MNN path only)
- DPM-Solver++ and Euler Ancestral schedulers
- SoC detection (`/sys/devices/soc0/` + HTP version probe) → variant selection (`8gen1` / `8gen2` / `8gen3` / `min`)
- Persistent UNet + VAE decoder sessions across generations
- 13 performance optimizations (CPU affinity, zero-copy MNN tensors, CLIP embedding cache, etc.)
- Tiled VAE for high-res, safety checker (optional)
- 5 image-processing modules (all JNI-wired):
  - **Upscaler** — Real-ESRGAN x4plus
  - **Segmenter** — MobileSAM tap-to-segment
  - **Inpainter** — LaMa fast object removal
  - **Depth** — MiDaS / DepthAnything monocular depth
  - **Style** — AdaIN arbitrary style transfer

See [`ai_sd/CLAUDE.md`](ai_sd/CLAUDE.md).

### ai_sherpa — Offline ASR + TTS

Wraps [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via ONNX Runtime Android.

Features:
- Offline ASR: Whisper, Paraformer, Transducer, NeMo CTC, TDNN, NeMo Transducer
- Offline TTS: VITS, Kokoro
- 16 kHz / 80-bin Mel feature extraction (configurable)
- `AutoCloseable` handles (forgetting `close()` leaks 50–500 MB)
- Errors emit through the unified `tn_security` event stream

Streaming ASR and VAD were removed in the Apr 2026 cleanup (no consumer used them).

See [`ai_sherpa/README.md`](ai_sherpa/README.md).

### tn_security — Unified diagnostic / crash layer

Every SDK in this repo routes native logs and structured errors through `tn_security`. Provides:
- Signal handlers (SIGSEGV/ABRT/BUS/ILL/FPE) that write JSON crash files using only async-signal-safe syscalls
- 256-entry ring buffer captured into the crash file at signal time
- Structured errors tagged by module / op-id / stage / code, with user-actionable suggestions
- Single Kotlin `SharedFlow<TnEvent>` for all logs / errors / cancellations / crashes
- Lenient UTF-8 JNI decoding (replaces invalid bytes with U+FFFD; required because upstream tokenizer logs aren't strict UTF-8)

See [`tn_security/README.md`](tn_security/README.md).

---

## Build

```bash
# All modules
./gradlew assembleRelease

# Single module
./gradlew :gguf_lib:assembleRelease
./gradlew :ai_sd:assembleRelease
./gradlew :ai_sherpa:assembleRelease
./gradlew :tn_security:assembleRelease

# In-repo test app
./gradlew :app:assembleDebug
```

Native C++ builds automatically via CMake. First build is slow (llama.cpp + MNN + sherpa-onnx all compile from source).

External source checkouts referenced by CMake (adjust per environment):

| Used by | Path |
|---|---|
| `gguf_lib` | `/home/home/dev/include/llama.cpp` |
| `ai_sherpa` | `/home/home/dev/include/sherpa-onnx` |
| `ai_sherpa` | `/home/home/dev/include/ort-android-1.24.3/` (prebuilt) |

---

## Layout

```
Ai-Systems/
├── tn_security/    # Unified diagnostic / crash SDK (foundation)
├── gguf_lib/       # LLM / VLM / RAG / Embedding / TextDigest
├── ai_sd/          # Stable Diffusion + image processing
├── ai_sherpa/      # ASR + TTS
├── app/            # In-repo test app (SD + VLM demos)
├── settings.gradle.kts
├── MODELS.md       # Recommended models per workload
├── SECURITY.md     # Security audit + posture
└── CONTRIBUTING.md
```

Each SDK is an independent Android library module. `:tn_security` is a foundation module pulled in via `api(project(":tn_security"))` from the others; the host app doesn't need to declare it manually.

---

## Used by

- **[ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)** — Android AI assistant: consumes `:gguf_lib`, `:ai_sd`, `:ai_sherpa`, `:tn_security`

---

## License

MIT
