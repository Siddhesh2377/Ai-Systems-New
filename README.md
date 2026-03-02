# Ai-Systems

On-device AI SDK for Android — LLM inference, image generation, image processing, and text-to-speech. No cloud, no internet, runs entirely on-device.

Built for **Android** (ARMv8/ARMv9 via NDK) with JNI + native C++ backends.

> **Note:** This repo is developed strictly for **[ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)**. If you want to use these SDKs in your own app, fork or clone this repo and integrate the modules you need.

---

## Modules

| Module | What it does | Backend | Package |
|--------|-------------|---------|---------|
| **[gguf_lib](gguf_lib/)** | LLM inference (chat, embeddings, tool calling) | llama.cpp (custom fork) | `com.dark.gguf_lib` |
| **[ai_sd](ai_sd/)** | Image generation (txt2img, img2img, inpaint) | QNN (Hexagon DSP) + MNN | `com.dark.ai_sd` |
| **[ai_supertonic_tts](ai_supertonic_tts/)** | Text-to-speech (5 languages, 10 voices) | ONNX Runtime | `com.mp.ai_supertonic_tts` |
| **[ai_chatterbox](ai_chatterbox/)** | Emotional TTS (voice cloning, expressive speech) | ONNX Runtime | `com.dark.ai_chatterbox` |

---

## Quick Setup

### Gradle

```kotlin
// settings.gradle.kts
include(":gguf_lib")
include(":ai_sd")
include(":ai_supertonic_tts")
include(":ai_chatterbox")

// app/build.gradle.kts
dependencies {
    implementation(project(":gguf_lib"))         // LLM
    implementation(project(":ai_sd"))           // Image Gen
    implementation(project(":ai_supertonic_tts")) // TTS
    implementation(project(":ai_chatterbox"))    // Emotional TTS
}
```

### Requirements

- **Min SDK**: 27 (Android 8.1)
- **Target SDK**: 36
- **ABI**: `arm64-v8a` (all modules)
- **CMake**: 3.31.4
- **JDK**: 17
- **Gradle**: 9.3.1
- **AGP**: 9.0.1

---

## Module Details

### gguf_lib — LLM Inference

On-device LLM inference powered by a [custom llama.cpp fork](https://github.com/Siddhesh2377/llama.cpp-custom) optimized for ARM CPU with KleidiAI micro-kernels.

**Key features**:
- Multi-turn chat with Flow-based streaming tokens
- Model-agnostic tool calling with GBNF grammar constraints (STRICT/LAZY modes)
- Text embeddings for semantic search
- Character personality engine: mood, emotion, uncensored mode via logit/sampling control
- KV cache prefix reuse, context shifting, disk-backed prompt cache
- Speculative decoding (ngram self-speculative)
- CPU affinity pinning, zero-copy token delivery, JNI method ID caching

See [gguf_lib/CLAUDE.md](gguf_lib/CLAUDE.md) for full API reference.

### ai_sd — Image Generation

On-device Stable Diffusion via Qualcomm QNN (Hexagon DSP) or MNN (CPU fallback).

**Key features**:
- txt2img, img2img, inpainting
- QNN acceleration on Snapdragon SoCs (8 Gen 1+)
- LoRA support
- DPM-Solver++ and Euler Ancestral schedulers
- Tiled VAE for high-resolution generation
- Safety checker (optional)

See [ai_sd/README.md](ai_sd/README.md) for full API reference.

### ai_supertonic_tts — Text-to-Speech

On-device TTS using Supertonic v2 (66M params, ONNX Runtime). Produces 44.1 kHz mono audio at up to 167x faster than real-time.

**Key features**:
- 5 languages: English, Korean, Spanish, Portuguese, French
- 10 voice presets (5 female, 5 male)
- Streaming playback via AudioTrack
- Save to WAV/PCM files
- Auto-chunking for long text
- Optional NNAPI GPU/NPU acceleration

See [ai_supertonic_tts/TTS_SDK_DOCS.md](ai_supertonic_tts/TTS_SDK_DOCS.md) for full API reference.

### ai_chatterbox — Emotional Text-to-Speech

On-device emotional TTS via Chatterbox (MIT, [ResembleAI](https://github.com/resemble-ai/chatterbox)). Supports voice cloning from reference audio and emotion control.

**Key features**:
- Two model variants: Turbo (350M, fast) and Original (500M, emotional)
- Voice cloning from 10s reference audio
- Emotion exaggeration control (Original variant)
- 24kHz mono PCM output
- Greedy autoregressive generation with KV cache
- Cancellation support via atomic stop flag

See [ai_chatterbox/README.md](ai_chatterbox/README.md) for full API reference.

---

## Build

```bash
# Full build (all modules)
./gradlew assembleRelease

# Single module
./gradlew :gguf_lib:assembleRelease
./gradlew :ai_sd:assembleRelease
./gradlew :ai_supertonic_tts:assembleRelease
./gradlew :ai_chatterbox:assembleRelease
```

Native C++ is built automatically via CMake during Gradle build. First build takes longer due to llama.cpp compilation.

---

## Architecture

```
Ai-Systems/
├── gguf_lib/          # LLM SDK
│   ├── src/main/cpp/  #   C++ (JNI → llama.cpp)
│   └── src/main/java/ #   Kotlin API
├── ai_sd/             # Image Gen SDK
│   ├── src/main/cpp/  #   C++ (JNI → QNN/MNN)
│   └── src/main/java/ #   Kotlin API
├── ai_supertonic_tts/ # TTS SDK
│   ├── src/main/cpp/  #   C++ (JNI → ONNX Runtime)
│   └── src/main/java/ #   Kotlin API
├── ai_chatterbox/     # Emotional TTS SDK
│   ├── src/main/cpp/  #   C++ (JNI → ONNX Runtime)
│   └── src/main/java/ #   Kotlin API
└── build.gradle.kts   # Root config
```

Each SDK is an independent Android library module with its own JNI layer. They share no native dependencies and can be included individually.

---

## Used By

- **[ToolNeuron](https://github.com/Siddhesh2377/ToolNeuron)** — Android AI assistant with character intelligence, tool calling, image generation, and TTS

---

## License

MIT
