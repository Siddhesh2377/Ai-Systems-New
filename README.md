# Ai-Systems

Android AI SDK monorepo — on-device LLM inference, image generation, and text-to-speech. All SDKs run entirely on-device with no cloud dependencies.

Built for **Android** (ARMv8/ARMv9 via NDK) with JNI + native C++ backends.

---

## Modules

| Module | What it does | Backend | Package |
|--------|-------------|---------|---------|
| **[ai_gguf](ai_gguf/)** | LLM inference (chat, embeddings, tool calling) | llama.cpp (custom fork) | `com.mp.ai_gguf` |
| **[ai_sd](ai_sd/)** | Image generation (txt2img, img2img, inpaint) | QNN (Hexagon DSP) + MNN | `com.dark.ai_sd` |
| **[ai_supertonic_tts](ai_supertonic_tts/)** | Text-to-speech (5 languages, 10 voices) | ONNX Runtime | `com.mp.ai_supertonic_tts` |

---

## Quick Setup

### Gradle

```kotlin
// settings.gradle.kts
include(":ai_gguf")
include(":ai_sd")
include(":ai_supertonic_tts")

// app/build.gradle.kts
dependencies {
    implementation(project(":ai_gguf"))         // LLM
    implementation(project(":ai_sd"))           // Image Gen
    implementation(project(":ai_supertonic_tts")) // TTS
}
```

### Requirements

- **Min SDK**: 27 (Android 8.1)
- **Target SDK**: 36
- **ABI**: `arm64-v8a` (all modules), `x86_64` (ai_gguf, ai_supertonic_tts)
- **CMake**: 3.22.1
- **JDK**: 17
- **Gradle**: 9.3.1
- **AGP**: 9.0.1

---

## Module Details

### ai_gguf — LLM Inference

On-device LLM inference powered by a [custom llama.cpp fork](https://github.com/Siddhesh2377/llama.cpp-custom) with 10 runtime intervention surfaces for personality/emotion control.

**Key features**:
- Multi-turn chat with streaming tokens
- Model-agnostic tool calling with GBNF grammar constraints (STRICT/LAZY modes)
- Text embeddings for semantic search
- 7 CPU backend variants selected at runtime (dotprod, fp16, i8mm, sve, sme)
- Character Intelligence Engine: control vectors, attention bias, head rescaling, attention temperature, fast weight memory, LayerNorm shift
- KV cache persistence (save/load conversation state)
- Speculative decoding, forward-only learning (SPSA)

See [ai_gguf/README.md](ai_gguf/README.md) for full API reference.

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

---

## Build

```bash
# Full build (all modules)
./gradlew assembleRelease

# Single module
./gradlew :ai_gguf:assembleRelease
./gradlew :ai_sd:assembleRelease
./gradlew :ai_supertonic_tts:assembleRelease
```

Native C++ is built automatically via CMake during Gradle build. First build takes longer due to llama.cpp compilation.

---

## Architecture

```
Ai-Systems/
├── ai_gguf/           # LLM SDK
│   ├── src/main/cpp/  #   C++ (JNI → llama.cpp)
│   └── src/main/java/ #   Kotlin API
├── ai_sd/             # Image Gen SDK
│   ├── src/main/cpp/  #   C++ (JNI → QNN/MNN)
│   └── src/main/java/ #   Kotlin API
├── ai_supertonic_tts/ # TTS SDK
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
