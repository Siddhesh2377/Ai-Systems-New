# Contributing to Ai-Systems

Thanks for your interest in contributing! This project provides on-device AI SDKs for Android, and we welcome contributions of all kinds.

## Getting Started

1. Fork the repo and clone it locally
2. Open in Android Studio (Ladybug or newer)
3. Sync Gradle — native C++ builds automatically via CMake
4. Pick a module to work on:
   - `gguf_lib/` — LLM inference (llama.cpp + JNI)
   - `ai_sd/` — Image generation (QNN/MNN + JNI)
   - `ai_supertonic_tts/` — Text-to-speech (ONNX Runtime + JNI)

## Requirements

- Android Studio with NDK 27.3+
- CMake 3.31.4
- JDK 17
- Target device: `arm64-v8a` (physical device recommended for native testing)

## What to Work On

Check the [Issues](https://github.com/Siddhesh2377/Ai-Systems-New/issues) tab for open tasks. Good first issues are labeled `good first issue`.

**Areas where help is needed:**
- Wiring C++ image processing modules (segmentation, inpainting, depth, style transfer) to JNI
- Performance optimization on specific SoCs
- Test coverage
- Documentation improvements

## How to Submit Changes

1. Create a branch from `master`: `git checkout -b feat/your-feature`
2. Make your changes in small, focused commits
3. Test on a physical ARM64 device if possible
4. Open a PR against `master` with a clear description

## Code Style

- **Kotlin**: Follow standard Kotlin conventions, no wildcard imports
- **C++**: Match existing style — `snake_case` functions, `camelCase` locals, `SD_LOG_*` macros for logging
- **JNI**: Keep native method signatures stable — changing them breaks the C++ side
- **No unnecessary dependencies** — each module should stay lean

## Module Independence

Each SDK module is independent. They share no native dependencies and can be included individually by consumers. Keep it that way.

## Questions?

Open an issue or start a discussion. We're happy to help you get oriented.
