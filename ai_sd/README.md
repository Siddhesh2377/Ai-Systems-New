# ai_sd — Image Generation SDK

On-device Stable Diffusion for Android via Qualcomm QNN (Hexagon DSP) or MNN (CPU fallback).

**Package**: `com.dark.ai_sd`
**ABI**: `arm64-v8a`
**Min SDK**: 27

---

## Core API — `SDNativeLib`

Singleton JNI interface for native image generation.

### Runtime Setup

```kotlin
// Initialize QNN runtime (extract libs first)
SDNativeLib.nativeInitRuntime(qnnLibDir: String)
```

### Model Loading

```kotlin
SDNativeLib.nativeLoadModel(
    clipPath: String,
    unetPath: String,
    vaeDecoderPath: String,
    vaeEncoderPath: String?,    // null if txt2img only
    tokenizerPath: String,
    safetyModelPath: String?,   // null to disable safety checker
    embeddingDir: String?,      // textual inversions directory
    schedulerType: String,      // "dpm" or "euler_a"
    isPony: Boolean,            // Pony Diffusion v5.5 mode
    useSafetyChecker: Boolean
): Boolean

SDNativeLib.nativeRelease()
SDNativeLib.nativeGetModelInfo(): String  // JSON metadata
```

### Image Generation

```kotlin
SDNativeLib.nativeGenerate(
    prompt: String,
    negativePrompt: String,
    steps: Int,              // 20 default
    cfgScale: Float,         // 7.5 default
    seed: Long,              // -1 for random
    width: Int,
    height: Int,
    scheduler: String,       // "dpm" or "euler_a"
    useOpenCL: Boolean,      // GPU acceleration for MNN
    inputImage: ByteArray?,  // null for txt2img, image bytes for img2img
    mask: ByteArray?,        // null for no mask, mask bytes for inpainting
    denoiseStrength: Float,  // 0.0-1.0, only for img2img
    showProcess: Boolean,    // Send intermediate step images
    showStride: Int,         // Every N steps
    callback: SDCallback
): Boolean

SDNativeLib.nativeStopGeneration()
```

### LoRA

```kotlin
SDNativeLib.nativeApplyLora(loraPath: String, weight: Float): Boolean
SDNativeLib.nativeClearLora()
```

### Callback

```kotlin
interface SDCallback {
    fun onProgress(step: Int, totalSteps: Int)
    fun onImageReady(imageData: ByteArray, width: Int, height: Int)
    fun onError(message: String)
}
```

---

## High-Level API — `StableDiffusionManager`

Kotlin singleton facade with state management and coroutine support.

```kotlin
val sdManager = StableDiffusionManager

// Initialize
sdManager.initialize(context)

// Load model
sdManager.loadModel(config: DiffusionModelConfig)

// Generate
sdManager.generateImage(params: DiffusionGenerationParams)

// State observation
sdManager.diffusionGenerationState  // StateFlow
sdManager.diffusionBackendState     // StateFlow
sdManager.isGenerating              // Boolean

// Control
sdManager.cancelGeneration()
sdManager.cleanup()
```

---

## Supported Configurations

### Backends

| Backend | Hardware | Precision | Models |
|---------|----------|-----------|--------|
| QNN (Hexagon DSP) | Snapdragon 8 Gen 1+ | INT8/INT16 | QNN-compiled models |
| MNN (CPU) | Any ARM64 | FP32 | SafeTensor/ONNX models |
| MNN + OpenCL | Adreno GPU | FP16 | SafeTensor/ONNX models |

### Generation Modes

| Mode | Inputs | Use Case |
|------|--------|----------|
| txt2img | prompt only | Generate from text |
| img2img | prompt + image | Transform existing image |
| inpainting | prompt + image + mask | Edit specific regions |

### Schedulers

| Scheduler | Quality | Speed |
|-----------|---------|-------|
| `dpm` (DPM-Solver++) | Higher | Slower |
| `euler_a` (Euler Ancestral) | Good | Faster |

---

## Native Build

```cmake
CMAKE_BUILD_TYPE=Release
-O3 -ffast-math -fno-finite-math-only -ffp-contract=fast
```

Dependencies: QNN SDK libs (extracted from assets at runtime), MNN (statically linked).
