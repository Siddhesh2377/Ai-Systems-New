# ai_sd — Image Generation & Processing SDK

On-device Stable Diffusion and AI image processing for Android.

**Package**: `com.dark.ai_sd`
**ABI**: `arm64-v8a`
**Min SDK**: 27

---

## Backends

| Backend | Hardware | Precision | Models |
|---------|----------|-----------|--------|
| QNN (Hexagon HTP) | Snapdragon 8 Gen 1+ | W8A16 | QNN-compiled .bin |
| MNN (CPU) | Any ARM64 | FP32 | .mnn converted models |
| MNN + OpenCL | Adreno GPU | FP16 | .mnn converted models |

---

## Quick Start

```kotlin
val sdManager = StableDiffusionManager.getInstance(context)

// Load model
sdManager.loadModel(DiffusionModelConfig(
    name = "SD 1.5",
    modelDir = "/path/to/model/",
    runOnCpu = true
))

// Generate
sdManager.generateImage(DiffusionGenerationParams(
    prompt = "a photo of a cat",
    steps = 28,
    width = 512,
    height = 512
))

// Observe state
sdManager.diffusionGenerationState.collect { state ->
    when (state) {
        is DiffusionGenerationState.Progress -> updateUI(state.progress)
        is DiffusionGenerationState.Complete -> showImage(state.bitmap)
        is DiffusionGenerationState.Error -> showError(state.message)
        else -> {}
    }
}
```

---

## Generation API

### StableDiffusionManager (High-Level)

```kotlin
// Lifecycle
loadModel(config: DiffusionModelConfig, width: Int = 512, height: Int = 512): Boolean
cleanup()

// Generation
generateImage(params: DiffusionGenerationParams)
cancelGeneration()
resetGenerationState()

// State flows
diffusionBackendState: StateFlow<DiffusionBackendState>
diffusionGenerationState: StateFlow<DiffusionGenerationState>
isGenerating: StateFlow<Boolean>

// Upscaler
loadUpscaler(modelPath: String, useMnn: Boolean = true, useOpenCL: Boolean = false): Boolean
upscaleImage(inputBitmap: Bitmap)
releaseUpscaler()
upscaleState: StateFlow<UpscaleState>
```

### SDNativeLib (Low-Level JNI)

```kotlin
nativeInitRuntime(qnnLibDir: String): Boolean
nativeLoadModel(clipPath, unetPath, vaeDecoderPath, vaeEncoderPath,
                tokenizerPath, safetyCheckerPath, patchPath, modelDir,
                qnnBackendPath, qnnSystemLibPath, textEmbeddingSize,
                runOnCpu, useCpuClip, isPony, useSafetyChecker): Boolean
nativeGenerate(prompt, negativePrompt, steps, cfgScale, seed,
               width, height, scheduler, useOpenCL, inputImage,
               mask, denoiseStrength, showProcess, showStride,
               callback: SDCallback): Boolean
nativeStopGeneration()
nativeRelease(): Boolean
nativeGetModelInfo(): String
nativeLoadUpscaler(modelPath, useMnn, useOpenCL): Boolean
nativeUpscaleImage(inputRgb, width, height, callback): Boolean
nativeReleaseUpscaler()
```

### SDCallback

```kotlin
interface SDCallback {
    fun onProgress(step: Int, totalSteps: Int)
    fun onImageProgress(step: Int, totalSteps: Int, rgbData: ByteArray, width: Int, height: Int)
    fun onComplete(rgbData: ByteArray, width: Int, height: Int, seed: Long, generationTimeMs: Int)
    fun onError(message: String)
}
```

---

## Generation Modes

| Mode | Inputs | Use Case |
|------|--------|----------|
| txt2img | prompt | Generate from text |
| img2img | prompt + image | Transform existing image |
| inpainting | prompt + image + mask | Edit specific regions |

### Schedulers

| Scheduler | Quality | Speed |
|-----------|---------|-------|
| `dpm` (DPM-Solver++) | Higher | ~28 steps |
| `euler_a` (Euler Ancestral) | Good | ~20 steps |

---

## Image Processing Modules

C++ implementations ready for MNN-converted models:

| Module | Class | Model | Speed |
|--------|-------|-------|-------|
| **4x Super-Resolution** | `upscaler/` | Real-ESRGAN x4v3 (17MB) | 4-8ms/tile |
| **Segmentation** | `segmenter/` | MobileSAM TinyViT (~50MB) | ~12ms/query |
| **Object Removal** | `lama_inpainter/` | LaMa-Dilated (~100MB) | 100-300ms |
| **Depth Estimation** | `depth_estimator/` | MiDaS v2.1 / DepthAnything (25-50MB) | 15-60ms |
| **Style Transfer** | `style_transfer/` | AdaIN arbitrary (10-70MB) | 30-60 FPS |

### Combo Pipelines

- **Smart Object Removal**: tap -> MobileSAM (12ms) -> LaMa (200ms) -> upscale (8ms) = ~220ms
- **Background Swap**: MobileSAM foreground + SD-generated background
- **Depth Bokeh**: depth estimation -> selective blur by depth
- **AI Photo Enhance**: depth + bokeh + upscale

---

## Data Classes

```kotlin
data class DiffusionModelConfig(
    val name: String,
    val modelDir: String,
    val textEmbeddingSize: Int = 768,
    val runOnCpu: Boolean = false,
    val useCpuClip: Boolean = false,
    val isPony: Boolean = false,
    val safetyMode: Boolean = false
)

data class DiffusionGenerationParams(
    val prompt: String,
    val negativePrompt: String = "",
    val steps: Int = 28,
    val cfgScale: Float = 7f,
    val seed: Long? = null,
    val width: Int = 512,
    val height: Int = 512,
    val scheduler: String = "dpm",
    val useOpenCL: Boolean = false,
    val inputImage: String? = null,
    val mask: String? = null,
    val denoiseStrength: Float = 0.6f,
    val showDiffusionProcess: Boolean = false,
    val showDiffusionStride: Int = 1
)

// State sealed classes: DiffusionBackendState, DiffusionGenerationState,
// DiffusionGenerationResult, UpscaleState
```

---

## Build

```
NDK: 27.3.13750724
CMake: 3.31.4
ABI: arm64-v8a (16KB page alignment)
C++: -O3 -ffast-math -march=armv8.2-a+dotprod+fp16 -flto=thin
R8: Enabled (consumer-rules.pro auto-applied to consumers)
```

Dependencies: MNN (static), QNN SDK (dlopen at runtime), tokenizers-cpp, xtensor, stb, zstd, nlohmann/json.
