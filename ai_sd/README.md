# ai_sd

On-device Stable Diffusion inference and image processing for Android.

Runs entirely on-device with no network dependency. Supports Qualcomm NPU
acceleration via QNN and cross-device CPU/GPU inference via MNN.

## Backends

| Backend | Hardware | Models |
|---------|----------|--------|
| QNN (Hexagon HTP) | Snapdragon 8 Gen 1+ | Pre-compiled `.bin` contexts |
| MNN (CPU) | Any ARM64 | `.mnn` converted models |
| MNN + OpenCL | Adreno/Mali GPU | `.mnn` converted models |

## Usage

```kotlin
val sd = StableDiffusionManager.getInstance(context)

// 1. Initialize runtime (extracts QNN libs from assets)
sd.initialize()

// 2. Load model
sd.loadModel(DiffusionModelConfig(
    name = "SD 1.5",
    modelDir = "/path/to/model/",
    runOnCpu = true
))

// 3. Generate
sd.generateImage(DiffusionGenerationParams(
    prompt = "a photo of a cat",
    steps = 28,
    width = 512,
    height = 512
))

// 4. Observe results
sd.diffusionGenerationState.collect { state ->
    when (state) {
        is DiffusionGenerationState.Progress -> updateUI(state.progress)
        is DiffusionGenerationState.Complete -> showImage(state.bitmap)
        is DiffusionGenerationState.Error -> showError(state.message)
        else -> {}
    }
}
```

## LoRA

Runtime LoRA support for MNN/CPU mode. Requires the base `.safetensors` file
in the model directory alongside the `.mnn` files.

```kotlin
// Apply a LoRA
sd.applyLora(LoRAConfig(
    path = "/path/to/lora.safetensors",
    weight = 0.8f
))

// Stack multiple LoRAs
sd.applyLora(LoRAConfig(path = "/path/to/another.safetensors", weight = 0.5f))

// Observe state
sd.loraState.collect { state ->
    when (state) {
        is LoRAState.Applying -> showProgress()
        is LoRAState.Applied -> showActive(state.loras)
        is LoRAState.Error -> showError(state.message)
        else -> {}
    }
}

// Clear all LoRAs (restores original weights)
sd.clearLora()
```

LoRA application regenerates CLIP and UNet weights from the base model, which
takes 30-90 seconds depending on storage speed. Generation after that is
unchanged. QNN mode does not support LoRA.

## Generation Modes

| Mode | Inputs | Description |
|------|--------|-------------|
| txt2img | prompt | Generate from text |
| img2img | prompt + image | Transform existing image |
| inpainting | prompt + image + mask | Edit masked regions |

Schedulers: `dpm` (DPM-Solver++, ~28 steps) and `euler_a` (Euler Ancestral, ~20 steps).

## Image Processing

Five standalone modules, each with load/process/release lifecycle:

| Module | Model | Typical Latency |
|--------|-------|-----------------|
| 4x Upscaler | Real-ESRGAN | ~2s per image |
| Segmentation | MobileSAM | ~12ms per query |
| Inpainting | LaMa | 100-300ms |
| Depth Estimation | MiDaS / DepthAnything | 15-60ms |
| Style Transfer | AdaIN | 30-60ms |

All modules follow the same pattern:

```kotlin
sd.loadSegmenter(encoderPath, decoderPath)
sd.segmenterEncodeImage(bitmap)
sd.segmentAtPoint(x, y)
sd.segmenterState.collect { /* SegmenterState.Complete has mask bitmap */ }
sd.releaseSegmenter()
```

## State Flows

All operations expose reactive state via `StateFlow`:

- `diffusionBackendState` -- Idle, Starting, Running, Error
- `diffusionGenerationState` -- Idle, Progress, Complete, Error
- `isGenerating` -- boolean guard
- `loraState` -- None, Applying, Applied, Error
- `upscaleState` -- Idle, Processing, Complete, Error
- `segmenterState`, `lamaState`, `depthState`, `styleState`

## Build

- NDK 27.3, CMake 3.31.4, arm64-v8a only
- C++17 with `-O3 -ffast-math -march=armv8.2-a+dotprod+fp16 -flto=thin`
- 16KB page alignment for Android 15+ compatibility
- R8 minification enabled, consumer ProGuard rules included

Native dependencies (statically linked): MNN, tokenizers-cpp, xtensor, zstd,
nlohmann/json, stb. QNN SDK loaded at runtime via `dlopen`.

## License

Proprietary. Internal use only.
