# C++ Restructuring Plan for JNI-based Stable Diffusion

## Overview
Transform the monolithic 2000-line HTTP server into a clean, modular JNI library with single-threaded execution, per-request configuration, and progress callbacks.

---

## File Structure

```
src/
├── jni/
│   ├── StableDiffusionJNI.cpp
│   ├── StableDiffusionJNI.h
│   ├── JNITypes.cpp
│   └── JNITypes.h
│
├── core/
│   ├── ModelManager.cpp
│   ├── ModelManager.h
│   ├── GenerationConfig.cpp
│   └── GenerationConfig.h
│
├── models/
│   ├── QnnModelLoader.cpp
│   ├── QnnModelLoader.h
│   ├── MnnModelLoader.cpp
│   ├── MnnModelLoader.h
│   ├── ModelCache.cpp
│   └── ModelCache.h
│
├── inference/
│   ├── ImageGenerator.cpp
│   ├── ImageGenerator.h
│   ├── InferenceContext.cpp
│   ├── InferenceContext.h
│   ├── Upscaler.cpp
│   └── Upscaler.h
│
├── processing/
│   ├── TextProcessor.cpp
│   ├── TextProcessor.h
│   ├── PromptProcessor.cpp
│   ├── PromptProcessor.h
│   ├── VAEProcessor.cpp
│   ├── VAEProcessor.h
│   ├── ImageProcessor.cpp
│   └── ImageProcessor.h
│
├── schedulers/
│   ├── SchedulerFactory.cpp
│   ├── SchedulerFactory.h
│   ├── DPMSchedulerWrapper.cpp
│   ├── DPMSchedulerWrapper.h
│   ├── EulerSchedulerWrapper.cpp
│   └── EulerSchedulerWrapper.h
│
├── utils/
│   ├── TilingUtils.cpp
│   ├── TilingUtils.h
│   ├── PatchUtils.cpp
│   ├── PatchUtils.h
│   ├── BlendingUtils.cpp
│   ├── BlendingUtils.h
│   ├── SafetyChecker.cpp
│   └── SafetyChecker.h
│
└── common/
    ├── Types.h
    ├── Constants.h
    └── Logger.h
```

---

## Detailed Component Breakdown

### 1. JNI Layer

#### `jni/StableDiffusionJNI.h`
```cpp
Purpose: Main JNI interface - entry points called from Java
Responsibilities:
  - JNI_OnLoad / JNI_OnUnload
  - Model initialization: Java_*_initializeModels()
  - Image generation: Java_*_generateImage()
  - Upscaling: Java_*_upscaleImage()
  - Model cleanup: Java_*_releaseModels()
  - Progress callback registration: Java_*_setProgressCallback()

Dependencies: ModelManager, JNITypes
Globals: None (uses handles)
Thread Safety: Single-threaded (enforced by Java layer)
```

#### `jni/JNITypes.h` & `JNITypes.cpp`
```cpp
Purpose: Conversion between JNI types and C++ types
Responsibilities:
  - Convert jobject to GenerationConfig
  - Convert jbyteArray to std::vector<uint8_t>
  - Convert std::vector<uint8_t> to jbyteArray
  - Convert jstring to std::string
  - Error handling and JNI exception throwing
  - Progress callback wrapper (stores JavaVM*, jobject, jmethodID)

Key Classes:
  - JNIProgressCallback (wraps Java callback)
  - JNIImageData (manages byte array lifecycle)
  - JNIStringGuard (RAII for jstring)

Dependencies: None (pure JNI utilities)
```

---

### 2. Core Management

#### `core/ModelManager.h` & `ModelManager.cpp`
```cpp
Purpose: Central model lifecycle manager (singleton or instance-based)
Responsibilities:
  - Initialize QNN system (getQnnSystemFunctionPointers)
  - Load and cache model instances
  - Provide access to models (CLIP, UNET, VAE, Safety, Upscaler)
  - Handle model patching (UNET patch buffer)
  - Cleanup all models on destruction

Key Methods:
  - initialize(paths, config) -> handle
  - getClipModel() -> QnnModel* or MnnModel*
  - getUnetModel() -> QnnModel*
  - getVAEDecoder() -> QnnModel*
  - getVAEEncoder() -> QnnModel* (nullable)
  - getSafetyChecker() -> MNN::Interpreter* (nullable)
  - loadUpscaler(path) -> Model* (dynamic load)
  - release()

State:
  - All model instances (unique_ptr)
  - QNN function pointers (g_qnnSystemFuncs)
  - Backend path (g_backendPathCmd)
  - Tokenizer (shared_ptr)
  - PromptProcessor instance
  - CLIP v2 embeddings (pos_emb, token_emb)
  - Configuration flags (use_mnn, ponyv55, etc.)

Dependencies: QnnModelLoader, MnnModelLoader, TextProcessor
Thread Safety: Single-threaded (no locks needed)
```

#### `core/GenerationConfig.h` & `GenerationConfig.cpp`
```cpp
Purpose: Encapsulate all generation parameters
Responsibilities:
  - Store all request parameters in one struct
  - Validate parameters
  - Provide defaults

Fields:
  - prompt, negative_prompt
  - steps, cfg, seed
  - width, height
  - scheduler_type ("dpm", "euler_a")
  - denoise_strength
  - use_opencl, use_mnn, use_mnn_clip, use_clip_v2
  - ponyv55, use_safety_checker
  - nsfw_threshold
  - img2img data (image_data, mask_data, mask_data_full)
  - has_img2img, has_mask

Methods:
  - validate() -> bool
  - getTextEmbeddingSize() -> int
  - getSampleSize() -> pair<int, int>

Dependencies: None (pure data struct)
```

---

### 3. Model Loading

#### `models/QnnModelLoader.h` & `QnnModelLoader.cpp`
```cpp
Purpose: Load and initialize QNN models
Responsibilities:
  - Create QnnModel instances
  - Initialize QNN backend
  - Handle buffer-based loading (for patched models)
  - Execute initialization sequence (from initializeQnnApp)

Key Methods:
  - loadFromFile(path, name, qnnFuncs, backendPath) -> unique_ptr<QnnModel>
  - loadFromBuffer(buffer, size, name, qnnFuncs, backendPath) -> unique_ptr<QnnModel>
  - initializeModel(model, name) -> StatusCode

Dependencies: QnnModel, DynamicLoadUtil
Extracted From: Lines 360-400 (createQnnModel, initializeQnnApp)
```

#### `models/MnnModelLoader.h` & `MnnModelLoader.cpp`
```cpp
Purpose: Load and initialize MNN models
Responsibilities:
  - Create MNN::Interpreter instances
  - Create MNN sessions with proper config
  - Handle CPU vs OpenCL configuration
  - Manage cache files for OpenCL

Key Methods:
  - loadInterpreter(path) -> MNN::Interpreter*
  - createSession(interpreter, useOpenCL, cacheFile) -> MNN::Session*
  - configureForInference(session, interpreter, inputDims)

Dependencies: MNN headers
Extracted From: Lines scattered throughout (MNN::Interpreter::createFromFile calls)
```

#### `models/ModelCache.h` & `ModelCache.cpp`
```cpp
Purpose: Cache dynamically loaded models (upscalers)
Responsibilities:
  - LRU cache for upscaler models
  - Automatic cleanup of unused models
  - Reference counting

Key Methods:
  - getOrLoad(path) -> Model*
  - release(path)
  - clear()

Dependencies: QnnModelLoader, MnnModelLoader
Note: Optional - can start without this, load upscaler per-request
```

---

### 4. Inference Pipeline

#### `inference/ImageGenerator.h` & `ImageGenerator.cpp`
```cpp
Purpose: Main image generation pipeline
Responsibilities:
  - Orchestrate entire diffusion process
  - CLIP text encoding
  - UNET denoising loop
  - VAE decoding
  - Img2Img support with masking
  - Progress callback invocation
  - Safety checking (optional)

Key Methods:
  - generate(config, models, progressCallback) -> GenerationResult
  
Internal Steps:
  1. Process prompts (via TextProcessor)
  2. CLIP inference (QNN or MNN)
  3. Initialize scheduler
  4. VAE encode (if img2img)
  5. UNET denoising loop
  6. VAE decode (with tiling if needed)
  7. Blend with mask (if inpainting)
  8. Safety check
  9. Return result

Dependencies: TextProcessor, VAEProcessor, SchedulerFactory, SafetyChecker
Extracted From: Lines 1150-1500 (generateImage function)
Thread Safety: Single-threaded
```

#### `inference/InferenceContext.h` & `InferenceContext.cpp`
```cpp
Purpose: Hold temporary state during inference
Responsibilities:
  - Manage latents tensors
  - Store intermediate results
  - Handle batch operations (uncond + cond)

Fields:
  - current_latents
  - timesteps
  - noise
  - batch_size
  - sample dimensions

Methods:
  - initializeLatents(seed, scheduler)
  - scaleLatents(scheduler, timestep)
  - applyNoise(scheduler, timestep, strength)

Dependencies: Scheduler, xtensor
Extracted From: Inline variables in generateImage
```

#### `inference/Upscaler.h` & `Upscaler.cpp`
```cpp
Purpose: Image upscaling with tiling
Responsibilities:
  - Tile-based upscaling (192x192 -> 768x768)
  - Handle both QNN and MNN upscaler models
  - Variable overlap calculation
  - Tile blending
  - Pre/post resize for small images

Key Methods:
  - upscale(imageData, width, height, model, useOpenCL) -> UpscaleResult
  - upscaleWithQNN(data, w, h, model) -> xarray
  - upscaleWithMNN(data, w, h, modelPath, useOpenCL) -> xarray

Dependencies: TilingUtils, BlendingUtils, QnnModel, MNN
Extracted From: Lines 930-1050 (upscaleImageWithModel, upscaleImageWithMNN)
```

---

### 5. Processing Modules

#### `processing/TextProcessor.h` & `TextProcessor.cpp`
```cpp
Purpose: Process prompts into embeddings
Responsibilities:
  - Tokenization
  - Weighted prompt parsing
  - CLIP v1 (token IDs) vs v2 (pre-computed embeddings)
  - Embedding lookup and weighting
  - Handle textual inversions

Key Methods:
  - processPromptPair(positive, negative, maxLen) -> ProcessedPromptPair
  - processWeightedPrompt(prompt, maxLen) -> ProcessedPrompt

Dependencies: tokenizers_cpp, PromptProcessor
Extracted From: Lines 700-850 (processWeightedPrompt, processPromptPair)
```

#### `processing/PromptProcessor.h` & `PromptProcessor.cpp`
```cpp
Purpose: Parse prompt syntax (weights, embeddings)
Responsibilities:
  - Already exists in codebase (from includes)
  - Load embedding files
  - Parse weighted syntax: (text:1.2), [text:0.8], <embedding>

Methods:
  - loadEmbeddings(dir)
  - process(prompt) -> vector<Token>

Dependencies: None (standalone)
Note: Keep as-is, just extract to separate file
```

#### `processing/VAEProcessor.h` & `VAEProcessor.cpp`
```cpp
Purpose: VAE encoding and decoding with tiling support
Responsibilities:
  - Encode images to latents (with tiling for >512px)
  - Decode latents to images (with tiling for >512px)
  - Tile position calculation
  - Tile blending (encoder: mean/std, decoder: pixel)

Key Methods:
  - encode(imageData, width, height, vaeEncoder, config) -> LatentData
  - decode(latents, width, height, vaeDecoder, config) -> ImageData
  - encodeTiled(...) -> LatentData
  - decodeTiled(...) -> ImageData

Dependencies: TilingUtils, BlendingUtils
Extracted From: 
  - Lines 1350-1450 (VAE encode with tiling)
  - Lines 1650-1750 (VAE decode with tiling)
```

#### `processing/ImageProcessor.h` & `ImageProcessor.cpp`
```cpp
Purpose: Image manipulation utilities
Responsibilities:
  - Decode images (JPEG/PNG) from bytes
  - Encode images to JPEG
  - Resize images (pre/post processing)
  - Format conversions (RGB <-> tensor)
  - Normalization

Key Methods:
  - decodeImage(bytes, width, height) -> vector<uint8_t>
  - encodeJPEG(rgb, width, height, quality) -> vector<uint8_t>
  - resizeImageToMinSize(...)
  - resizeImageToTarget(...)
  - rgbToTensor(data, width, height) -> xarray
  - tensorToRGB(tensor) -> vector<uint8_t>

Dependencies: stb_image / libjpeg (from SDUtils.hpp)
Extracted From: Inline image processing code
```

---

### 6. Schedulers

#### `schedulers/SchedulerFactory.h` & `SchedulerFactory.cpp`
```cpp
Purpose: Create scheduler instances
Responsibilities:
  - Factory pattern for schedulers
  - Configure scheduler based on type string

Methods:
  - createScheduler(type, steps, ponyv55) -> unique_ptr<Scheduler>

Supported Types:
  - "dpm" -> DPMSolverMultistepScheduler
  - "euler_a" / "eulera" -> EulerAncestralDiscreteScheduler

Dependencies: DPMSchedulerWrapper, EulerSchedulerWrapper
Extracted From: Lines 1260-1275
```

#### `schedulers/DPMSchedulerWrapper.h` & `DPMSchedulerWrapper.cpp`
```cpp
Purpose: Wrap DPMSolverMultistepScheduler
Responsibilities:
  - Thin wrapper around existing DPMSolverMultistepScheduler
  - Common interface with Euler

Dependencies: DPMSolverMultistepScheduler.hpp
Note: Existing class, just wrap if needed
```

#### `schedulers/EulerSchedulerWrapper.h` & `EulerSchedulerWrapper.cpp`
```cpp
Purpose: Wrap EulerAncestralDiscreteScheduler
Responsibilities:
  - Thin wrapper around existing EulerAncestralDiscreteScheduler
  - Common interface with DPM

Dependencies: EulerAncestralDiscreteScheduler.hpp
Note: Existing class, just wrap if needed
```

---

### 7. Utilities

#### `utils/TilingUtils.h` & `TilingUtils.cpp`
```cpp
Purpose: Calculate tile positions and overlaps
Responsibilities:
  - Generic tile position calculator
  - VAE-specific tile position calculator
  - Overlap calculation

Key Methods:
  - calculateTilePositions(dimension, tileSize, minOverlap) -> vector<int>
  - calculateVAETilePositions(pixelWidth, pixelHeight) 
      -> tuple<pixelPos, latentPos, pixelOverlap, latentOverlap>

Dependencies: None
Extracted From: Lines 890-930
```

#### `utils/BlendingUtils.h` & `BlendingUtils.cpp`
```cpp
Purpose: Tile blending algorithms
Responsibilities:
  - Blend VAE encoder tiles (mean/std fusion)
  - Blend VAE decoder tiles (pixel fusion)
  - Blend upscaler tiles
  - Laplacian pyramid blending (inpainting)

Key Methods:
  - blendVAEEncoderTiles(...) -> xarray
  - blendVAEOutputTiles(...) -> xarray
  - blendUpscalerTiles(...) -> xarray
  - laplacianPyramidBlend(...) -> xarray

Dependencies: xtensor, LaplacianBlend.hpp
Extracted From: Lines 850-890, and LaplacianBlend code
```

#### `utils/PatchUtils.h` & `PatchUtils.cpp`
```cpp
Purpose: Apply ZSTD patches to models
Responsibilities:
  - Read patch files
  - Apply ZSTD decompression with dictionary
  - Create in-memory patched buffers
  - Clean up old patch files

Key Methods:
  - applyPatchToBuffer(oldPath, patchPath) -> unique_ptr<PatchedModelBuffer>
  - applyPatchToFile(oldPath, patchPath, newPath) -> int
  - cleanupOldPatches(directory)

Dependencies: zstd library
Extracted From: Lines 200-350
```

#### `utils/SafetyChecker.h` & `SafetyChecker.cpp`
```cpp
Purpose: NSFW content detection
Responsibilities:
  - Run safety checker model on generated images
  - Return NSFW score
  - Optional: blank image if score > threshold

Key Methods:
  - check(imageData, width, height, interpreter, session) -> float
  - checkAndCensor(imageData, width, height, threshold, ...) -> bool

Dependencies: MNN, safety_check function from SDUtils
Extracted From: Lines 1780-1800 (inline safety check)
```

---

### 8. Common Definitions

#### `common/Types.h`
```cpp
Purpose: Shared type definitions
Contents:
  - GenerationResult struct
  - LatentData struct
  - ImageData struct
  - UpscaleResult struct
  - ProcessedPrompt struct
  - ProcessedPromptPair struct
  - PatchedModelBuffer struct
  - Model handle types

Dependencies: None
```

#### `common/Constants.h`
```cpp
Purpose: Shared constants
Contents:
  - VAE_TILE_SIZE = 512
  - VAE_LATENT_TILE_SIZE = 64
  - MIN_LATENT_OVERLAP = 16
  - UPSCALE_TILE_SIZE = 192
  - UPSCALE_OUTPUT_TILE_SIZE = 768
  - UPSCALE_MIN_OVERLAP = 12
  - VAE_SCALE_FACTOR = 8
  - UPSCALE_FACTOR = 4
  - DEFAULT_STEPS = 20
  - DEFAULT_CFG = 7.5
  - etc.

Dependencies: None
```

#### `common/Logger.h`
```cpp
Purpose: Logging macros/utilities
Contents:
  - Wrapper around QNN_INFO, QNN_ERROR, QNN_WARN
  - Optional: Add JNI-friendly logging (callback to Java)

Dependencies: QNN Logger
```

---

## Migration Strategy

### Phase 1: Extract Utilities (Low Risk)
**Goal:** No behavioral changes, just code organization

1. Create `utils/TilingUtils.*` - Move `calculate_tile_positions`, `calculate_vae_tile_positions`
2. Create `utils/BlendingUtils.*` - Move `blend_vae_encoder_tiles`, `blend_vae_output_tiles`
3. Create `utils/PatchUtils.*` - Move patch application code
4. Create `utils/SafetyChecker.*` - Move safety check logic
5. Create `common/Types.h` - Move all struct definitions
6. Create `common/Constants.h` - Extract magic numbers

**Validation:** Compile and run with HTTP server still working

---

### Phase 2: Extract Processing Modules
**Goal:** Isolate processing logic from inference

1. Create `processing/PromptProcessor.*` - Already exists, just move to new location
2. Create `processing/TextProcessor.*` - Move `processWeightedPrompt`, `processPromptPair`
3. Create `processing/ImageProcessor.*` - Move image decode/encode/resize
4. Create `processing/VAEProcessor.*` - Move VAE encode/decode with tiling

**Validation:** Compile and run with HTTP server still working

---

### Phase 3: Extract Schedulers
**Goal:** Clean up scheduler creation

1. Create `schedulers/SchedulerFactory.*` - Move scheduler creation logic
2. Create `schedulers/DPMSchedulerWrapper.*` (if needed)
3. Create `schedulers/EulerSchedulerWrapper.*` (if needed)

**Validation:** Compile and run with HTTP server still working

---

### Phase 4: Extract Model Management
**Goal:** Centralize all model lifecycle

1. Create `core/GenerationConfig.*` - Move all config fields
2. Create `models/QnnModelLoader.*` - Extract QNN loading
3. Create `models/MnnModelLoader.*` - Extract MNN loading
4. Create `core/ModelManager.*` - Move all global model instances here
    - Convert globals to class members
    - Keep singleton instance for now
    - HTTP server uses ModelManager::getInstance()

**Validation:** Compile and run with HTTP server still working

---

### Phase 5: Extract Inference Pipeline
**Goal:** Isolate generation logic

1. Create `inference/InferenceContext.*` - Move intermediate state
2. Create `inference/Upscaler.*` - Move upscaling logic
3. Create `inference/ImageGenerator.*` - Move `generateImage` function
    - Convert to class method
    - Take ModelManager& and GenerationConfig as params
    - Return GenerationResult

**Validation:** Compile and run with HTTP server still working
**At this point:** Everything is modular, HTTP server is just a thin wrapper

---

### Phase 6: Add JNI Layer
**Goal:** Create JNI interface without breaking HTTP

1. Create `jni/JNITypes.*` - JNI conversion utilities
2. Create `jni/StableDiffusionJNI.*` - JNI entry points
    - Parallel to HTTP server, doesn't replace it yet
    - Uses same ModelManager instance

**Validation:** Test JNI calls from Java, HTTP still works

---

### Phase 7: Remove HTTP Server
**Goal:** Final cleanup

1. Delete HTTP server code from main.cpp
2. Delete `httplib.h` dependency
3. Remove global request state (prompt, steps, etc.)
4. Keep command-line parsing for standalone testing (optional)
5. Or remove main() entirely if pure library

**Validation:** JNI tests pass, library can be loaded from Java