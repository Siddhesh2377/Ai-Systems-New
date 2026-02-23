# ai_gguf — LLM Inference SDK

On-device LLM inference for Android via [llama.cpp](https://github.com/Siddhesh2377/llama.cpp-custom) (custom fork with runtime intervention surfaces).

**Package**: `com.mp.ai_gguf`
**ABI**: `arm64-v8a`, `x86_64`
**Min SDK**: 27

---

## Core API — `GGUFNativeLib`

Singleton JNI interface. All native functions are called through this object.

### Model Loading

```kotlin
// Load from file descriptor (SAF-compatible)
GGUFNativeLib.nativeLoadModelFromFd(
    fd: Int,            // ParcelFileDescriptor.detachFd()
    threads: Int,       // CPU threads (4-8 recommended)
    contextSize: Int,   // KV cache size in tokens (2048-8192)
    batchSize: Int,     // Batch size for prompt processing
    flashAttn: Int,     // 0=disabled, 1=enabled (recommended)
    cacheTypeK: Int,    // GGML type for K cache (9=Q8_0)
    cacheTypeV: Int     // GGML type for V cache (9=Q8_0)
): Boolean

// Unload
GGUFNativeLib.nativeUnloadModel()
```

### Text Generation

```kotlin
// Single-turn streaming
GGUFNativeLib.nativeGenerateStream(
    prompt: String,
    maxTokens: Int,
    callback: StreamCallback
)

// Multi-turn streaming (full conversation history)
GGUFNativeLib.nativeGenerateStreamMultiTurn(
    messagesJson: String,  // JSON array of {role, content}
    maxTokens: Int,
    callback: StreamCallback
): Boolean

// StreamCallback interface
interface StreamCallback {
    fun onToken(token: String)
    fun onToolCall(name: String, argsJson: String)
    fun onDone()
    fun onError(message: String)
}
```

### Embeddings

```kotlin
GGUFNativeLib.nativeLoadEmbeddingModel(fd: Int, threads: Int): Boolean
GGUFNativeLib.nativeEncodeText(text: String): FloatArray?
GGUFNativeLib.nativeReleaseEmbeddingModel()
```

### Sampler Control

```kotlin
// Update sampling parameters at runtime
GGUFNativeLib.nativeUpdateSamplerParams(
    temperature: Float,   // 0.0 = greedy, 0.7 = balanced, 1.0+ = creative
    topK: Int,
    topP: Float,
    minP: Float,
    dryMultiplier: Float, // DRY repetition penalty
    dryBase: Float,
    dryAllowedLength: Int,
    dryPenaltyLastN: Int,
    xtcProbability: Float,
    xtcThreshold: Float
)
```

### KV Cache Persistence

```kotlin
GGUFNativeLib.nativeGetStateSize(): Long
GGUFNativeLib.nativeStateSaveToFile(path: String): Boolean
GGUFNativeLib.nativeStateLoadFromFile(path: String): Boolean
```

### Stop Strings

```kotlin
GGUFNativeLib.nativeSetStopStrings(stopStrings: Array<String>)
```

---

## Tool Calling — `ToolCallManager`

Model-agnostic tool calling with GBNF grammar constraints. Works with any model that has a chat template.

### Setup

```kotlin
val toolManager = ToolCallManager(GGUFNativeLib)

// Define tools with DSL
toolManager.registerTools(
    tool("get_weather", "Get current weather") {
        stringParam("location", "City name", required = true)
        stringParam("units", "Temperature units",
            enum = listOf("celsius", "fahrenheit"))
    },
    tool("set_alarm", "Set an alarm") {
        stringParam("time", "Time in HH:MM", required = true)
        stringParam("label", "Alarm label")
    }
)

// Enable (STRICT = forces JSON, LAZY = model chooses text or tool call)
toolManager.enable(ToolCallingConfig(
    grammarMode = GrammarMode.LAZY,
    maxRounds = 5,
    maxTokensPerTurn = 256
))
```

### Multi-Turn Orchestration

```kotlin
toolManager.generateWithTools(
    userMessage = "What's the weather in Tokyo?",
    executor = { call ->
        when (call.name) {
            "get_weather" -> {
                val city = call.getString("location")
                ToolResult("get_weather", fetchWeather(city))
            }
            else -> ToolResult(call.name, "Unknown tool", isError = true)
        }
    },
    onToken = { token -> print(token) },
    onToolCallDetected = { call -> Log.d("Tool", "Calling ${call.name}") },
    onDone = { response -> Log.d("Done", response) }
)
```

### Grammar Modes

| Mode | Behavior | Use When |
|------|----------|----------|
| `STRICT` | Forces JSON tool call output from first token | You know the request needs a tool |
| `LAZY` | Model freely outputs text or tool call (grammar activates on `{`) | General chat with optional tools |

### ToolCall Access

```kotlin
val call: ToolCall = toolManager.parseToolCall(json)!!
call.getString("location", "Unknown")
call.getInt("count", 0)
call.getBoolean("enabled", false)
call.getDouble("value", 0.0)
call.has("optional_param")
```

---

## Character Intelligence Engine

7-part runtime intervention system for personality/emotion control. All interventions are RAM-only — the GGUF model file is never modified.

### A. Control Vectors (Contrastive Activation Steering)

```kotlin
// Compute direction vectors from positive/negative prompt pairs
GGUFNativeLib.nativeComputePersonalityVectors(
    posPrompt: String,  // "You are happy and cheerful"
    negPrompt: String,  // "You are sad and gloomy"
    axisName: String    // "happiness"
): Boolean

// Load cached direction vectors
GGUFNativeLib.nativeLoadControlVectors(cachePath: String): Boolean

// Apply with emotion-gated blending
GGUFNativeLib.nativeApplyEmotionGatedVectors(
    axisStrengths: FloatArray  // Per-axis strength [-1.0, 1.0]
): Boolean
```

### B. Logit Bias

```kotlin
// Boost/suppress specific tokens
GGUFNativeLib.nativeSetLogitBias(
    tokenIds: IntArray,
    biases: FloatArray  // Positive = boost, negative = suppress
)
```

### C. Attention Bias

```kotlin
// Inject bias on KQ scores pre-softmax
GGUFNativeLib.nativeSetAttentionBias(
    startPos: Int, endPos: Int,
    bias: Float,         // 2.0 = ~7.4x boost, -2.0 = ~0.13x suppress
    layerStart: Int, layerEnd: Int  // -1 = all layers
)
```

### D. Head Rescaling

```kotlin
// Per-head scalar multiplier from probed direction vectors
GGUFNativeLib.nativeProbeAndSetHeadScales()
// Or set manually
GGUFNativeLib.nativeSetHeadScales(layer: Int, scales: FloatArray)
```

### E. Attention Temperature

```kotlin
// Per-layer softmax sharpness profile
GGUFNativeLib.nativeSetAttentionTemperatureProfile(
    earlyTemp: Float,   // 1.3 = sharp (pattern matching)
    midTemp: Float,     // 1.0 = default
    lateTemp: Float     // 0.8 = flat (broader generation)
)
```

### F. Fast Weight Memory (Hopfield-style)

```kotlin
GGUFNativeLib.nativeFastWeightInit(dReduced: Int)  // 128 recommended
GGUFNativeLib.nativeFastWeightUpdate()               // Auto-updates each token
GGUFNativeLib.nativeFastWeightInject(strength: Float)
GGUFNativeLib.nativeFastWeightReset()
```

### G. LayerNorm Affine Shift

```kotlin
GGUFNativeLib.nativeSetNormOffsets(layer: Int, offsets: FloatArray)
GGUFNativeLib.nativeResetNormOffsets()
```

### Advanced: Hypernetwork LoRA (P4)

```kotlin
GGUFNativeLib.nativeInitHypernetwork(rank: Int)  // 4 recommended
GGUFNativeLib.nativeSetHypernetworkLora(layer: Int, matA: FloatArray, matB: FloatArray)
GGUFNativeLib.nativeInitHypernetworkFromDirections()  // Auto from control vectors
```

### Advanced: Sparse Masks (P5)

```kotlin
GGUFNativeLib.nativeInitSparseMasks(sparsity: Float, seed: Int)  // 0.1 = disable 10% neurons
GGUFNativeLib.nativeSetSparseMask(layer: Int, mask: FloatArray)
GGUFNativeLib.nativeUpdateSparseMasks()
```

### Advanced: KAN-lite Activation Overlay (P6)

```kotlin
GGUFNativeLib.nativeInitKan(alpha: Float)  // 0.01-0.1
GGUFNativeLib.nativeSetKanLayerCoefficients(layer: Int, coeffs: FloatArray)
```

### Advanced: Forward Learning — SPSA (P7)

```kotlin
GGUFNativeLib.nativeForwardLearnStep(
    tokens: IntArray,
    learningRate: Float,  // 0.001-0.01
    noiseScale: Float     // 0.01-0.1
): Float  // Returns loss improvement (positive = better)
```

---

## Native Build

Uses CMake with runtime CPU variant selection:

```
GGML_CPU_ALL_VARIANTS=ON   → 7 .so variants (armv8.0, dotprod, fp16, i8mm, sve, sme, sme2)
GGML_BACKEND_DL=ON         → Dynamic backend loading at runtime
GGML_LLAMAFILE=OFF         → Not supported on Android
```

Optimization flags: `-O3 -ffast-math -fno-finite-math-only -ffp-contract=fast`

---

## Device Tier Defaults

```kotlin
GGUFNativeLib.LowEndDefaults    // 2 threads, 1024 ctx, Q4_0 cache
GGUFNativeLib.MidRangeDefaults  // 4 threads, 2048 ctx, Q8_0 cache
GGUFNativeLib.HighEndDefaults   // 6 threads, 4096 ctx, FP16 cache
```
