# Backend Plugin API Guide

## What is this?

`backend_plugin_api` defines the contract that every AI backend must implement to work with the plugin system. If you're building a backend (text gen, image gen, TTS, or anything else), this module is your only dependency.

**You implement the interfaces. The plugin manager handles everything else** — loading, lifecycle, memory management, conflict resolution, hardware detection.

## Architecture

```
Your Backend (e.g., ai_gguf)
  │
  ├── depends on: backend_plugin_api (interfaces only)
  ├── implements: BackendPlugin + capability interfaces
  └── ships as: .zip (classes.dex + .so + manifest.json)

backend_manager (you don't touch this)
  │
  ├── reads manifest.json
  ├── loads your .dex + .so via DexClassLoader
  ├── instantiates your entry class via reflection
  └── manages lifecycle, memory, conflicts
```

**Key rule:** Your backend depends ONLY on `backend_plugin_api`. Never on `backend_manager`.

## Quick Start: Implementing a Backend

### Step 1: Add the dependency

```kotlin
// your_backend/build.gradle.kts
dependencies {
    implementation(project(":backend_plugin_api"))
}
```

### Step 2: Implement BackendPlugin + capability interfaces

Pick which capabilities your backend supports:

| Capability | Interface to implement | What it does |
|---|---|---|
| `TEXT_GEN` | `TextGenBackend` | Streaming text generation |
| `IMAGE_GEN` | `ImageGenBackend` | Image generation (txt2img, img2img) |
| `TTS` | `TTSBackend` | Text-to-speech synthesis |
| `EMBEDDING` | `EmbeddingBackend` | Text embedding vectors |
| `TOOL_CALLING` | `ToolCallingBackend` | Grammar-constrained JSON tool calls |
| `VISION` | `VisionBackend` | Multimodal text+image input |

Example for a text generation backend:

```kotlin
class GGUFBackendPlugin : BackendPlugin, TextGenBackend, EmbeddingBackend, ToolCallingBackend {

    override val id = "ai_gguf"
    override val name = "GGUF Text Generation"
    override val capabilities = setOf(
        Capability.TEXT_GEN,
        Capability.EMBEDDING,
        Capability.TOOL_CALLING
    )

    private val _state = MutableStateFlow(BackendState.UNLOADED)
    override val state: StateFlow<BackendState> = _state.asStateFlow()

    override fun estimateMemory(model: ModelMetadata): Long {
        // Return estimated RAM in bytes
        return model.fileSizeBytes + (model.extras["contextSize"]?.toLong()?.times(2048) ?: 0)
    }

    override suspend fun loadModel(model: ModelMetadata): Result<Unit> = runCatching {
        _state.value = BackendState.LOADING
        // Your native model loading code here
        // Use model.path for file path
        // Use model.extras for backend-specific params (context size, quant, etc.)
        _state.value = BackendState.READY
    }

    override suspend fun unloadModel() {
        // Free native resources
        _state.value = BackendState.UNLOADED
    }

    override suspend fun pause(): Result<Unit> = runCatching {
        // Stop generation but keep model in memory
        _state.value = BackendState.PAUSED
    }

    override suspend fun resume(): Result<Unit> = runCatching {
        _state.value = BackendState.READY
    }

    override fun canHandle(model: ModelMetadata): Boolean {
        return model.requiredBackend == id
    }

    override fun activeTask(): TaskType? {
        return if (_state.value == BackendState.RUNNING) TaskType.TEXT_GENERATION else null
    }

    override fun getInfo(): Map<String, String> = mapOf(
        "model" to "currently loaded model name",
        "quantization" to "Q8_0",
        "contextUsed" to "1024/2048"
    )

    override suspend fun release() {
        unloadModel()
    }

    // --- TextGenBackend ---

    override suspend fun generateStream(
        messagesJson: String,
        maxTokens: Int,
        callback: TextGenCallback
    ) {
        _state.value = BackendState.RUNNING
        try {
            // Your generation loop here
            // Call callback.onToken() for each token
            // Call callback.onToolCall() if tool call detected
            // Call callback.onDone() when finished
        } catch (e: Exception) {
            callback.onError(e.message ?: "Generation failed")
        } finally {
            _state.value = BackendState.READY
        }
    }

    override fun stopGeneration() {
        // Signal native code to stop
    }

    override fun setStopStrings(strings: Array<String>) {
        // Pass to native layer
    }

    override fun updateSamplerParams(paramsJson: String) {
        // Update temperature, top_p, etc. at runtime
    }

    // --- EmbeddingBackend ---

    override suspend fun encode(text: String, normalize: Boolean): Result<FloatArray> {
        // Return embedding vector
    }

    override suspend fun encodeBatch(texts: List<String>, normalize: Boolean): Result<List<FloatArray>> {
        // Return batch of embedding vectors
    }

    override fun embeddingDimension(): Int = 1024

    // --- ToolCallingBackend ---

    override fun enableToolCalling(toolsJson: String) { /* ... */ }
    override fun disableToolCalling() { /* ... */ }
    override fun isToolCallingSupported(): Boolean = true
    override fun setGrammarMode(mode: Int) { /* 0=STRICT, 1=LAZY */ }
}
```

### Step 3: Create manifest.json

```json
{
  "id": "ai_gguf",
  "name": "GGUF Text Generation",
  "version": "1.0.0",
  "apiVersion": 1,
  "capabilities": ["TEXT_GEN", "EMBEDDING", "TOOL_CALLING"],
  "entryClass": "com.mp.ai_gguf.plugin.GGUFBackendPlugin",
  "nativeLibs": ["libai_gguf.so"],
  "minSdk": 27,
  "abi": ["arm64-v8a"]
}
```

### Step 4: Package as a zip

```
ai_gguf-arm64-v1.zip/
├── manifest.json
├── classes.dex          ← compiled from your Kotlin code
└── lib/
    └── arm64-v8a/
        └── libai_gguf.so
```

## Interface Reference

### BackendPlugin (core/BackendPlugin.kt)

The main lifecycle interface. Every backend MUST implement this.

| Method | When it's called | What you do |
|---|---|---|
| `estimateMemory(model)` | Before loading, to check RAM | Return bytes needed. 0 if unknown. |
| `loadModel(model)` | User selected a model | Load native model. Update state LOADING→READY. |
| `unloadModel()` | Switching models or freeing memory | Free all native resources. State→UNLOADED. |
| `pause()` | Another backend needs to run | Stop generation, keep model in memory. State→PAUSED. |
| `resume()` | Paused backend gets to continue | Resume where you left off. State→READY. |
| `canHandle(model)` | Routing — which backend for this model? | Return true if you can load this model. |
| `activeTask()` | Conflict detection | Return current TaskType if running, null if idle. |
| `getInfo()` | UI display, debugging | Return key-value pairs about current state. |
| `release()` | App shutdown | Free everything. Called once. |

**State machine:**
```
UNLOADED ──loadModel()──→ LOADING ──success──→ READY
    ↑                                           │  ↑
    │                                    generate() │
    │                                           ↓  │
    │                                        RUNNING
    │                                           │
    │                                       pause()
    │                                           ↓
    │                                        PAUSED
    │                                           │
    └──────────unloadModel()/release()──────────┘

Any state ──error──→ ERROR ──unloadModel()──→ UNLOADED
```

### TextGenCallback (callback/TextGenCallback.kt)

You call these methods during generation:

| Method | When to call |
|---|---|
| `onToken(token)` | Each decoded token |
| `onToolCall(name, argsJson)` | Grammar detected a complete tool call |
| `onDone(metrics)` | Generation finished (include timing metrics if available) |
| `onError(message)` | Something went wrong |

### ConflictCallback (callback/ConflictCallback.kt)

You do NOT implement this. The app (ToolNeuron) implements it to show the user:
> "Text generation is running. Pause it to start image generation?"

The plugin manager calls it automatically when two backends conflict.

### ModelMetadata (model/ModelMetadata.kt)

Passed to your `loadModel()`. Maps to ToolNeuron's Room DB `Model` entity.

| Field | What it is |
|---|---|
| `id` | Unique model ID (SHA-256 checksum) |
| `name` | Display name |
| `path` | File path or content:// URI |
| `requiredBackend` | Which backend this needs ("ai_gguf", "ai_sd", etc.) |
| `capabilities` | What this specific model supports |
| `fileSizeBytes` | Model file size on disk |
| `extras` | Backend-specific params as key-value strings |

**Common extras by backend type:**

For GGUF:
```
"contextSize" → "2048"
"flashAttn" → "true"
"cacheTypeK" → "9"     (Q8_0)
"cacheTypeV" → "9"
"gpuLayers" → "0"
```

For SD:
```
"textEmbeddingSize" → "768"
"runOnCpu" → "true"
"isPony" → "false"
"width" → "512"
"height" → "512"
```

For TTS:
```
"useNNAPI" → "false"
"voice" → "F1"
"speed" → "1.05"
```

### HardwareObserver (core/HardwareObserver.kt)

Available through `PluginManager.hardware`. Use it to make smart decisions:

```kotlin
val hw = pluginManager.hardware
val info = hw.getHardwareInfo()

// Decide thread count
val threads = hw.getRecommendedThreadCount()

// Check before loading a big model
val available = hw.getAvailableRamBytes()
if (available < estimateMemory(model)) {
    // warn user or pick smaller quant
}

// Throttle on thermal
when (hw.getThermalState()) {
    ThermalState.SEVERE -> reduceThreads()
    ThermalState.CRITICAL -> pauseGeneration()
    else -> { /* normal operation */ }
}
```

### BackendManifest (model/BackendManifest.kt)

Parsed from your manifest.json. Fields:

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique backend identifier. Must match `BackendPlugin.id`. |
| `name` | yes | Human-readable name |
| `version` | yes | Semver string (e.g., "1.0.0") |
| `apiVersion` | yes | Must exactly match `PluginManager.API_VERSION` (currently 1) |
| `capabilities` | yes | Array of Capability enum names |
| `entryClass` | yes | Fully qualified class implementing BackendPlugin |
| `nativeLibs` | yes | List of .so filenames to load |
| `minSdk` | yes | Minimum Android SDK version |
| `abi` | yes | Supported ABIs (e.g., ["arm64-v8a"]) |

## How the Plugin Manager Uses Your Backend

1. **Discovery**: Scans `plugins/` directory, parses your `manifest.json`
2. **Validation**: Checks `apiVersion` (exact match), `minSdk`, ABI compatibility
3. **Loading**: `DexClassLoader` loads `classes.dex`, `System.load()` for each `.so`
4. **Instantiation**: Calls `Class.forName(entryClass).newInstance()` — **your class needs a no-arg constructor**
5. **Memory check**: Calls `estimateMemory()` before loading a model
6. **Conflict resolution**: If another backend is RUNNING, asks user via `ConflictCallback`
7. **Model load**: Calls `loadModel(metadata)` with the user's selected model
8. **Usage**: App casts to capability interface and calls methods
9. **Cleanup**: Calls `release()` on shutdown

## Important Rules

1. **No-arg constructor required** — your entry class must have `constructor()` with no params
2. **State transitions are your responsibility** — update `_state` at each lifecycle point
3. **Thread safety** — `generateStream` runs on a coroutine, but native callbacks may come from native threads. Ensure your callback dispatching is safe.
4. **Don't hold references to the manager** — your backend is self-contained
5. **API version must match exactly** — if the manager is API v1 and your manifest says v2, it won't load. Re-download the backend.
6. **estimateMemory should be fast** — it's called before loading, don't do I/O
7. **pause() should be quick** — just set a flag to stop generation, don't unload the model
