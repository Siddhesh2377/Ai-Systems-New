# Multi-Backend Plugin System Design

**Date**: 2026-02-28
**Modules**: `backend_plugin_api`, `backend_manager`

## Architecture

```
backend_plugin_api  (pure Kotlin interfaces — ZERO implementation)
    ↑ implements                    ↑ depends on
    |                               |
ai_gguf adapter                  backend_manager
ai_sd adapter                    (registry, lifecycle, C++ hardware observer)
ai_tts adapter                      ↑ depends on
                                    |
                                ToolNeuron (wires everything + file_manager + local-backend-db)
```

## Decisions

- Plugin orchestration lives in ToolNeuron, SDKs stay independent
- Open plugin system — new engine types addable without changing core
- Targets all device tiers (4GB–12GB+)
- User-prompted conflict resolution ("a task is running, pause it?")
- Single service process
- Model metadata-based routing
- Backends are downloadable zips (.so + DEX + manifest.json) hosted on HuggingFace
- Single HF repo with folder structure + index.json for discovery
- Exact API version match required
- file_manager module in ToolNeuron handles all downloads
- local-backend-db (Room): download manager writes, backend manager reads

## backend_plugin_api — Pure Interfaces

### Enums
- `Capability`: TEXT_GEN, IMAGE_GEN, TTS, EMBEDDING, TOOL_CALLING, VISION, ASR
- `BackendState`: UNLOADED, LOADING, READY, RUNNING, PAUSED, ERROR
- `TaskType`: TEXT_GENERATION, IMAGE_GENERATION, TTS_SYNTHESIS, EMBEDDING_COMPUTE, TOOL_EXECUTION

### Core Interface: BackendPlugin
- id, name, capabilities, state (StateFlow)
- estimateMemory(model): Long
- loadModel(model): Result<Unit>
- unloadModel()
- pause() / resume()
- canHandle(model): Boolean
- activeTask(): TaskType?
- getInfo(): Map<String, String>
- release()

### Capability Interfaces
- TextGenBackend: generateStream(), stopGeneration()
- ImageGenBackend: generate(), stopGeneration()
- TTSBackend: synthesize(), stopSynthesis()
- EmbeddingBackend: encode(), encodeBatch()

### Callbacks
- TextGenCallback: onToken, onToolCall, onDone, onError
- ImageGenCallback: onProgress, onImageReady, onError
- TTSCallback: onAudioChunk, onComplete, onError

## backend_manager — Implementation

### Kotlin
- PluginManager: registry, lifecycle, conflict resolution
- PluginLoader: DexClassLoader + System.load() for .so
- PluginRegistry: reads local-backend-db, tracks loaded plugins
- ConflictResolver: checks active tasks, asks user via callback

### C++ (JNI)
- HardwareObserver: RAM available, thermal state, CPU topology
- MemoryTracker: per-backend memory accounting

## HuggingFace Layout
```
HF Repo/
├── index.json
├── ai_gguf/v1/ai_gguf-arm64-v1.zip
├── ai_sd/v1/ai_sd-arm64-v1.zip
└── ai_tts/v1/ai_tts-arm64-v1.zip
```

## Plugin Zip Contents
```
ai_gguf-arm64-v1.zip/
├── manifest.json
├── classes.dex
└── lib/arm64-v8a/libai_gguf.so
```

## manifest.json
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

## local-backend-db Table
```
installed_backends
  - id: String (PK)
  - name: String
  - version: String
  - apiVersion: Int
  - installPath: String
  - capabilities: String (JSON array)
  - entryClass: String
  - nativeLibs: String (JSON array)
  - installedAt: Long
  - status: String (INSTALLED, UPDATING, CORRUPT)
```
