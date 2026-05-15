# ai_sherpa

Android library wrapping [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
for on-device, offline speech-to-text and text-to-speech.

- Package: `com.dark.ai_sherpa`
- Min SDK: 29
- ABIs: `arm64-v8a`, `armeabi-v7a`

## Scope

Only the offline (non-streaming) recognizer and offline TTS are exposed. Online
streaming ASR and Voice Activity Detection were removed in the Apr 2026 cleanup
because no consumer used them; reintroduce by reverting through git history if
needed.

| Surface              | Supported model families                                 |
| -------------------- | -------------------------------------------------------- |
| `OfflineRecognizer`  | Whisper, Paraformer, Transducer, NeMo CTC, TDNN          |
| `OfflineTts`         | VITS, Kokoro                                              |
| `SherpaLib`          | Process-wide native crash + last-error JSON              |

## Usage

### Offline ASR (Whisper)

```kotlin
val cfg = OfflineRecognizerConfig(
    modelConfig = OfflineModelConfig(
        whisper = OfflineWhisperModelConfig(
            encoder = "/path/to/encoder.onnx",
            decoder = "/path/to/decoder.onnx",
            language = "en",
            task = "transcribe",
        ),
        tokens = "/path/to/tokens.txt",
        // numThreads defaults to min(cpus, 4); override only if needed.
    ),
)

OfflineRecognizer.fromFile(cfg).use { recognizer ->
    recognizer.createStream().use { stream ->
        stream.acceptWaveform(sampleRate = 16_000, samples = floatArray)
        recognizer.decode(stream)
        val result = recognizer.getResult(stream)
        println(result.text)
    }
}
```

`createStream()` allocates a single-utterance decoder. Reuse the recognizer
across utterances; create a fresh stream per utterance.

### TTS (VITS)

```kotlin
val cfg = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
        vits = OfflineTtsVitsModelConfig(
            model  = "/path/to/model.onnx",
            tokens = "/path/to/tokens.txt",
        ),
    ),
)

OfflineTts.fromFile(cfg).use { tts ->
    val audio = tts.generate(text = "Hello world", sid = 0, speed = 1.0f)
    // audio.samples: FloatArray, audio.sampleRate: Int (model-defined)
}
```

### Crash + error tracker

Diagnostics, error capture, and signal-handler crash files are owned by the
`:tn_security` module. The host app installs handlers + the crash-file pattern
once via `com.dark.tn_security.TnSecurity`; every log line and every error
emitted by this SDK (and by the upstream sherpa-onnx library — its
`SHERPA_ONNX_LOGE` macro is rerouted in the fork) is delivered to that same
sink with the `TN_MODULE_AI_SHERPA` / `TN_MODULE_SHERPA_ONNX` tag set.

No per-SDK error API is exposed any more; consumers read errors out of the
unified `TnSecurity` event stream.

## Threading and lifecycle

- `fromFile`, `decode`, `getResult`, `generate` are blocking. Call them from
  a background dispatcher (`Dispatchers.IO`).
- All native handles are `AutoCloseable`. Forgetting to close leaks the ONNX
  session, which can be 50–500 MB.
- After `close()`, calling any method throws — this is deliberate, to surface
  use-after-close bugs early.

## Memory notes

- `numThreads` defaults to `min(availableProcessors, 4)` and is clamped to
  `>= 1` both in Kotlin and again in C++ defensively.
- Audio buffers passed to `acceptWaveform` are read via
  `GetPrimitiveArrayCritical` (zero-copy when the JVM allows). The buffer is
  not retained — reuse or free it immediately after the call.
- Generated TTS audio is copied into a fresh Java `FloatArray`; the native
  buffer is destroyed before the call returns.

## Build prerequisites

- Android NDK 27.3.13750724
- CMake 3.22.1
- Pre-built ONNX Runtime Android at `/home/home/dev/include/ort-android-1.24.3/`
- sherpa-onnx checkout at `/home/home/dev/include/sherpa-onnx`

These paths are hard-coded in `src/main/cpp/CMakeLists.txt`. Adjust there if
your environment differs.

## Layout

```
src/main/
├── cpp/
│   ├── CMakeLists.txt        sherpa-onnx subbuild + JNI shared lib
│   ├── jni_cache.{h,cpp}     JNI_OnLoad: cached jclass / jmethodID refs
│   ├── jni_common.h          field-getter helpers, CHECK_PTR macro
│   ├── offline_recognizer.cpp
│   └── offline_tts.cpp
└── java/com/dark/ai_sherpa/
    ├── SherpaLib.kt              library loader
    ├── OfflineRecognizer.kt      recognizer + stream
    ├── OfflineRecognizerConfig.kt
    ├── OfflineTts.kt
    ├── OfflineTtsConfig.kt
    └── Models.kt                 result data classes
```
