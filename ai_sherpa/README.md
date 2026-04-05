# ai_sherpa

Android JNI module wrapping [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for on-device
speech recognition (STT), text-to-speech (TTS), and voice activity detection (VAD).

Package: `com.dark.ai_sherpa`  
Min SDK: 29  
ABI: `arm64-v8a`, `armeabi-v7a`

## Features

- **Online ASR** — streaming speech recognition (Transducer, Paraformer, Zipformer2 CTC, NeMo CTC)
- **Offline ASR** — non-streaming recognition (Transducer, Paraformer, Whisper, SenseVoice, Moonshine, and more)
- **TTS** — offline text-to-speech (VITS, Kokoro)
- **VAD** — voice activity detection (Silero VAD, Ten VAD)
- **Wave I/O** — read/write `.wav` files

## Usage

### Streaming ASR

```kotlin
val config = OnlineRecognizerConfig(
    modelConfig = OnlineModelConfig(
        transducer = OnlineTransducerModelConfig(
            encoder = "/path/to/encoder.onnx",
            decoder = "/path/to/decoder.onnx",
            joiner  = "/path/to/joiner.onnx"
        ),
        tokens = "/path/to/tokens.txt",
        provider = "cpu",
        numThreads = 2
    ),
    decodingMethod = "greedy_search",
    enableEndpoint = true
)

OnlineRecognizer.fromFile(config).use { recognizer ->
    recognizer.createStream().use { stream ->
        stream.acceptWaveform(sampleRate = 16000, samples = floatArray)
        stream.inputFinished()
        while (recognizer.isReady(stream)) {
            recognizer.decode(stream)
        }
        val result = recognizer.getResult(stream)
        println(result.text)
    }
}
```

### Offline ASR (Whisper)

```kotlin
val config = OfflineRecognizerConfig(
    modelConfig = OfflineModelConfig(
        whisper = OfflineWhisperModelConfig(
            encoder = "/path/to/encoder.onnx",
            decoder = "/path/to/decoder.onnx",
            language = "en",
            task = "transcribe"
        ),
        tokens = "/path/to/tokens.txt",
        provider = "cpu",
        numThreads = 2
    )
)

OfflineRecognizer.fromFile(config).use { recognizer ->
    recognizer.createStream().use { stream ->
        stream.acceptWaveform(sampleRate = 16000, samples = floatArray)
        recognizer.decode(stream)
        val result = recognizer.getResult(stream)
        println(result.text)
    }
}
```

### TTS

```kotlin
val config = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
        vits = OfflineTtsVitsModelConfig(
            model  = "/path/to/model.onnx",
            tokens = "/path/to/tokens.txt"
        ),
        numThreads = 2,
        provider = "cpu"
    )
)

OfflineTts.fromFile(config).use { tts ->
    val audio = tts.generate(text = "Hello world", sid = 0, speed = 1.0f)
    // audio.samples: FloatArray, audio.sampleRate: Int
}
```

### VAD

```kotlin
val config = VadModelConfig(
    sileroVadModelConfig = SileroVadModelConfig(
        model = "/path/to/silero_vad.onnx",
        threshold = 0.5f,
        minSilenceDuration = 0.5f,
        minSpeechDuration = 0.25f,
        windowSize = 512
    ),
    sampleRate = 16000,
    numThreads = 1,
    provider = "cpu"
)

VoiceActivityDetector.fromFile(config, bufferSizeInSeconds = 30).use { vad ->
    vad.acceptWaveform(samples = floatArray)
    while (!vad.isEmpty()) {
        val segment = vad.front()
        vad.pop()
        // segment.samples: FloatArray, segment.start: Int
    }
}
```

### Wave I/O

```kotlin
val wave = WaveReader.read("/path/to/file.wav")
// wave.samples: FloatArray, wave.sampleRate: Int

WaveWriter.write("/output/path.wav", samples = floatArray, sampleRate = 16000)
```

## Architecture

- **JNI layer** (`src/main/cpp/`) — C++ bridging sherpa-onnx C API to Java
  - `jni_cache.cpp` — `JNI_OnLoad` caches all class refs and method IDs
  - `jni_common.h` — shared helpers (field getters, CHECK_PTR macro)
  - `online_recognizer.cpp`, `offline_recognizer.cpp`, `offline_tts.cpp`, `vad.cpp`, `wave_io.cpp`
- **Kotlin layer** (`src/main/java/com/dark/ai_sherpa/`) — idiomatic `AutoCloseable` wrappers

## Build requirements

- Android NDK 28+
- CMake 3.22.1
- ONNX Runtime Android prebuilt at `/home/home/dev/include/ort-android-1.24.3/`
- sherpa-onnx fork at `/home/home/dev/include/sherpa-onnx`

## Dependencies

- `sherpa-onnx-c-api`, `sherpa-onnx-core` (static, built from source fork)
- `libonnxruntime.so` (prebuilt per-ABI)
