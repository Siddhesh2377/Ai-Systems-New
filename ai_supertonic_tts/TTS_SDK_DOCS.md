# Supertonic TTS Android SDK — `ai_supertonic_tts`

## Overview

On-device text-to-speech SDK for Android using **Supertonic v2** (66M parameters, ONNX Runtime). Produces 44.1 kHz mono audio at up to 167x faster than real-time on modern hardware. Supports 5 languages, 10 voices, and runs entirely on CPU (with optional NNAPI GPU/NPU acceleration).

**Module**: `com.mp.ai_supertonic_tts`
**Main class**: `SupertonicTTS`
**License**: Code — MIT, Model weights — OpenRAIL-M

---

## Model Download

### Supertonic v2 (recommended — multilingual)

**HuggingFace repo**: https://huggingface.co/Supertone/supertonic-2

Clone with Git LFS:
```bash
git lfs install
git clone https://huggingface.co/Supertone/supertonic-2
```

Or download individual files:

#### ONNX Models (~263 MB total)
| File | Size | URL |
|------|------|-----|
| `onnx/duration_predictor.onnx` | 1.5 MB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/duration_predictor.onnx |
| `onnx/text_encoder.onnx` | 27.4 MB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/text_encoder.onnx |
| `onnx/vector_estimator.onnx` | 132 MB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vector_estimator.onnx |
| `onnx/vocoder.onnx` | 101 MB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vocoder.onnx |

#### Config Files
| File | Size | URL |
|------|------|-----|
| `onnx/tts.json` | 8.7 KB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/tts.json |
| `onnx/unicode_indexer.json` | 262 KB | https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/unicode_indexer.json |

#### Voice Styles (~420 KB each)
| File | URL |
|------|-----|
| `voice_styles/F1.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F1.json |
| `voice_styles/F2.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F2.json |
| `voice_styles/F3.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F3.json |
| `voice_styles/F4.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F4.json |
| `voice_styles/F5.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F5.json |
| `voice_styles/M1.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M1.json |
| `voice_styles/M2.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M2.json |
| `voice_styles/M3.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M3.json |
| `voice_styles/M4.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M4.json |
| `voice_styles/M5.json` | https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M5.json |

### Supertonic v1 (legacy — English only)

**HuggingFace repo**: https://huggingface.co/Supertone/supertonic

Same file structure but outputs 24,000 Hz audio and only supports English.

### Required Directory Structure on Device

```
/path/to/supertonic-2/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── tts.json
│   └── unicode_indexer.json
└── voice_styles/
    ├── F1.json
    ├── F2.json
    ├── F3.json
    ├── F4.json
    ├── F5.json
    ├── M1.json
    ├── M2.json
    ├── M3.json
    ├── M4.json
    └── M5.json
```

---

## SDK API Reference

### SupertonicTTS — Main Entry Point

```kotlin
import com.mp.ai_supertonic_tts.SupertonicTTS
import com.mp.ai_supertonic_tts.models.TTSConfig
import com.mp.ai_supertonic_tts.models.Language
import com.mp.ai_supertonic_tts.models.AudioFormat

// Create instance (context needed for URI operations)
val tts = SupertonicTTS(context)
```

#### Model Loading

```kotlin
// Load models from directory
val success = tts.loadModel("/sdcard/supertonic-2")

// Load with NNAPI GPU/NPU acceleration
val success = tts.loadModel("/sdcard/supertonic-2", useNNAPI = true)

// Check if loaded
tts.isModelLoaded()  // Boolean

// Get available voices
tts.getAvailableVoices()  // ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]

// Release resources
tts.release()
```

#### Synthesis (returns audio data)

```kotlin
// Basic synthesis (suspend function — call from coroutine)
val result = tts.synthesize("Hello world")

// With config
val result = tts.synthesize("Hello world", TTSConfig(
    speed = 1.0f,           // 1.0 = normal, >1.0 = faster
    steps = 5,              // 2 = fast, 5 = high quality
    language = Language.EN,  // EN, KO, ES, PT, FR
    voice = "F2"            // F1-F5, M1-M5
))

// With progress callback
val result = tts.synthesize("Hello world", TTSConfig(), object : TTSCallback {
    override fun onSynthesisStart(textLength: Int, chunkCount: Int) {
        Log.d("TTS", "Starting: $textLength chars, $chunkCount chunks")
    }
    override fun onChunkProgress(chunkIndex: Int, totalChunks: Int) {
        Log.d("TTS", "Progress: $chunkIndex/$totalChunks")
    }
    override fun onAudioReady(result: SynthesisResult) {
        Log.d("TTS", "Done: ${result.durationMs}ms audio, RTF=${result.realtimeFactor}")
    }
    override fun onError(error: String) {
        Log.e("TTS", "Error: $error")
    }
})
```

#### Playback (synthesize + play via AudioTrack)

```kotlin
// Speak (synthesize + play)
tts.speak("Hello world")

// Speak with config
tts.speak("Bonjour le monde", TTSConfig(
    voice = "M1",
    language = Language.FR,
    speed = 0.9f
))

// Playback controls
tts.stopPlayback()
tts.pausePlayback()
tts.resumePlayback()
tts.isPlaying()         // Boolean
tts.setVolume(0.8f)     // 0.0 to 1.0
```

#### Read from Any Source

```kotlin
// From file path
tts.speakFromFile("/sdcard/script.txt")

// From content URI (e.g. SAF picker)
tts.speakFromUri(uri)

// From InputStream
tts.speakFromStream(inputStream)

// Synthesize without playback
val result = tts.synthesizeFromFile("/sdcard/script.txt")
val result = tts.synthesizeFromUri(uri)
```

#### Save Audio

```kotlin
val result = tts.synthesize("Hello world")

// Save as 16-bit PCM WAV (standard, smallest)
tts.saveAudio(result, "/sdcard/output.wav", AudioFormat.WAV_16)

// Save as 32-bit float WAV (highest quality)
tts.saveAudio(result, "/sdcard/output.wav", AudioFormat.WAV_32F)

// Save as raw 16-bit PCM (no header)
tts.saveAudio(result, "/sdcard/output.pcm", AudioFormat.PCM_16)

// Save to content URI
tts.saveAudio(result, uri, AudioFormat.WAV_16)

// Get byte array (for custom handling)
val bytes = tts.toByteArray(result, AudioFormat.WAV_16)
```

### SynthesisResult

```kotlin
data class SynthesisResult(
    val audioData: FloatArray,    // Raw float32 samples [-1.0, 1.0]
    val sampleRate: Int,          // 44100 (v2) or 24000 (v1)
    val channels: Int,            // 1 (mono)
    val durationMs: Long,         // Audio duration in ms
    val synthesisTimeMs: Long     // Synthesis wall-clock time in ms
) {
    val realtimeFactor: Float     // < 1.0 = faster than real-time
    val sampleCount: Int          // Total number of audio samples
}
```

### TTSConfig

```kotlin
data class TTSConfig(
    val speed: Float = 1.05f,              // Speech speed (1.0 = normal)
    val steps: Int = 2,                    // Denoising steps (2=fast, 5=quality)
    val language: Language = Language.EN,   // Target language
    val voice: String = "F1",              // Voice style name
    val useNNAPI: Boolean = false,         // GPU/NPU acceleration
    val chunkingEnabled: Boolean = true,   // Auto-split long text
    val chunkSilenceMs: Int = 300          // Silence between chunks (ms)
)
```

### Language

```kotlin
enum class Language(val tag: String) {
    EN("en"),  // English
    KO("ko"),  // Korean
    ES("es"),  // Spanish
    PT("pt"),  // Portuguese
    FR("fr")   // French
}
```

### AudioFormat

```kotlin
enum class AudioFormat {
    WAV_16,     // 16-bit PCM WAV (standard)
    WAV_32F,    // 32-bit float WAV (highest quality)
    PCM_16,     // Raw 16-bit PCM (no header)
    PCM_32F,    // Raw 32-bit float PCM (no header)
    RAW_FLOAT   // Raw float array bytes (native byte order)
}
```

---

## Architecture

### Inference Pipeline

```
Text input
  → TextProcessor (Unicode NFKD normalize, emoji removal, language tags, tokenize via unicode_indexer.json)
  → TextChunker (split at sentence boundaries if >300 chars)
  → For each chunk:
      → Duration Predictor ONNX (text_ids + style_dp + text_mask → duration_seconds)
      → Text Encoder ONNX (text_ids + style_ttl + text_mask → text_embeddings)
      → Gaussian Noise Init (Box-Muller → noisy_latent [1, 144, L])
      → Vector Estimator ONNX × N steps (Euler flow-matching denoising)
      → Vocoder ONNX (clean_latent → float32 audio)
  → Concatenate chunks with silence gaps
  → Clip to [-1, 1] via C++ JNI
  → AudioPlayer (AudioTrack) or AudioSaver (WAV/PCM file)
```

### Module Structure

```
ai_supertonic_tts/
├── build.gradle.kts                          # Kotlin, ONNX Runtime, coroutines, NDK
├── src/main/
│   ├── cpp/
│   │   ├── CMakeLists.txt                    # C++17, 16KB page alignment
│   │   └── src/
│   │       ├── supertonic_jni.cpp            # JNI bridge (4 native functions)
│   │       ├── audio/
│   │       │   ├── wav_encoder.h             # WAV/PCM encoding API
│   │       │   └── wav_encoder.cpp           # RIFF/WAVE encoding, float→int16
│   │       └── utils/
│   │           └── logger.h                  # Android logcat macros
│   └── java/com/mp/ai_supertonic_tts/
│       ├── SupertonicTTS.kt                  # Main SDK facade (speak, synthesize, save)
│       ├── SupertonicNativeLib.kt            # JNI declarations
│       ├── engine/
│       │   ├── TTSEngine.kt                 # ONNX 4-model inference pipeline
│       │   ├── TextProcessor.kt             # Unicode normalization + tokenization
│       │   └── TextChunker.kt               # Sentence-boundary text splitting
│       ├── models/
│       │   ├── TTSConfig.kt                 # Synthesis configuration
│       │   ├── VoiceStyle.kt                # Voice embedding loader
│       │   ├── SynthesisResult.kt           # Audio result container
│       │   └── AudioFormat.kt               # Output format enum
│       ├── audio/
│       │   ├── AudioPlayer.kt               # AudioTrack playback
│       │   └── AudioSaver.kt                # File/URI saving
│       └── callback/
│           └── TTSCallback.kt               # Progress callbacks
```

### Dependencies

```toml
# gradle/libs.versions.toml
onnxruntime-android = "1.21.0"
kotlinx-coroutines = "1.10.1"

# build.gradle.kts
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.21.0")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1")
```

### ONNX Model Details (v2)

| Model | File Size | Input(s) | Output |
|-------|-----------|----------|--------|
| Duration Predictor | 1.5 MB | `text_ids` [1,S] int64, `style_dp` [1,8,64] f32, `text_mask` [1,1,S] f32 | `duration` [1] f32 (seconds) |
| Text Encoder | 27.4 MB | `text_ids` [1,S] int64, `style_ttl` [1,10,256] f32, `text_mask` [1,1,S] f32 | `text_emb` [1,S,256] f32 |
| Vector Estimator | 132 MB | `noisy_latent` [1,144,L] f32, `text_emb` [1,S,256] f32, `style_ttl` [1,10,256] f32, `text_mask` [1,1,S] f32, `latent_mask` [1,1,L] f32, `current_step` [1] f32, `total_step` [1] f32 | `xt` [1,144,L] f32 |
| Vocoder | 101 MB | `latent` [1,144,L] f32 | `wav` [1,samples] f32 |

Where: S = text sequence length, L = ceil(duration * 44100 / (512 * 6))

### Model Config (tts.json key values)

```
sample_rate:           44100
base_chunk_size:       512
chunk_compress_factor: 6
latent_dim:            24
latent_dim_total:      144 (= 24 * 6)
```

### Voice Style JSON Structure

Each voice file (e.g. `F1.json`) contains:

```json
{
  "style_ttl": {
    "data": [[[...256 floats...], ...10 rows...]],
    "dims": [1, 10, 256],
    "type": "float64"
  },
  "style_dp": {
    "data": [[[...64 floats...], ...8 rows...]],
    "dims": [1, 8, 64],
    "type": "float64"
  }
}
```

- `style_ttl`: Speaker embedding for Text-to-Latent module — shape [1, 10, 256]
- `style_dp`: Speaker embedding for Duration Predictor — shape [1, 8, 64]

---

## Integration Example (Full Android Activity)

```kotlin
class TTSActivity : AppCompatActivity() {
    private lateinit var tts: SupertonicTTS

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        tts = SupertonicTTS(this)

        // Load model (assumes files are already on device)
        val modelDir = "${filesDir.absolutePath}/supertonic-2"
        if (!tts.loadModel(modelDir)) {
            Log.e("TTS", "Failed to load: ${tts.lastError}")
            return
        }

        Log.d("TTS", "Voices: ${tts.getAvailableVoices()}")

        // Synthesize in coroutine
        lifecycleScope.launch {
            // Quick synthesis (2 steps)
            tts.speak("Hello, this is a test of Supertonic TTS.")

            // High quality synthesis (5 steps)
            val result = tts.synthesize(
                "The quick brown fox jumps over the lazy dog.",
                TTSConfig(steps = 5, voice = "M2", speed = 1.0f)
            )
            Log.d("TTS", "Result: $result")

            // Save to file
            tts.saveAudio(result, "${filesDir}/output.wav", AudioFormat.WAV_16)

            // French
            tts.speak("Bonjour le monde", TTSConfig(
                language = Language.FR, voice = "F3"
            ))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.release()
    }
}
```

---

## Performance Notes

- **Steps**: 2 steps is ~2.5x faster than 5 steps, with slightly lower quality. Use 2 for interactive, 5 for saved audio.
- **Speed**: Default 1.05 is Supertonic's recommended speed. 1.0 sounds slightly slower but more natural.
- **Chunking**: Text >300 chars is auto-chunked at sentence boundaries. Korean uses 120 char threshold.
- **Memory**: Models use ~300 MB RAM total when loaded. ONNX Runtime manages its own memory pool.
- **NNAPI**: Depends on device SoC. May not improve performance on all devices. Falls back to CPU if unavailable.
- **Audio output**: 44,100 Hz mono (v2). Float32 internally, converted to int16 only when saving WAV_16 or PCM_16.

## Supported Languages

| Language | Tag | Voice Styles |
|----------|-----|-------------|
| English | `en` | F1-F5, M1-M5 |
| Korean | `ko` | F1-F5, M1-M5 |
| Spanish | `es` | F1-F5, M1-M5 |
| Portuguese | `pt` | F1-F5, M1-M5 |
| French | `fr` | F1-F5, M1-M5 |

## Reference Links

- **GitHub**: https://github.com/supertone-inc/supertonic
- **HuggingFace v2**: https://huggingface.co/Supertone/supertonic-2
- **HuggingFace v1**: https://huggingface.co/Supertone/supertonic
- **Paper**: https://arxiv.org/abs/2503.23108
- **ONNX Runtime Android**: https://onnxruntime.ai/docs/get-started/with-android.html
