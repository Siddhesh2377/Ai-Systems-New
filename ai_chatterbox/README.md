# ai_chatterbox — Emotional Text-to-Speech SDK

On-device emotional TTS using [Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT). Voice cloning + emotion control, runs entirely on-device via ONNX Runtime.

## Quick Start

```kotlin
val manager = ChatterboxManager.getInstance()

// Load model
manager.loadModel(ChatterboxConfig(
    modelDir = "/path/to/chatterbox_turbo/onnx",
    tokenizerPath = "/path/to/tokenizer.json",
    voicePresetDir = "/path/to/voice_preset",
    variant = ChatterboxVariant.TURBO
))

// Synthesize
val pcm = manager.synthesize("Hello, this is Chatterbox TTS on Android!")
// pcm is ShortArray of 24kHz mono PCM

// Observe state
manager.state.collect { state ->
    when (state) {
        is ChatterboxState.Generating -> println("Tokens: ${state.tokensGenerated}")
        is ChatterboxState.Complete -> playAudio(state.pcmData, state.sampleRate)
        is ChatterboxState.Error -> println("Error: ${state.message}")
        else -> {}
    }
}

// Cleanup
manager.release()
```

## Model Variants

| Variant | Params | Layers | Emotion | Size (q4f16) |
|---------|--------|--------|---------|--------------|
| **Turbo** | 350M (GPT-2 Medium) | 24 | No | ~558 MB |
| **Original** | 500M (Llama) | 30 | Yes | ~660 MB |

### Download Turbo (recommended for speed)

```bash
pip install huggingface-hub
huggingface-cli download ResembleAI/chatterbox-turbo-ONNX \
  --include "onnx/*q4f16*" "tokenizer.json" \
  --local-dir chatterbox_turbo
adb push chatterbox_turbo/ /sdcard/chatterbox/models/turbo/
```

### Download Original (for emotion control)

```bash
huggingface-cli download onnx-community/chatterbox-ONNX \
  --include "onnx/*q4*" "tokenizer.json" \
  --local-dir chatterbox_original
adb push chatterbox_original/ /sdcard/chatterbox/models/original/
```

## Voice Presets

A voice preset is 4 binary files extracted from a reference audio clip:
- `cond_emb.bin` — conditioning embedding (float32)
- `prompt_token.bin` — prompt speech tokens (int64)
- `speaker_embeddings.bin` — 192-dim speaker embedding (float32)
- `speaker_features.bin` — 500x80 speaker features (float32)

Generate with the Python Chatterbox library:
```python
from chatterbox.tts import ChatterboxTTS
import numpy as np

model = ChatterboxTTS.from_pretrained("cuda")
cond = model.cond_encoder.encode_reference("reference.wav")
# Save the 4 .bin files...
```

Or use [DDATT's extract_embeddings.py](https://github.com/DDATT/Chatterbox-turbo-cpp).

## API Reference

### ChatterboxConfig

```kotlin
data class ChatterboxConfig(
    val modelDir: String,               // Path to ONNX model directory
    val tokenizerPath: String,           // Path to tokenizer.json
    val voicePresetDir: String? = null,  // Path to voice preset .bin files
    val repetitionPenalty: Float = 1.2f, // Speech token repetition penalty
    val maxTokens: Int = 1024,           // Max speech tokens (~42s audio)
    val variant: ChatterboxVariant = ChatterboxVariant.TURBO,
    val exaggeration: Float = 1.0f       // Emotion strength (Original only)
)
```

### ChatterboxManager

| Method | Description |
|--------|-------------|
| `getInstance()` | Get singleton instance |
| `loadModel(config)` | Load models + voice (suspend) |
| `loadVoicePreset(dir)` | Swap voice preset |
| `synthesize(text)` | Text -> PCM audio (suspend) |
| `stop()` | Cancel generation (thread-safe) |
| `release()` | Free resources |
| `isReady()` | Models + voice loaded? |
| `state` | `StateFlow<ChatterboxState>` |

### ChatterboxState

| State | Description |
|-------|-------------|
| `Idle` | No model loaded |
| `Loading` | Loading models/voice |
| `Ready` | Ready to synthesize |
| `Generating(tokens)` | Speech token generation in progress |
| `Complete(pcm, rate)` | Audio ready (24kHz int16 PCM) |
| `Error(message)` | Something went wrong |

## Architecture

```
ChatterboxManager (Kotlin, StateFlow)
  └-> ChatterboxNativeLib (JNI externals)
       └-> chatterbox_jni.cpp (global state, mutex, callbacks)
            ├-> BPETokenizer (tokenizer.json -> token IDs)
            └-> ChatterboxEngine (ONNX Runtime)
                 ├-> embed_tokens.onnx (text -> embeddings)
                 ├-> language_model.onnx (AR generation + KV cache)
                 └-> conditional_decoder.onnx (tokens -> 24kHz PCM)
```

## Constants

| Constant | Value |
|----------|-------|
| Audio sample rate | 24000 Hz |
| Output format | int16 PCM mono |
| START_SPEECH_TOKEN | 6561 |
| STOP_SPEECH_TOKEN | 6562 |
| SILENCE_TOKEN | 4299 |
| EOS_TOKEN | 50256 (GPT-2) |
| Max generation | 1024 tokens |
| Repetition penalty | 1.2 |
| KV cache (Turbo) | 24 layers x 16 heads x 64 dim |
| KV cache (Original) | 30 layers x 16 heads x 64 dim |

## License

MIT — same as [Chatterbox](https://github.com/resemble-ai/chatterbox)
