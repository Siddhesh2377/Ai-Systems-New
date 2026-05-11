# AiSystems — Model Catalog

Single source of truth for which models the SDKs support, where to download them, and how to test each surface on-device.

Device target: Snapdragon 7s Gen 3 / 7+ Gen 3 / 8 Gen 1+ (8 GB RAM class). Anything marked **Large** is tight on 8 GB; close other apps first.

---

## LLM / VLM (`:gguf_lib` — llama.cpp + mtmd CLIP)

All models below are auto-listed in the ToolNeuron-New repository picker after install (Settings → Repositories). Tap a repo → pick a quant → download.

| # | Family   | Repo                                            | Type    | Size (Q4_K_M) | Notes                                                |
|---|----------|-------------------------------------------------|---------|----------------|------------------------------------------------------|
| 1 | LFM      | `LiquidAI/LFM2.5-350M-GGUF`                     | Text    | ~210 MB        | Fastest text model in the catalog (38+ tk/s decode). |
| 2 | LFM      | `LiquidAI/LFM2-VL-450M-GGUF`                    | Vision  | ~260 MB + 90 MB mmproj | Smallest VLM. Use ImageQuality.MEDIUM by default.    |
| 3 | Qwen     | `Qwen/Qwen3-0.6B-GGUF`                          | Text    | ~400 MB        | Smallest Qwen3. Tool-calling tested with grammar engine. |
| 4 | Qwen     | `unsloth/Qwen3.5-0.8B-GGUF`                     | Text    | ~520 MB        | Newer Qwen 3.5 base, slightly stronger than 0.6B.    |
| 5 | Qwen     | `unsloth/Qwen3.5-4B-GGUF`                       | Text    | ~2.5 GB        | Best-quality Qwen that fits comfortably in 8 GB.     |
| 6 | Qwen-VL  | `Qwen/Qwen3-VL-2B-Instruct-GGUF`                | Vision  | ~1.5 GB + ~400 MB mmproj | M-RoPE; needs the matching `mmproj-*.gguf`. The reference VLM we validate against. |
| 7 | Mistral  | `bartowski/Mistral-7B-Instruct-v0.3-GGUF`       | Text    | ~4.4 GB        | **Large.** Pick Q3_K_S (~3.2 GB) on 8 GB devices.    |
| 8 | Gemma    | `unsloth/gemma-3-1b-it-GGUF`                    | Text    | ~600 MB        | LLM_ARCH_GEMMA3 — Gemma 4 not yet supported.         |
| 9 | SmolLM3  | `HuggingFaceTB/SmolLM3-3B-GGUF`                 | Text    | ~1.9 GB        | Best on-device tool-calling model (92.3% BFCL).      |
| 10| Phi      | `unsloth/Phi-3.5-mini-instruct-GGUF`            | Text    | ~2.3 GB        | Solid general-purpose 3.8B.                          |

Quant guide for mobile:
- **Q4_K_M** — default; best quality/size tradeoff.
- **Q3_K_S / Q3_K_M** — pick on 8 GB devices if Q4 OOMs.
- **Q8_0** — only for the 350M–1B sizes where it still fits.
- Avoid F16/BF16 on phone — too large.

### Currently **not** supported (will show "unsupported model architecture" error after the May 2026 port)
- Gemma 4 (`LLM_ARCH_GEMMA4`), Mistral Small 4 (`LLM_ARCH_MISTRAL4`), Hunyuan VL, DeepSeekOCR, MiniCPM-V 4.6, Phi-4 vision. See `memory/project_llamacpp_upstream_ports.md` for status.

---

## Speech-to-Text (`:ai_sherpa` — sherpa-onnx)

Sherpa supports Whisper, Paraformer, Transducer, NeMo CTC, TDNN. All models come from the sherpa-onnx release tarballs.

| Model                            | Lang    | Size   | HuggingFace tarball                                                                  |
|----------------------------------|---------|--------|--------------------------------------------------------------------------------------|
| Whisper tiny.en                  | EN only | ~75 MB | `csukuangfj/sherpa-onnx-whisper-tiny.en`                                              |
| Whisper base                     | 99 lang | ~140 MB| `csukuangfj/sherpa-onnx-whisper-base`                                                  |
| Whisper small.en                 | EN only | ~470 MB| `csukuangfj/sherpa-onnx-whisper-small.en`                                              |
| SenseVoice small (multilingual) | 50+ lang| ~230 MB| `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`                       |
| Paraformer (Chinese)            | ZH      | ~220 MB| `csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14`                                      |

Each tarball contains `encoder.onnx`, `decoder.onnx`, `tokens.txt`. Point `OfflineWhisperModelConfig` at the extracted folder. See `ai_sherpa/README.md` for the exact init code.

---

## Text-to-Speech (`:ai_sherpa` — VITS / Kokoro; `:ai_tts` — Supertonic)

| Engine    | Voice / Model                                | Lang | Size    | Source                                                              |
|-----------|----------------------------------------------|------|---------|---------------------------------------------------------------------|
| Kokoro    | Kokoro-82M (multivoice)                      | EN   | ~310 MB | `csukuangfj/kokoro-onnx`                                            |
| VITS      | Piper en_US-amy (and many other voices)      | EN   | ~75 MB  | `csukuangfj/sherpa-onnx-vits-piper-en_US-amy-low`                   |
| VITS      | Piper en_GB-northern_english_male            | EN   | ~63 MB  | `csukuangfj/sherpa-onnx-vits-piper-en_GB-northern_english_male-medium` |
| Supertonic| Bundled in `:ai_tts`                          | EN   | n/a     | Vendored — no download needed.                                      |

Piper has voices in 30+ languages — browse the `csukuangfj/sherpa-onnx-vits-piper-*` namespace on HF.

---

## Stable Diffusion / NPU (`:ai_sd` — QNN HTP + MNN fallback)

NPU bundles run on Hexagon HTP via QNN. QNN is **mandatory** on Snapdragon — MNN is a slow CPU fallback only.

| Variant suffix | Compatible SoCs                | RAM   | Notes |
|----------------|--------------------------------|-------|-------|
| `8gen1`        | SM8450 (Snapdragon 8 Gen 1)    | 12 GB | Original Hexagon V69. Only on 8 Gen 1. |
| `8gen2`        | SM8550 (Snapdragon 8 Gen 2)    | 12 GB | Hexagon V73. |
| `min`          | Gen 3+ (also works below 8gen1)| 8 GB  | Reduced-token / smaller-resolution. **Default for 7s Gen 3 / A56.** |

| Model bundle                           | Source                                   |
|----------------------------------------|------------------------------------------|
| `AbsoluteReality_qnn2.28_min.zip`      | `xororz/sd-qnn` (currently auto-installed) |
| `mistoonAnime_v30-8gen1.zip`           | `Mr-J-369/sd-qnn`                          |
| `realhotspice-qnn2.28-8gen1.zip`       | `Mr-J-369/sd-qnn`                          |

Native SoC detection: `SDNativeLib.nativeGetSocInfo()` returns SoC ID + HTP version. Use it to auto-pick the right variant. See `ai_sd/CLAUDE.md` for the full pipeline.

### Auxiliary image modules (also `:ai_sd`)
| Module          | Model              | Size   | Speed       |
|-----------------|--------------------|--------|-------------|
| Upscaler        | Real-ESRGAN x4plus | ~65 MB | 1-2 s       |
| Segmentation    | MobileSAM          | ~46 MB | 12 ms/query |
| Inpainting      | LaMa               | ~200 MB| 100-300 ms  |
| Depth           | MiDaS v2.1 small   | ~66 MB | 15-120 ms   |
| Style Transfer  | AdaIN VGG          | ~25 MB | 30-60 ms    |

Conversion source links live in `ai_sd/CLAUDE.md` under "Image Processing Modules".

---

## How to test on-device

Build the test app (ToolNeuron-New `:app`) and try each surface:

```sh
# Build + install
cd /home/home/AndroidStudioProjects/ToolNeuron-New
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Inside the app:
1. **Model Store** → tap a repo (e.g. Qwen3-VL 2B Instruct) → pick `Q4_K_M.gguf` and `mmproj-Q8_0.gguf` → Download.
2. **Home → Load Model** → select downloaded model.
3. **Text test**: send "what is 27 * 38?" — confirm decode rate matches expected (~24-38 tk/s on LFM 350M, ~12-18 on 3B-class, ~4-6 on 7B).
4. **VLM test**: attach an image — wait for the spinner → checkmark on the thumbnail (pre-warm done), then send "describe this image". Confirm no crash and reasonable output.
5. **Tool calling test** (SmolLM3 / Qwen3-0.6B): enable a tool in Settings → ask "what's the weather in Tokyo?" — should emit `<tool_call>{…}</tool_call>`.
6. **ASR test**: in voice input — speak — confirm transcribed text appears (Whisper tiny.en is fastest to validate).
7. **TTS test**: tap speaker icon on a response — Kokoro voice should play.
8. **SD test**: open the AiSystems `:app` module separately (this repo's `app/`) — run a 512x512 prompt on AbsoluteReality_qnn2.28_min.

### Expected sanity numbers (Snapdragon 7s Gen 3, PERFORMANCE thread mode)
| Workload                          | Expected     |
|-----------------------------------|--------------|
| LFM 2.5 350M Q4 decode            | 34-38 tk/s   |
| Qwen3 0.6B Q4 decode              | 22-28 tk/s   |
| Qwen3.5 4B Q4 decode              | 7-9 tk/s     |
| Mistral 7B Q3_K_S decode          | 3-5 tk/s     |
| Qwen3-VL 2B image-prefill (1024 tokens) | 4-6 s   |
| AbsoluteReality SD `_min` (4 steps, 512x512) | 8-10 s |

If you see anything > 2× worse than these, check `Settings → Performance → Thread Mode` (should be Balanced or Performance), and verify no other heavy app is running.

---

## When you actually pick a model

- **You only want fast replies, short context**: LFM 2.5 350M.
- **You want a vision model that fits anywhere**: LFM2-VL 450M.
- **You want a vision model that's actually smart**: Qwen3-VL 2B Instruct.
- **You want tool calling that works**: SmolLM3 3B or Qwen3 0.6B.
- **You want maximum quality and you have 8+ GB free RAM**: Qwen3.5 4B or Mistral 7B (Q3).
- **You want Google's vibes**: Gemma 3 1B.
- **You want Microsoft's vibes**: Phi 3.5 Mini.
