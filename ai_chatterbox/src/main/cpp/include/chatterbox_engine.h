#pragma once

/**
 * ChatterboxEngine - ONNX Runtime inference engine for Chatterbox TTS.
 *
 * Ported from DDATT/Chatterbox-turbo-cpp with Android-specific adaptations:
 *   - No CUDA (CPU / NNAPI only)
 *   - Android logging (LOGI/LOGE/LOGD)
 *   - Graph optimization enabled (ORT_ENABLE_ALL)
 *   - Memory arena enabled (default)
 *   - std::atomic<bool> stop flag for cancellation
 *   - Deferred session construction via unique_ptr
 *   - I/O name arrays generated programmatically
 *
 * Pipeline stages:
 *   1. embed_tokens    - converts text token IDs to embeddings
 *   2. language_model  - autoregressive speech token generation with KV cache
 *   3. conditional_decoder - converts speech tokens to 24kHz PCM audio
 *
 * Voice preset (precomputed from speech_encoder or offline):
 *   - cond_emb.bin          (float32, conditioning embedding)
 *   - prompt_token.bin      (int64, prompt speech tokens)
 *   - speaker_embeddings.bin (float32, 192-dim speaker embedding)
 *   - speaker_features.bin  (float32, 500x80 speaker features)
 */

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <cstdint>

#include <onnxruntime/onnxruntime_cxx_api.h>

/**
 * Model variant — determines architecture constants and feature availability.
 *
 * TURBO:    GPT-2 Medium 350M, 24 layers, no exaggeration, no CFG
 * ORIGINAL: Llama 500M, 30 layers, exaggeration input on embed_tokens, cfg_weight=0.5
 */
enum class ChatterboxVariant {
    TURBO,    // GPT-2 Medium 350M, 24 layers, no exaggeration
    ORIGINAL  // Llama 500M, 30 layers, has exaggeration
};

class ChatterboxEngine {
public:
    ChatterboxEngine();
    ~ChatterboxEngine();

    // Non-copyable, non-movable (owns ONNX sessions)
    ChatterboxEngine(const ChatterboxEngine&) = delete;
    ChatterboxEngine& operator=(const ChatterboxEngine&) = delete;

    /**
     * Load the 3 ONNX model files from a directory:
     *   embed_tokens.onnx, language_model.onnx, conditional_decoder.onnx
     *
     * @param modelDir  Path to directory containing ONNX files
     * @return true on success
     */
    bool loadModels(const std::string& modelDir);

    /**
     * Load a precomputed voice preset from 4 binary files:
     *   cond_emb.bin, prompt_token.bin, speaker_embeddings.bin, speaker_features.bin
     *
     * @param styleDir  Path to directory containing voice .bin files
     * @return true on success
     */
    bool loadVoicePreset(const std::string& styleDir);

    /**
     * Full pipeline: text token IDs -> 24kHz mono int16 PCM audio.
     * Runs embed_tokens -> language_model (autoregressive) -> conditional_decoder.
     *
     * @param tokenIds  BPE-encoded text token IDs
     * @return PCM audio samples (24kHz, mono, int16)
     */
    std::vector<int16_t> synthesize(const std::vector<int64_t>& tokenIds);

    /**
     * Stage 1+2: text token IDs -> speech tokens (autoregressive generation).
     * Includes embed_tokens and language_model stages.
     *
     * @param tokenIds  BPE-encoded text token IDs
     * @return Generated speech token IDs
     */
    std::vector<int64_t> generateSpeechTokens(const std::vector<int64_t>& tokenIds);

    /**
     * Stage 3: speech tokens -> 24kHz mono int16 PCM audio.
     * Runs the conditional decoder (vocoder).
     *
     * @param speechTokens  Speech token IDs from generateSpeechTokens()
     * @return PCM audio samples (24kHz, mono, int16)
     */
    std::vector<int16_t> decodeSpeechTokens(const std::vector<int64_t>& speechTokens);

    /**
     * Release all ONNX sessions and clear voice state.
     */
    void release();

    /**
     * @return true if ONNX models are loaded
     */
    bool isLoaded() const;

    /**
     * @return true if a voice preset is loaded
     */
    bool isVoiceLoaded() const;

    /**
     * Set the repetition penalty for speech token generation.
     * Default: 1.2 (from DDATT reference)
     */
    void setRepetitionPenalty(float penalty);

    /**
     * Set the maximum number of speech tokens to generate.
     * Default: 1024 (from DDATT reference, ~42 seconds of audio)
     */
    void setMaxTokens(int maxTokens);

    /**
     * Set the model variant. MUST be called BEFORE loadModels() because it
     * determines the number of KV cache layers and I/O name arrays.
     *
     * Calling after loadModels() will rebuild I/O names but they won't match
     * the already-loaded ONNX graph — only call before loading.
     */
    void setVariant(ChatterboxVariant variant);

    /**
     * Set the emotion exaggeration parameter.
     * Only effective for ORIGINAL variant (silently ignored for TURBO).
     *
     * Values:  0.0 = flat/monotone, 1.0 = normal, 2.0 = very expressive
     * Default: 1.0
     */
    void setExaggeration(float exaggeration);

    /**
     * Request cancellation of an in-progress generation.
     * Safe to call from any thread. Resets automatically on next synthesize() call.
     */
    void requestStop();

private:
    // ── ONNX Runtime ────────────────────────────────────────────
    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> embedTokensSession_;
    std::unique_ptr<Ort::Session> languageModelSession_;
    std::unique_ptr<Ort::Session> conditionalDecoderSession_;
    Ort::MemoryInfo memoryInfo_{nullptr};

    // ── Voice preset state ──────────────────────────────────────
    std::vector<float>   condEmb_;            // conditioning embedding
    std::vector<int64_t> promptTokens_;       // prompt speech tokens
    std::vector<float>   speakerEmbeddings_;  // 192-dim speaker embedding
    std::vector<float>   speakerFeatures_;    // 500x80 speaker features

    // ── State flags ─────────────────────────────────────────────
    bool loaded_ = false;
    bool voiceLoaded_ = false;
    std::atomic<bool> stopFlag_{false};
    float repetitionPenalty_ = 1.2f;
    int maxTokens_ = 1024;

    // ── I/O name arrays (built programmatically) ────────────────
    // embed_tokens session
    static constexpr const char* kEmbedInputName  = "input_ids";
    static constexpr const char* kEmbedOutputName = "inputs_embeds";

    // Language model session — 51 inputs, 49 outputs (generated in buildIONames)
    std::vector<std::string> lmInputNameStrings_;
    std::vector<const char*> lmInputNames_;
    std::vector<std::string> lmOutputNameStrings_;
    std::vector<const char*> lmOutputNames_;

    // conditional_decoder session
    static constexpr const char* kDecoderInputNames[]  = {"speech_tokens", "speaker_embeddings", "speaker_features"};
    static constexpr const char* kDecoderOutputNames[] = {"audio"};

    // ── Model constants ─────────────────────────────────────────
    static constexpr int64_t START_SPEECH_TOKEN = 6561;
    static constexpr int64_t STOP_SPEECH_TOKEN  = 6562;
    static constexpr int64_t SILENCE_TOKEN      = 4299;
    static constexpr float   MAX_WAV_VALUE      = 32767.0f;
    static constexpr int     NUM_KV_HEADS       = 16;
    static constexpr int     HEAD_DIM           = 64;
    static constexpr int     EMBED_DIM          = 1024;

    // ── Variant-dependent state ──────────────────────────────────
    ChatterboxVariant variant_ = ChatterboxVariant::TURBO;
    int numLayers_ = 24;           // 24 for TURBO, 30 for ORIGINAL
    float exaggeration_ = 1.0f;   // emotion exaggeration (ORIGINAL only)

    // ── Helpers ─────────────────────────────────────────────────
    std::vector<float>   loadBinaryFile(const std::string& path);
    std::vector<int64_t> loadBinaryFileInt64(const std::string& path);
    void applyRepetitionPenalty(float* logits, int64_t vocabSize,
                                const std::vector<int64_t>& generated);
    void buildIONames();
};
