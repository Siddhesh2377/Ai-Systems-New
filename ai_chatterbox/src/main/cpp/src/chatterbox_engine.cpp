/**
 * ChatterboxEngine — ONNX Runtime inference engine for Chatterbox TTS.
 *
 * Ported from DDATT/Chatterbox-turbo-cpp with Android adaptations.
 * See chatterbox_engine.h for class documentation.
 */

#include "chatterbox_engine.h"

#include <android/log.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

// ── Android logging ─────────────────────────────────────────────
#define LOG_TAG "ChatterboxTTS"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ═══════════════════════════════════════════════════════════════
//  Construction / Destruction
// ═══════════════════════════════════════════════════════════════

ChatterboxEngine::ChatterboxEngine() {
    buildIONames();
}

ChatterboxEngine::~ChatterboxEngine() {
    release();
}

// ═══════════════════════════════════════════════════════════════
//  I/O Name Generation
// ═══════════════════════════════════════════════════════════════

void ChatterboxEngine::buildIONames() {
    // Language model inputs: inputs_embeds, attention_mask, position_ids,
    //   then 48 KV cache tensors: past_key_values.{0..23}.key, past_key_values.{0..23}.value
    lmInputNameStrings_.clear();
    lmInputNameStrings_.push_back("inputs_embeds");
    lmInputNameStrings_.push_back("attention_mask");
    lmInputNameStrings_.push_back("position_ids");

    for (int i = 0; i < NUM_LAYERS; i++) {
        lmInputNameStrings_.push_back("past_key_values." + std::to_string(i) + ".key");
        lmInputNameStrings_.push_back("past_key_values." + std::to_string(i) + ".value");
    }

    // Language model outputs: logits,
    //   then 48 present KV tensors: present.{0..23}.key, present.{0..23}.value
    lmOutputNameStrings_.clear();
    lmOutputNameStrings_.push_back("logits");

    for (int i = 0; i < NUM_LAYERS; i++) {
        lmOutputNameStrings_.push_back("present." + std::to_string(i) + ".key");
        lmOutputNameStrings_.push_back("present." + std::to_string(i) + ".value");
    }

    // Build const char* pointer arrays (stable because strings are in vector, not reallocated after this)
    lmInputNames_.clear();
    lmInputNames_.reserve(lmInputNameStrings_.size());
    for (const auto& s : lmInputNameStrings_) {
        lmInputNames_.push_back(s.c_str());
    }

    lmOutputNames_.clear();
    lmOutputNames_.reserve(lmOutputNameStrings_.size());
    for (const auto& s : lmOutputNameStrings_) {
        lmOutputNames_.push_back(s.c_str());
    }

    LOGD("Built LM I/O names: %zu inputs, %zu outputs",
         lmInputNames_.size(), lmOutputNames_.size());
}

// ═══════════════════════════════════════════════════════════════
//  Model Loading
// ═══════════════════════════════════════════════════════════════

bool ChatterboxEngine::loadModels(const std::string& modelDir) {
    try {
        LOGI("Loading Chatterbox models from: %s", modelDir.c_str());

        // Create ORT environment
        env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ChatterboxTTS");
        env_.DisableTelemetryEvents();

        // Session options: enable all graph optimizations for mobile performance
        // (DDATT disables these — we ENABLE them for Android)
        sessionOptions_ = Ort::SessionOptions();
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Memory arena left enabled (default) — DDATT disables it, we keep it for mobile
        sessionOptions_.SetIntraOpNumThreads(4);

        // Create memory info
        memoryInfo_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Load each model
        std::string embedTokensPath       = modelDir + "/embed_tokens.onnx";
        std::string languageModelPath      = modelDir + "/language_model.onnx";
        std::string conditionalDecoderPath = modelDir + "/conditional_decoder.onnx";

        LOGI("Loading embed_tokens...");
        embedTokensSession_ = std::make_unique<Ort::Session>(
            env_, embedTokensPath.c_str(), sessionOptions_);

        LOGI("Loading language_model...");
        languageModelSession_ = std::make_unique<Ort::Session>(
            env_, languageModelPath.c_str(), sessionOptions_);

        LOGI("Loading conditional_decoder...");
        conditionalDecoderSession_ = std::make_unique<Ort::Session>(
            env_, conditionalDecoderPath.c_str(), sessionOptions_);

        loaded_ = true;
        LOGI("All Chatterbox models loaded successfully");
        return true;

    } catch (const Ort::Exception& e) {
        LOGE("ORT error loading models: %s", e.what());
        release();
        return false;
    } catch (const std::exception& e) {
        LOGE("Error loading models: %s", e.what());
        release();
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Voice Preset Loading
// ═══════════════════════════════════════════════════════════════

bool ChatterboxEngine::loadVoicePreset(const std::string& styleDir) {
    try {
        LOGI("Loading voice preset from: %s", styleDir.c_str());

        condEmb_            = loadBinaryFile(styleDir + "/cond_emb.bin");
        promptTokens_       = loadBinaryFileInt64(styleDir + "/prompt_token.bin");
        speakerEmbeddings_  = loadBinaryFile(styleDir + "/speaker_embeddings.bin");
        speakerFeatures_    = loadBinaryFile(styleDir + "/speaker_features.bin");

        if (condEmb_.empty() || promptTokens_.empty() ||
            speakerEmbeddings_.empty() || speakerFeatures_.empty()) {
            LOGE("One or more voice preset files are empty or missing");
            voiceLoaded_ = false;
            return false;
        }

        voiceLoaded_ = true;
        LOGI("Voice preset loaded: condEmb=%zu, promptTokens=%zu, "
             "speakerEmb=%zu, speakerFeat=%zu",
             condEmb_.size(), promptTokens_.size(),
             speakerEmbeddings_.size(), speakerFeatures_.size());
        return true;

    } catch (const std::exception& e) {
        LOGE("Error loading voice preset: %s", e.what());
        voiceLoaded_ = false;
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════
//  Full Synthesis Pipeline
// ═══════════════════════════════════════════════════════════════

std::vector<int16_t> ChatterboxEngine::synthesize(const std::vector<int64_t>& tokenIds) {
    stopFlag_.store(false);

    // Stage 1+2: generate speech tokens
    auto speechTokens = generateSpeechTokens(tokenIds);
    if (speechTokens.empty()) {
        LOGW("Speech token generation produced no tokens");
        return {};
    }

    if (stopFlag_.load()) {
        LOGI("Synthesis cancelled after token generation");
        return {};
    }

    // Stage 3: decode to audio
    return decodeSpeechTokens(speechTokens);
}

// ═══════════════════════════════════════════════════════════════
//  Stage 1+2: Autoregressive Speech Token Generation
// ═══════════════════════════════════════════════════════════════

std::vector<int64_t> ChatterboxEngine::generateSpeechTokens(
        const std::vector<int64_t>& tokenIds) {

    if (!loaded_ || !voiceLoaded_) {
        LOGE("Models or voice not loaded");
        return {};
    }

    try {
        std::vector<int64_t> generatedTokens;
        generatedTokens.push_back(START_SPEECH_TOKEN);

        int64_t condEmbLength = static_cast<int64_t>(condEmb_.size() / EMBED_DIM);
        int64_t currentSeqLen = static_cast<int64_t>(tokenIds.size()) + condEmbLength;
        int64_t currentPosition = currentSeqLen - 1;

        std::vector<Ort::Value> pastKeyValues;
        int64_t nextTokenId = 0;

        LOGI("Starting speech token generation: inputTokens=%zu, condEmbLen=%lld, maxTokens=%d",
             tokenIds.size(), (long long)condEmbLength, maxTokens_);

        for (int i = 0; i < maxTokens_; i++) {
            // ── Check cancellation ──
            if (stopFlag_.load()) {
                LOGI("Generation cancelled at step %d", i);
                return generatedTokens;
            }

            std::vector<float>   currentEmbedsData;
            std::vector<int64_t> currentEmbedsShape;

            if (i == 0) {
                // ── First iteration: embed all input tokens, prepend condEmb ──
                std::vector<int64_t> inputIdsCopy(tokenIds.begin(), tokenIds.end());
                std::vector<int64_t> embedDim = {
                    1, static_cast<int64_t>(inputIdsCopy.size())
                };

                Ort::Value embedInput = Ort::Value::CreateTensor<int64_t>(
                    memoryInfo_,
                    inputIdsCopy.data(), inputIdsCopy.size(),
                    embedDim.data(), embedDim.size());

                const char* inputNames[]  = { kEmbedInputName };
                const char* outputNames[] = { kEmbedOutputName };

                auto output = embedTokensSession_->Run(
                    Ort::RunOptions{nullptr},
                    inputNames, &embedInput, 1,
                    outputNames, 1);

                const float* promptData = output.front().GetTensorData<float>();
                size_t promptSize = output.front().GetTensorTypeAndShapeInfo().GetElementCount();

                // Prepend conditioning embedding, then prompt embeddings
                currentEmbedsData = condEmb_;
                currentEmbedsData.insert(currentEmbedsData.end(),
                                         promptData, promptData + promptSize);
                currentEmbedsShape = {
                    1,
                    static_cast<int64_t>(tokenIds.size()) + condEmbLength,
                    EMBED_DIM
                };

                LOGD("First step: embedsSize=%zu, shape=[1,%lld,%d]",
                     currentEmbedsData.size(),
                     (long long)currentEmbedsShape[1], EMBED_DIM);
            } else {
                // ── Subsequent iterations: embed single next token ──
                std::vector<int64_t> nextVec = { nextTokenId };
                std::vector<int64_t> embedDim = { 1, 1 };

                Ort::Value embedInput = Ort::Value::CreateTensor<int64_t>(
                    memoryInfo_,
                    nextVec.data(), nextVec.size(),
                    embedDim.data(), embedDim.size());

                const char* inputNames[]  = { kEmbedInputName };
                const char* outputNames[] = { kEmbedOutputName };

                auto output = embedTokensSession_->Run(
                    Ort::RunOptions{nullptr},
                    inputNames, &embedInput, 1,
                    outputNames, 1);

                const float* newData = output.front().GetTensorData<float>();
                size_t newSize = output.front().GetTensorTypeAndShapeInfo().GetElementCount();

                currentEmbedsData.assign(newData, newData + newSize);
                currentEmbedsShape = { 1, 1, EMBED_DIM };
            }

            // ── Build language model inputs ──
            std::vector<Ort::Value> lmInputs;

            // Input 0: inputs_embeds
            lmInputs.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo_,
                currentEmbedsData.data(), currentEmbedsData.size(),
                currentEmbedsShape.data(), currentEmbedsShape.size()));

            // Input 1: attention_mask (all 1s, length = currentSeqLen)
            std::vector<int64_t> mask(currentSeqLen, 1);
            std::vector<int64_t> maskShape = { 1, currentSeqLen };
            lmInputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo_,
                mask.data(), mask.size(),
                maskShape.data(), maskShape.size()));

            // Input 2: position_ids
            std::vector<int64_t> posIds;
            std::vector<int64_t> posShape;
            if (i == 0) {
                posIds.resize(currentSeqLen);
                for (int64_t k = 0; k < currentSeqLen; ++k) posIds[k] = k;
                posShape = { 1, currentSeqLen };
            } else {
                currentPosition++;
                posIds.push_back(currentPosition);
                posShape = { 1, 1 };
            }
            lmInputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memoryInfo_,
                posIds.data(), posIds.size(),
                posShape.data(), posShape.size()));

            // Input 3..50: past_key_values (48 tensors = 24 layers x 2 key/value)
            if (i == 0) {
                // First step: empty KV cache tensors with seq_len=0
                for (int j = 0; j < NUM_LAYERS * 2; j++) {
                    std::vector<int64_t> pastShape = {
                        1, NUM_KV_HEADS, 0, HEAD_DIM
                    };
                    // Empty tensor (0 elements)
                    pastKeyValues.push_back(Ort::Value::CreateTensor<float>(
                        memoryInfo_,
                        nullptr, 0,
                        pastShape.data(), pastShape.size()));
                }
            }

            for (auto& kv : pastKeyValues) {
                lmInputs.push_back(std::move(kv));
            }
            pastKeyValues.clear();

            // ── Run language model ──
            auto lmOutput = languageModelSession_->Run(
                Ort::RunOptions{nullptr},
                lmInputNames_.data(), lmInputs.data(), lmInputs.size(),
                lmOutputNames_.data(), lmOutputNames_.size());

            // ── Extract logits and sample ──
            float* logitsRaw = lmOutput[0].GetTensorMutableData<float>();
            auto logitsShape = lmOutput[0].GetTensorTypeAndShapeInfo().GetShape();
            int64_t vocabSize = logitsShape[2];
            int64_t seqDim    = logitsShape[1];

            // Get logits for the last position
            float* lastLogits = logitsRaw + (seqDim - 1) * vocabSize;

            // Apply repetition penalty
            applyRepetitionPenalty(lastLogits, vocabSize, generatedTokens);

            // Greedy argmax sampling
            int64_t bestId = 0;
            float maxScore = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < vocabSize; v++) {
                if (lastLogits[v] > maxScore) {
                    maxScore = lastLogits[v];
                    bestId = v;
                }
            }

            nextTokenId = bestId;

            // Check for stop token
            if (nextTokenId == STOP_SPEECH_TOKEN) {
                LOGI("Stop token reached at step %d, generated %zu speech tokens",
                     i, generatedTokens.size());
                break;
            }

            generatedTokens.push_back(nextTokenId);

            // ── Update KV cache from language model outputs ──
            // Output[0] = logits, Output[1..48] = present KV cache tensors
            for (size_t k = 1; k < lmOutput.size(); k++) {
                pastKeyValues.push_back(std::move(lmOutput[k]));
            }

            currentSeqLen++;

            if (i % 100 == 0 && i > 0) {
                LOGD("Generation progress: step %d, tokens=%zu", i, generatedTokens.size());
            }
        }

        LOGI("Speech token generation complete: %zu tokens", generatedTokens.size());
        return generatedTokens;

    } catch (const Ort::Exception& e) {
        LOGE("ORT error in generateSpeechTokens: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        LOGE("Error in generateSpeechTokens: %s", e.what());
        return {};
    }
}

// ═══════════════════════════════════════════════════════════════
//  Stage 3: Conditional Decoder (Vocoder)
// ═══════════════════════════════════════════════════════════════

std::vector<int16_t> ChatterboxEngine::decodeSpeechTokens(
        const std::vector<int64_t>& speechTokens) {

    if (!loaded_ || !voiceLoaded_) {
        LOGE("Models or voice not loaded for decoding");
        return {};
    }

    try {
        // Build full speech token sequence:
        //   promptTokens + generatedTokens (skip START_SPEECH_TOKEN) + 3x silence
        std::vector<int64_t> fullSpeechTokens;
        fullSpeechTokens.insert(fullSpeechTokens.end(),
                                promptTokens_.begin(), promptTokens_.end());

        // Skip the first token (START_SPEECH_TOKEN) from generated tokens
        if (speechTokens.size() > 1) {
            fullSpeechTokens.insert(fullSpeechTokens.end(),
                                    speechTokens.begin() + 1, speechTokens.end());
        }

        // Append silence tokens
        fullSpeechTokens.push_back(SILENCE_TOKEN);
        fullSpeechTokens.push_back(SILENCE_TOKEN);
        fullSpeechTokens.push_back(SILENCE_TOKEN);

        LOGI("Decoding %zu speech tokens (prompt=%zu + generated=%zu + silence=3)",
             fullSpeechTokens.size(), promptTokens_.size(),
             speechTokens.size() > 1 ? speechTokens.size() - 1 : 0);

        // Build decoder inputs
        std::vector<Ort::Value> inputs;

        // Input 0: speech_tokens [1, seq_len]
        std::vector<int64_t> speechDim = {
            1, static_cast<int64_t>(fullSpeechTokens.size())
        };
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memoryInfo_,
            fullSpeechTokens.data(), fullSpeechTokens.size(),
            speechDim.data(), speechDim.size()));

        // Input 1: speaker_embeddings [1, 192]
        std::vector<int64_t> speakerEmbDim = { 1, 192 };
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_,
            speakerEmbeddings_.data(), speakerEmbeddings_.size(),
            speakerEmbDim.data(), speakerEmbDim.size()));

        // Input 2: speaker_features [1, 500, 80]
        std::vector<int64_t> speakerFeatDim = { 1, 500, 80 };
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_,
            speakerFeatures_.data(), speakerFeatures_.size(),
            speakerFeatDim.data(), speakerFeatDim.size()));

        // Run conditional decoder
        auto output = conditionalDecoderSession_->Run(
            Ort::RunOptions{nullptr},
            kDecoderInputNames, inputs.data(), inputs.size(),
            kDecoderOutputNames, 1);

        // Extract audio data
        const float* audioData = output.front().GetTensorData<float>();
        auto audioShape = output.front().GetTensorTypeAndShapeInfo().GetShape();
        int64_t numSamples = audioShape[audioShape.size() - 1];

        LOGI("Decoder output: %lld samples (%.2f seconds at 24kHz)",
             (long long)numSamples, (float)numSamples / 24000.0f);

        // Convert float32 -> int16 PCM with clamping
        std::vector<int16_t> audioBuffer;
        audioBuffer.reserve(numSamples);

        for (int64_t i = 0; i < numSamples; i++) {
            float sample = audioData[i] * MAX_WAV_VALUE;
            sample = std::clamp(sample,
                                static_cast<float>(std::numeric_limits<int16_t>::min()),
                                static_cast<float>(std::numeric_limits<int16_t>::max()));
            audioBuffer.push_back(static_cast<int16_t>(sample));
        }

        return audioBuffer;

    } catch (const Ort::Exception& e) {
        LOGE("ORT error in decodeSpeechTokens: %s", e.what());
        return {};
    } catch (const std::exception& e) {
        LOGE("Error in decodeSpeechTokens: %s", e.what());
        return {};
    }
}

// ═══════════════════════════════════════════════════════════════
//  Lifecycle
// ═══════════════════════════════════════════════════════════════

void ChatterboxEngine::release() {
    conditionalDecoderSession_.reset();
    languageModelSession_.reset();
    embedTokensSession_.reset();

    condEmb_.clear();
    promptTokens_.clear();
    speakerEmbeddings_.clear();
    speakerFeatures_.clear();

    loaded_ = false;
    voiceLoaded_ = false;
    stopFlag_.store(false);

    LOGI("ChatterboxEngine released");
}

bool ChatterboxEngine::isLoaded() const {
    return loaded_;
}

bool ChatterboxEngine::isVoiceLoaded() const {
    return voiceLoaded_;
}

void ChatterboxEngine::setRepetitionPenalty(float penalty) {
    repetitionPenalty_ = penalty;
    LOGD("Repetition penalty set to %.2f", penalty);
}

void ChatterboxEngine::setMaxTokens(int maxTokens) {
    maxTokens_ = maxTokens;
    LOGD("Max tokens set to %d", maxTokens);
}

void ChatterboxEngine::requestStop() {
    stopFlag_.store(true);
    LOGI("Stop requested");
}

// ═══════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════

std::vector<float> ChatterboxEngine::loadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOGE("Failed to open binary file: %s", path.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(float) != 0) {
        LOGW("Binary file size not aligned to float32: %s (%lld bytes)",
             path.c_str(), (long long)size);
    }

    std::vector<float> data(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        LOGE("Failed to read binary file: %s", path.c_str());
        return {};
    }

    LOGD("Loaded float32 binary: %s (%zu elements)", path.c_str(), data.size());
    return data;
}

std::vector<int64_t> ChatterboxEngine::loadBinaryFileInt64(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOGE("Failed to open binary file: %s", path.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(int64_t) != 0) {
        LOGW("Binary file size not aligned to int64: %s (%lld bytes)",
             path.c_str(), (long long)size);
    }

    std::vector<int64_t> data(size / sizeof(int64_t));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        LOGE("Failed to read binary file: %s", path.c_str());
        return {};
    }

    LOGD("Loaded int64 binary: %s (%zu elements)", path.c_str(), data.size());
    return data;
}

void ChatterboxEngine::applyRepetitionPenalty(float* logits, int64_t vocabSize,
                                               const std::vector<int64_t>& generated) {
    std::unordered_set<int64_t> seenTokens;
    for (auto id : generated) {
        seenTokens.insert(id);
    }

    for (int64_t id : seenTokens) {
        if (id < 0 || id >= vocabSize) continue;  // bounds check (safety)
        float& score = logits[id];
        if (score < 0.0f) {
            score *= repetitionPenalty_;
        } else {
            score /= repetitionPenalty_;
        }
    }
}
