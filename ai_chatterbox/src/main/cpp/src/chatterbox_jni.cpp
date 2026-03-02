/**
 * JNI entry points for ai_chatterbox TTS module.
 *
 * Pattern follows ai_sd/src/main/cpp/src/ai_sd_jni.cpp:
 * - Global state singletons with mutex protection
 * - jstring <-> std::string conversion with proper release
 *
 * JNI naming: Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_<method>
 * (the _1 encodes the underscore in "ai_chatterbox")
 *
 * Threading model:
 * - g_mtx protects g_engine/g_tokenizer ownership (create, destroy, replace)
 * - nativeSynthesize() copies raw pointers under lock, then runs inference unlocked
 * - nativeStop() takes lock briefly to read g_engine, then calls requestStop() unlocked
 * - nativeRelease() takes lock and resets pointers (waits for synthesize to finish via stopFlag)
 */

#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include <thread>

#include "chatterbox_engine.h"
#include "bpe_tokenizer.h"

#define LOG_TAG "ChatterboxTTS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Global state ────────────────────────────────────────────────────
static std::unique_ptr<ChatterboxEngine> g_engine;
static std::unique_ptr<BPETokenizer> g_tokenizer;
static std::mutex g_mtx;

// Tracks whether nativeSynthesize is running (prevents release during inference)
static std::atomic<bool> g_synthesizing{false};

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Convert jstring to std::string, handling null gracefully.
 * Caller does NOT need to release — this function handles it.
 */
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (!chars) return "";
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

/**
 * Resolve callback method IDs fresh each call.
 * jmethodID resolution is a fast hash lookup — no need to cache across calls,
 * and caching is unsafe if different callback classes are passed.
 */
struct CallbackIds {
    jmethodID onSpeechTokenProgress = nullptr;
    jmethodID onAudioReady = nullptr;
    jmethodID onError = nullptr;
};

static bool resolveCallbackIds(JNIEnv* env, jobject callback, CallbackIds& ids) {
    jclass cls = env->GetObjectClass(callback);
    if (!cls) {
        LOGE("resolveCallbackIds: failed to get callback class");
        return false;
    }

    ids.onSpeechTokenProgress = env->GetMethodID(cls, "onSpeechTokenProgress", "(I)V");
    ids.onAudioReady = env->GetMethodID(cls, "onAudioReady", "([SI)V");
    ids.onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");

    env->DeleteLocalRef(cls);

    if (!ids.onSpeechTokenProgress || !ids.onAudioReady || !ids.onError) {
        LOGE("resolveCallbackIds: failed to resolve one or more callback method IDs");
        return false;
    }

    return true;
}

/**
 * Report an error to the callback (if available) and log it.
 */
static void reportError(JNIEnv* env, jobject callback, jmethodID errorMethod, const char* msg) {
    LOGE("%s", msg);
    if (callback && errorMethod) {
        jstring jmsg = env->NewStringUTF(msg);
        env->CallVoidMethod(callback, errorMethod, jmsg);
        env->DeleteLocalRef(jmsg);
    }
}

// ── JNI Methods ─────────────────────────────────────────────────────

extern "C" {

// ────────────────── Lifecycle ──────────────────

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeLoadModels(
        JNIEnv* env, jobject /* thiz */, jstring jModelDir) {

    std::lock_guard<std::mutex> lock(g_mtx);

    std::string modelDir = jstring_to_string(env, jModelDir);
    if (modelDir.empty()) {
        LOGE("nativeLoadModels: modelDir is empty");
        return JNI_FALSE;
    }

    LOGI("nativeLoadModels: loading from %s", modelDir.c_str());

    try {
        if (!g_engine) {
            g_engine = std::make_unique<ChatterboxEngine>();
        }

        bool ok = g_engine->loadModels(modelDir);
        if (!ok) {
            LOGE("nativeLoadModels: engine loadModels failed");
            return JNI_FALSE;
        }

        LOGI("nativeLoadModels: success");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeLoadModels: exception: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeLoadVoicePreset(
        JNIEnv* env, jobject /* thiz */, jstring jStyleDir) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_engine) {
        LOGE("nativeLoadVoicePreset: engine not initialized");
        return JNI_FALSE;
    }

    std::string styleDir = jstring_to_string(env, jStyleDir);
    if (styleDir.empty()) {
        LOGE("nativeLoadVoicePreset: styleDir is empty");
        return JNI_FALSE;
    }

    LOGI("nativeLoadVoicePreset: loading from %s", styleDir.c_str());

    try {
        bool ok = g_engine->loadVoicePreset(styleDir);
        if (!ok) {
            LOGE("nativeLoadVoicePreset: loadVoicePreset failed");
            return JNI_FALSE;
        }

        LOGI("nativeLoadVoicePreset: success");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeLoadVoicePreset: exception: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeLoadTokenizer(
        JNIEnv* env, jobject /* thiz */, jstring jTokenizerPath) {

    std::lock_guard<std::mutex> lock(g_mtx);

    std::string tokenizerPath = jstring_to_string(env, jTokenizerPath);
    if (tokenizerPath.empty()) {
        LOGE("nativeLoadTokenizer: tokenizerPath is empty");
        return JNI_FALSE;
    }

    LOGI("nativeLoadTokenizer: loading from %s", tokenizerPath.c_str());

    try {
        if (!g_tokenizer) {
            g_tokenizer = std::make_unique<BPETokenizer>();
        }

        bool ok = g_tokenizer->loadFromFile(tokenizerPath);
        if (!ok) {
            LOGE("nativeLoadTokenizer: loadFromFile failed");
            return JNI_FALSE;
        }

        LOGI("nativeLoadTokenizer: success (vocab=%zu, merges=%zu, added=%zu)",
             g_tokenizer->vocabSize(), g_tokenizer->mergeCount(),
             g_tokenizer->addedTokenCount());
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("nativeLoadTokenizer: exception: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeRelease(
        JNIEnv* /* env */, jobject /* thiz */) {

    LOGI("nativeRelease: releasing engine and tokenizer");

    // If synthesis is running, request stop and wait for it to finish
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_engine) {
            g_engine->requestStop();
        }
    }

    // Spin-wait for synthesis to finish (stop flag ensures it exits quickly)
    int waitCount = 0;
    while (g_synthesizing.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (++waitCount > 500) { // 5 second timeout
            LOGE("nativeRelease: timed out waiting for synthesis to finish");
            break;
        }
    }

    std::lock_guard<std::mutex> lock(g_mtx);

    if (g_engine) {
        try {
            g_engine->release();
        } catch (const std::exception& e) {
            LOGE("nativeRelease: engine release exception: %s", e.what());
        }
        g_engine.reset();
    }

    g_tokenizer.reset();

    LOGI("nativeRelease: done");
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeIsLoaded(
        JNIEnv* /* env */, jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(g_mtx);

    return (g_engine && g_engine->isLoaded()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeIsVoiceLoaded(
        JNIEnv* /* env */, jobject /* thiz */) {

    std::lock_guard<std::mutex> lock(g_mtx);

    return (g_engine && g_engine->isVoiceLoaded()) ? JNI_TRUE : JNI_FALSE;
}

// ────────────────── Synthesis ──────────────────

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeSynthesize(
        JNIEnv* env, jobject /* thiz */, jstring jText, jobject callback) {

    // Resolve callback method IDs fresh each call (safe across classloaders)
    CallbackIds cbIds;
    if (!resolveCallbackIds(env, callback, cbIds)) {
        LOGE("nativeSynthesize: failed to resolve callback methods");
        return JNI_FALSE;
    }

    // Grab raw pointers under lock, then release lock before long-running inference.
    // g_synthesizing flag prevents nativeRelease from destroying objects mid-inference.
    ChatterboxEngine* engine = nullptr;
    BPETokenizer* tokenizer = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_mtx);

        if (!g_engine || !g_engine->isLoaded()) {
            reportError(env, callback, cbIds.onError, "nativeSynthesize: engine not loaded");
            return JNI_FALSE;
        }

        if (!g_engine->isVoiceLoaded()) {
            reportError(env, callback, cbIds.onError, "nativeSynthesize: voice preset not loaded");
            return JNI_FALSE;
        }

        if (!g_tokenizer) {
            reportError(env, callback, cbIds.onError, "nativeSynthesize: tokenizer not loaded");
            return JNI_FALSE;
        }

        engine = g_engine.get();
        tokenizer = g_tokenizer.get();
        g_synthesizing.store(true);
    }
    // Lock released — other JNI calls (isLoaded, setConfig, stop) can proceed.
    // nativeRelease will wait for g_synthesizing to clear.

    std::string text = jstring_to_string(env, jText);
    if (text.empty()) {
        g_synthesizing.store(false);
        reportError(env, callback, cbIds.onError, "nativeSynthesize: text is empty");
        return JNI_FALSE;
    }

    LOGI("nativeSynthesize: text length=%zu", text.size());

    jboolean result = JNI_FALSE;
    try {
        // Step 1: Tokenize
        auto tokenIds = tokenizer->encode(text, true);
        LOGI("nativeSynthesize: tokenized %zu tokens", tokenIds.size());

        if (tokenIds.empty()) {
            reportError(env, callback, cbIds.onError,
                        "nativeSynthesize: tokenization produced 0 tokens");
            g_synthesizing.store(false);
            return JNI_FALSE;
        }

        // Step 2: Generate speech tokens (autoregressive — the long-running part)
        auto speechTokens = engine->generateSpeechTokens(tokenIds);
        LOGI("nativeSynthesize: generated %zu speech tokens", speechTokens.size());

        // Report speech token progress
        env->CallVoidMethod(callback, cbIds.onSpeechTokenProgress,
                            static_cast<jint>(speechTokens.size()));
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            LOGE("nativeSynthesize: callback exception in onSpeechTokenProgress");
            g_synthesizing.store(false);
            return JNI_FALSE;
        }

        if (speechTokens.empty()) {
            reportError(env, callback, cbIds.onError,
                        "nativeSynthesize: speech token generation produced 0 tokens");
            g_synthesizing.store(false);
            return JNI_FALSE;
        }

        // Step 3: Decode speech tokens to PCM audio
        auto pcm = engine->decodeSpeechTokens(speechTokens);
        LOGI("nativeSynthesize: decoded %zu PCM samples", pcm.size());

        if (pcm.empty()) {
            reportError(env, callback, cbIds.onError,
                        "nativeSynthesize: decoder produced 0 PCM samples");
            g_synthesizing.store(false);
            return JNI_FALSE;
        }

        // Step 4: Deliver result via callback
        jshortArray jpcm = env->NewShortArray(static_cast<jsize>(pcm.size()));
        if (!jpcm) {
            reportError(env, callback, cbIds.onError,
                        "nativeSynthesize: failed to allocate ShortArray");
            g_synthesizing.store(false);
            return JNI_FALSE;
        }

        env->SetShortArrayRegion(jpcm, 0, static_cast<jsize>(pcm.size()),
                                 reinterpret_cast<const jshort*>(pcm.data()));

        // 24000 Hz mono PCM — the Chatterbox model output sample rate
        env->CallVoidMethod(callback, cbIds.onAudioReady, jpcm, 24000);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            LOGE("nativeSynthesize: callback exception in onAudioReady");
        }

        env->DeleteLocalRef(jpcm);

        LOGI("nativeSynthesize: success");
        result = JNI_TRUE;

    } catch (const std::exception& e) {
        std::string msg = std::string("nativeSynthesize: exception: ") + e.what();
        reportError(env, callback, cbIds.onError, msg.c_str());
    }

    g_synthesizing.store(false);
    return result;
}

/**
 * Request stop — brief lock to read g_engine safely, then call requestStop() unlocked.
 * requestStop() writes an atomic bool, so it's safe after we've verified the pointer.
 * The lock prevents a TOCTOU race with nativeRelease() resetting g_engine.
 */
JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeStop(
        JNIEnv* /* env */, jobject /* thiz */) {

    LOGI("nativeStop: requesting stop");
    ChatterboxEngine* engine = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        engine = g_engine.get();
    }
    if (engine) {
        engine->requestStop();
    }
}

// ────────────────── Tokenizer ──────────────────

JNIEXPORT jlongArray JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeTokenize(
        JNIEnv* env, jobject /* thiz */, jstring jText) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_tokenizer) {
        LOGE("nativeTokenize: tokenizer not loaded");
        return nullptr;
    }

    std::string text = jstring_to_string(env, jText);
    if (text.empty()) {
        LOGE("nativeTokenize: text is empty");
        return nullptr;
    }

    try {
        auto tokenIds = g_tokenizer->encode(text, true);

        jlongArray result = env->NewLongArray(static_cast<jsize>(tokenIds.size()));
        if (!result) {
            LOGE("nativeTokenize: failed to allocate LongArray");
            return nullptr;
        }

        env->SetLongArrayRegion(result, 0, static_cast<jsize>(tokenIds.size()),
                                reinterpret_cast<const jlong*>(tokenIds.data()));
        return result;

    } catch (const std::exception& e) {
        LOGE("nativeTokenize: exception: %s", e.what());
        return nullptr;
    }
}

// ────────────────── Config ──────────────────

JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeSetRepetitionPenalty(
        JNIEnv* /* env */, jobject /* thiz */, jfloat penalty) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_engine) {
        LOGE("nativeSetRepetitionPenalty: engine not initialized");
        return;
    }

    LOGI("nativeSetRepetitionPenalty: %.3f", static_cast<float>(penalty));
    g_engine->setRepetitionPenalty(static_cast<float>(penalty));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeSetMaxTokens(
        JNIEnv* /* env */, jobject /* thiz */, jint maxTokens) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_engine) {
        LOGE("nativeSetMaxTokens: engine not initialized");
        return;
    }

    LOGI("nativeSetMaxTokens: %d", static_cast<int>(maxTokens));
    g_engine->setMaxTokens(static_cast<int>(maxTokens));
}

// ────────────────── Variant & Exaggeration ──────────────────

JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeSetVariant(
        JNIEnv* /* env */, jobject /* thiz */, jint variant) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_engine) {
        // Engine not yet created — create it so setVariant can rebuild I/O names
        g_engine = std::make_unique<ChatterboxEngine>();
    }

    LOGI("nativeSetVariant: %d (%s)", static_cast<int>(variant),
         variant == 1 ? "ORIGINAL" : "TURBO");
    g_engine->setVariant(variant == 1 ? ChatterboxVariant::ORIGINAL : ChatterboxVariant::TURBO);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeSetExaggeration(
        JNIEnv* /* env */, jobject /* thiz */, jfloat exaggeration) {

    std::lock_guard<std::mutex> lock(g_mtx);

    if (!g_engine) {
        LOGE("nativeSetExaggeration: engine not initialized");
        return;
    }

    LOGI("nativeSetExaggeration: %.2f", static_cast<float>(exaggeration));
    g_engine->setExaggeration(static_cast<float>(exaggeration));
}

} // extern "C"
