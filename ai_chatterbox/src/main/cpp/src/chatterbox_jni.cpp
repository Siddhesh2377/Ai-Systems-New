/**
 * JNI entry points for ai_chatterbox TTS module.
 *
 * Pattern follows ai_sd/src/main/cpp/src/ai_sd_jni.cpp:
 * - Global state singletons with mutex protection
 * - Cached JNI callback method IDs
 * - jstring <-> std::string conversion with proper release
 *
 * JNI naming: Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_<method>
 * (the _1 encodes the underscore in "ai_chatterbox")
 */

#include <jni.h>
#include <android/log.h>
#include <mutex>
#include <memory>
#include <string>

#include "chatterbox_engine.h"
#include "bpe_tokenizer.h"

#define LOG_TAG "ChatterboxTTS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Global state ────────────────────────────────────────────────────
static std::unique_ptr<ChatterboxEngine> g_engine;
static std::unique_ptr<BPETokenizer> g_tokenizer;
static std::mutex g_mtx;

// Cached JNI callback method IDs (populated on first use)
static jmethodID g_onSpeechTokenProgress = nullptr;
static jmethodID g_onAudioReady = nullptr;
static jmethodID g_onError = nullptr;

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
 * Cache callback method IDs on first use.
 * Returns true if all method IDs are valid.
 */
static bool ensureCallbackIds(JNIEnv* env, jobject callback) {
    if (g_onSpeechTokenProgress != nullptr) return true;

    jclass cls = env->GetObjectClass(callback);
    if (!cls) {
        LOGE("ensureCallbackIds: failed to get callback class");
        return false;
    }

    g_onSpeechTokenProgress = env->GetMethodID(cls, "onSpeechTokenProgress", "(I)V");
    g_onAudioReady = env->GetMethodID(cls, "onAudioReady", "([SI)V");
    g_onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");

    env->DeleteLocalRef(cls);

    if (!g_onSpeechTokenProgress || !g_onAudioReady || !g_onError) {
        LOGE("ensureCallbackIds: failed to resolve one or more callback method IDs");
        g_onSpeechTokenProgress = nullptr;
        g_onAudioReady = nullptr;
        g_onError = nullptr;
        return false;
    }

    return true;
}

/**
 * Report an error to the callback (if available) and log it.
 */
static void reportError(JNIEnv* env, jobject callback, const char* msg) {
    LOGE("%s", msg);
    if (callback && g_onError) {
        jstring jmsg = env->NewStringUTF(msg);
        env->CallVoidMethod(callback, g_onError, jmsg);
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

    std::lock_guard<std::mutex> lock(g_mtx);

    LOGI("nativeRelease: releasing engine and tokenizer");

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

    std::lock_guard<std::mutex> lock(g_mtx);

    // Validate prerequisites
    if (!g_engine || !g_engine->isLoaded()) {
        reportError(env, callback, "nativeSynthesize: engine not loaded");
        return JNI_FALSE;
    }

    if (!g_engine->isVoiceLoaded()) {
        reportError(env, callback, "nativeSynthesize: voice preset not loaded");
        return JNI_FALSE;
    }

    if (!g_tokenizer) {
        reportError(env, callback, "nativeSynthesize: tokenizer not loaded");
        return JNI_FALSE;
    }

    // Cache callback method IDs
    if (!ensureCallbackIds(env, callback)) {
        LOGE("nativeSynthesize: failed to resolve callback methods");
        return JNI_FALSE;
    }

    std::string text = jstring_to_string(env, jText);
    if (text.empty()) {
        reportError(env, callback, "nativeSynthesize: text is empty");
        return JNI_FALSE;
    }

    LOGI("nativeSynthesize: text length=%zu", text.size());

    try {
        // Step 1: Tokenize
        auto tokenIds = g_tokenizer->encode(text, true);
        LOGI("nativeSynthesize: tokenized %zu tokens", tokenIds.size());

        if (tokenIds.empty()) {
            reportError(env, callback, "nativeSynthesize: tokenization produced 0 tokens");
            return JNI_FALSE;
        }

        // Step 2: Generate speech tokens (report progress via callback)
        auto speechTokens = g_engine->generateSpeechTokens(tokenIds);
        LOGI("nativeSynthesize: generated %zu speech tokens", speechTokens.size());

        // Report speech token progress
        env->CallVoidMethod(callback, g_onSpeechTokenProgress,
                            static_cast<jint>(speechTokens.size()));

        if (speechTokens.empty()) {
            reportError(env, callback, "nativeSynthesize: speech token generation produced 0 tokens");
            return JNI_FALSE;
        }

        // Step 3: Decode speech tokens to PCM audio
        auto pcm = g_engine->decodeSpeechTokens(speechTokens);
        LOGI("nativeSynthesize: decoded %zu PCM samples", pcm.size());

        if (pcm.empty()) {
            reportError(env, callback, "nativeSynthesize: decoder produced 0 PCM samples");
            return JNI_FALSE;
        }

        // Step 4: Deliver result via callback
        jshortArray jpcm = env->NewShortArray(static_cast<jsize>(pcm.size()));
        if (!jpcm) {
            reportError(env, callback, "nativeSynthesize: failed to allocate ShortArray");
            return JNI_FALSE;
        }

        env->SetShortArrayRegion(jpcm, 0, static_cast<jsize>(pcm.size()),
                                 reinterpret_cast<const jshort*>(pcm.data()));

        // 24000 Hz mono PCM — the Chatterbox model output sample rate
        env->CallVoidMethod(callback, g_onAudioReady, jpcm, 24000);

        env->DeleteLocalRef(jpcm);

        LOGI("nativeSynthesize: success");
        return JNI_TRUE;

    } catch (const std::exception& e) {
        std::string msg = std::string("nativeSynthesize: exception: ") + e.what();
        reportError(env, callback, msg.c_str());
        return JNI_FALSE;
    }
}

/**
 * Request stop — NO MUTEX.
 * Must be callable from any thread while synthesize() is running.
 */
JNIEXPORT void JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativeStop(
        JNIEnv* /* env */, jobject /* thiz */) {

    LOGI("nativeStop: requesting stop");
    if (g_engine) {
        g_engine->requestStop();
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

} // extern "C"
