#include <jni.h>
#include <string>
#include "audio/wav_encoder.h"
#include "utils/logger.h"

// JNI package: com.mp.ai_supertonic_tts.SupertonicNativeLib
// Note: underscores in package name become _1 in JNI function names

extern "C" {

JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1supertonic_1tts_SupertonicNativeLib_nativeEncodeWav16(
        JNIEnv* env, jobject /* this */,
        jfloatArray jaudio, jint sampleRate, jint channels) {

    jfloat* audio = env->GetFloatArrayElements(jaudio, nullptr);
    jint len = env->GetArrayLength(jaudio);

    auto wav = audio::encode_wav_16(audio, len, sampleRate, channels);

    env->ReleaseFloatArrayElements(jaudio, audio, JNI_ABORT);

    jbyteArray result = env->NewByteArray(static_cast<jsize>(wav.size()));
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(wav.size()),
                            reinterpret_cast<const jbyte*>(wav.data()));
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1supertonic_1tts_SupertonicNativeLib_nativeEncodeWav32f(
        JNIEnv* env, jobject /* this */,
        jfloatArray jaudio, jint sampleRate, jint channels) {

    jfloat* audio = env->GetFloatArrayElements(jaudio, nullptr);
    jint len = env->GetArrayLength(jaudio);

    auto wav = audio::encode_wav_32f(audio, len, sampleRate, channels);

    env->ReleaseFloatArrayElements(jaudio, audio, JNI_ABORT);

    jbyteArray result = env->NewByteArray(static_cast<jsize>(wav.size()));
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(wav.size()),
                            reinterpret_cast<const jbyte*>(wav.data()));
    return result;
}

JNIEXPORT jbyteArray JNICALL
Java_com_mp_ai_1supertonic_1tts_SupertonicNativeLib_nativeEncodePcm16(
        JNIEnv* env, jobject /* this */,
        jfloatArray jaudio) {

    jfloat* audio = env->GetFloatArrayElements(jaudio, nullptr);
    jint len = env->GetArrayLength(jaudio);

    auto pcm = audio::encode_pcm_16(audio, len);

    env->ReleaseFloatArrayElements(jaudio, audio, JNI_ABORT);

    jbyteArray result = env->NewByteArray(static_cast<jsize>(pcm.size()));
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(pcm.size()),
                            reinterpret_cast<const jbyte*>(pcm.data()));
    return result;
}

JNIEXPORT void JNICALL
Java_com_mp_ai_1supertonic_1tts_SupertonicNativeLib_nativeClipAudio(
        JNIEnv* env, jobject /* this */,
        jfloatArray jaudio) {

    jfloat* audio = env->GetFloatArrayElements(jaudio, nullptr);
    jint len = env->GetArrayLength(jaudio);

    audio::clip_audio(audio, len);

    // Commit changes back (0 = copy back and free)
    env->ReleaseFloatArrayElements(jaudio, audio, 0);
}

} // extern "C"
