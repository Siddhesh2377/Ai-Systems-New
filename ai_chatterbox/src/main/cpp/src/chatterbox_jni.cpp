#include <jni.h>
#include <android/log.h>

#define LOG_TAG "ChatterboxTTS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1chatterbox_ChatterboxNativeLib_nativePing(JNIEnv* env, jobject thiz) {
    LOGI("Chatterbox TTS native library loaded");
    return JNI_TRUE;
}

}
