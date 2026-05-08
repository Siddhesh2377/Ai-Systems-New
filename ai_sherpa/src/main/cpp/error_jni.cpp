// Copyright (c) 2025 Dark Matter Labs
#include <jni.h>

#include "error_tracker.h"

extern "C" {

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_SherpaLib_nativeErrorInit(JNIEnv*, jobject) {
  tn_error_init();
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_SherpaLib_nativeErrorSetCrashLogPath(
    JNIEnv* env, jobject, jstring jpath) {
  if (!jpath) return;
  const char* p = env->GetStringUTFChars(jpath, nullptr);
  tn_error_set_crash_log_path(p);
  env->ReleaseStringUTFChars(jpath, p);
}

JNIEXPORT jstring JNICALL
Java_com_dark_ai_1sherpa_SherpaLib_nativeErrorGetLastJson(JNIEnv* env, jobject) {
  const char* j = tn_error_get_last_json();
  return env->NewStringUTF(j ? j : "{}");
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_SherpaLib_nativeErrorClear(JNIEnv*, jobject) {
  tn_error_clear_last();
  tn_error_clear_op();
}

}  // extern "C"
