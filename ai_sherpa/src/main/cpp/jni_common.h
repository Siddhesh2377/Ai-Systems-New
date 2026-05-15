// Copyright (c) 2025 Dark Matter Labs
#pragma once

#include <jni.h>

#include <string>

#ifndef TN_MODULE
#define TN_MODULE TN_MODULE_AI_SHERPA
#endif
#ifndef TN_TAG
#define TN_TAG "ai_sherpa"
#endif
#include <tn_security/tn_security_macros.h>

#define LOGE(...) TN_E(__VA_ARGS__)
#define LOGW(...) TN_W(__VA_ARGS__)
#define LOGD(...) TN_D(__VA_ARGS__)
#define LOGI(...) TN_I(__VA_ARGS__)

// Throws java.lang.NullPointerException and returns retval if ptr is 0.
// Used to guard JNI entries that take Java-side handles (jlong) — a 0 value
// after Kotlin close() means use-after-close from the consumer's side.
#define CHECK_PTR(env, ptr, retval) \
  do { \
    if (!(ptr)) { \
      jclass _ex = (env)->FindClass("java/lang/NullPointerException"); \
      (env)->ThrowNew(_ex, "Native pointer is null (used after close?)"); \
      (env)->DeleteLocalRef(_ex); \
      return retval; \
    } \
  } while (0)

inline std::string GetStringField(JNIEnv* env, jobject obj, jclass cls, const char* name) {
  jfieldID fid = env->GetFieldID(cls, name, "Ljava/lang/String;");
  if (!fid) return {};
  auto s = (jstring)env->GetObjectField(obj, fid);
  if (!s) return {};
  const char* p = env->GetStringUTFChars(s, nullptr);
  std::string out(p ? p : "");
  if (p) env->ReleaseStringUTFChars(s, p);
  env->DeleteLocalRef(s);
  return out;
}

inline jint GetIntField(JNIEnv* env, jobject obj, jclass cls, const char* name) {
  jfieldID fid = env->GetFieldID(cls, name, "I");
  return fid ? env->GetIntField(obj, fid) : 0;
}

inline jfloat GetFloatField(JNIEnv* env, jobject obj, jclass cls, const char* name) {
  jfieldID fid = env->GetFieldID(cls, name, "F");
  return fid ? env->GetFloatField(obj, fid) : 0.f;
}

inline bool GetBoolField(JNIEnv* env, jobject obj, jclass cls, const char* name) {
  jfieldID fid = env->GetFieldID(cls, name, "Z");
  return fid && env->GetBooleanField(obj, fid);
}

inline jobject GetObjField(JNIEnv* env, jobject obj, jclass cls, const char* name, const char* sig) {
  jfieldID fid = env->GetFieldID(cls, name, sig);
  return fid ? env->GetObjectField(obj, fid) : nullptr;
}

inline void ThrowIllegalState(JNIEnv* env, const char* msg) {
  jclass ex = env->FindClass("java/lang/IllegalStateException");
  env->ThrowNew(ex, msg);
  env->DeleteLocalRef(ex);
}
