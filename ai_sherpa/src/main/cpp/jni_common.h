// Copyright (c) 2025 Dark Matter Labs
#pragma once
#include <android/log.h>
#include <jni.h>
#include <string>

#define TAG "ai_sherpa"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

#define CHECK_PTR(env, ptr, retval) \
  if (!(ptr)) { \
    jclass ex = (env)->FindClass("java/lang/NullPointerException"); \
    (env)->ThrowNew(ex, "Native pointer is null"); \
    (env)->DeleteLocalRef(ex); \
    return (retval); \
  }

inline std::string GetStringField(JNIEnv *env, jobject obj, jclass cls, const char *name) {
  jfieldID fid = env->GetFieldID(cls, name, "Ljava/lang/String;");
  if (!fid) return {};
  jstring s = (jstring)env->GetObjectField(obj, fid);
  if (!s) return {};
  const char *p = env->GetStringUTFChars(s, nullptr);
  std::string result(p);
  env->ReleaseStringUTFChars(s, p);
  env->DeleteLocalRef(s);
  return result;
}

inline int GetIntField(JNIEnv *env, jobject obj, jclass cls, const char *name) {
  jfieldID fid = env->GetFieldID(cls, name, "I");
  if (!fid) return 0;
  return env->GetIntField(obj, fid);
}

inline float GetFloatField(JNIEnv *env, jobject obj, jclass cls, const char *name) {
  jfieldID fid = env->GetFieldID(cls, name, "F");
  if (!fid) return 0.f;
  return env->GetFloatField(obj, fid);
}

inline bool GetBoolField(JNIEnv *env, jobject obj, jclass cls, const char *name) {
  jfieldID fid = env->GetFieldID(cls, name, "Z");
  if (!fid) return false;
  return env->GetBooleanField(obj, fid);
}

inline jobject GetObjField(JNIEnv *env, jobject obj, jclass cls, const char *name, const char *sig) {
  jfieldID fid = env->GetFieldID(cls, name, sig);
  if (!fid) return nullptr;
  return env->GetObjectField(obj, fid);
}
