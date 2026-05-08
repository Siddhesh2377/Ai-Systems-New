// Copyright (c) 2025 Dark Matter Labs
#pragma once

#include <jni.h>

// Global refs to result classes used from JNI hot paths. Populated in
// JNI_OnLoad — looking up classes per-call would be allocator-heavy, and
// JNIEnv->FindClass() can fail off the main thread without a class loader.
struct JniCache {
  jclass    offline_result_cls;
  jmethodID offline_result_ctor;   // (Ljava/lang/String;[Ljava/lang/String;[F)V
  jclass    generated_audio_cls;
  jmethodID generated_audio_ctor;  // ([FI)V
  jclass    string_cls;
};

extern JniCache g_cache;

bool InitJniCache(JNIEnv* env);
void DestroyJniCache(JNIEnv* env);
