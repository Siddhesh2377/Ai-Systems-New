// Copyright (c) 2025 Dark Matter Labs
#pragma once
#include <jni.h>

struct JniCache {
  jclass online_result_cls;
  jmethodID online_result_ctor;
  jclass offline_result_cls;
  jmethodID offline_result_ctor;
  jclass wave_data_cls;
  jmethodID wave_data_ctor;
  jclass speech_segment_cls;
  jmethodID speech_segment_ctor;
  jclass generated_audio_cls;
  jmethodID generated_audio_ctor;
  jclass string_cls;
};

extern JniCache g_cache;

bool InitJniCache(JNIEnv *env);
void DestroyJniCache(JNIEnv *env);
