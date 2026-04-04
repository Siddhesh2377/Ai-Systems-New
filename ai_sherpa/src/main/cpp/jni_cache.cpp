// Copyright (c) 2025 Dark Matter Labs
#include "jni_cache.h"
#include "jni_common.h"

JniCache g_cache{};

static jclass CacheClass(JNIEnv *env, const char *name) {
  jclass local = env->FindClass(name);
  if (!local) {
    LOGE("JniCache: class not found: %s", name);
    return nullptr;
  }
  jclass global = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);
  return global;
}

bool InitJniCache(JNIEnv *env) {
  g_cache.online_result_cls = CacheClass(env, "com/dark/ai_sherpa/OnlineRecognizerResult");
  if (!g_cache.online_result_cls) return false;
  g_cache.online_result_ctor = env->GetMethodID(
      g_cache.online_result_cls, "<init>",
      "(Ljava/lang/String;[Ljava/lang/String;[F[F)V");
  if (!g_cache.online_result_ctor) {
    LOGE("JniCache: OnlineRecognizerResult ctor not found");
    return false;
  }

  g_cache.offline_result_cls = CacheClass(env, "com/dark/ai_sherpa/OfflineRecognizerResult");
  if (!g_cache.offline_result_cls) return false;
  g_cache.offline_result_ctor = env->GetMethodID(
      g_cache.offline_result_cls, "<init>",
      "(Ljava/lang/String;[Ljava/lang/String;[F)V");
  if (!g_cache.offline_result_ctor) {
    LOGE("JniCache: OfflineRecognizerResult ctor not found");
    return false;
  }

  g_cache.wave_data_cls = CacheClass(env, "com/dark/ai_sherpa/WaveData");
  if (!g_cache.wave_data_cls) return false;
  g_cache.wave_data_ctor = env->GetMethodID(
      g_cache.wave_data_cls, "<init>", "([FI)V");
  if (!g_cache.wave_data_ctor) {
    LOGE("JniCache: WaveData ctor not found");
    return false;
  }

  g_cache.speech_segment_cls = CacheClass(env, "com/dark/ai_sherpa/SpeechSegment");
  if (!g_cache.speech_segment_cls) return false;
  g_cache.speech_segment_ctor = env->GetMethodID(
      g_cache.speech_segment_cls, "<init>", "(I[F)V");
  if (!g_cache.speech_segment_ctor) {
    LOGE("JniCache: SpeechSegment ctor not found");
    return false;
  }

  g_cache.generated_audio_cls = CacheClass(env, "com/dark/ai_sherpa/GeneratedAudio");
  if (!g_cache.generated_audio_cls) return false;
  g_cache.generated_audio_ctor = env->GetMethodID(
      g_cache.generated_audio_cls, "<init>", "([FI)V");
  if (!g_cache.generated_audio_ctor) {
    LOGE("JniCache: GeneratedAudio ctor not found");
    return false;
  }

  g_cache.string_cls = CacheClass(env, "java/lang/String");
  if (!g_cache.string_cls) return false;

  return true;
}

void DestroyJniCache(JNIEnv *env) {
  if (g_cache.online_result_cls) env->DeleteGlobalRef(g_cache.online_result_cls);
  if (g_cache.offline_result_cls) env->DeleteGlobalRef(g_cache.offline_result_cls);
  if (g_cache.wave_data_cls) env->DeleteGlobalRef(g_cache.wave_data_cls);
  if (g_cache.speech_segment_cls) env->DeleteGlobalRef(g_cache.speech_segment_cls);
  if (g_cache.generated_audio_cls) env->DeleteGlobalRef(g_cache.generated_audio_cls);
  if (g_cache.string_cls) env->DeleteGlobalRef(g_cache.string_cls);
  g_cache = {};
}

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    LOGE("JNI_OnLoad: GetEnv failed");
    return -1;
  }
  if (!InitJniCache(env)) {
    LOGE("JNI_OnLoad: cache init failed");
    return -1;
  }
  return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *) {
  JNIEnv *env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) {
    DestroyJniCache(env);
  }
}
