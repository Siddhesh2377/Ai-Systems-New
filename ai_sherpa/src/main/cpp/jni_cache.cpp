// Copyright (c) 2025 Dark Matter Labs
#include "jni_cache.h"

#include "jni_common.h"

JniCache g_cache{};

static jclass CacheClass(JNIEnv* env, const char* name) {
  jclass local = env->FindClass(name);
  if (!local) {
    LOGE("JniCache: class not found: %s", name);
    return nullptr;
  }
  auto global = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);
  return global;
}

bool InitJniCache(JNIEnv* env) {
  g_cache.offline_result_cls = CacheClass(env, "com/dark/ai_sherpa/OfflineRecognizerResult");
  if (!g_cache.offline_result_cls) return false;
  g_cache.offline_result_ctor = env->GetMethodID(
      g_cache.offline_result_cls, "<init>",
      "(Ljava/lang/String;[Ljava/lang/String;[F)V");
  if (!g_cache.offline_result_ctor) {
    LOGE("JniCache: OfflineRecognizerResult.<init> not found");
    return false;
  }

  g_cache.generated_audio_cls = CacheClass(env, "com/dark/ai_sherpa/GeneratedAudio");
  if (!g_cache.generated_audio_cls) return false;
  g_cache.generated_audio_ctor = env->GetMethodID(
      g_cache.generated_audio_cls, "<init>", "([FI)V");
  if (!g_cache.generated_audio_ctor) {
    LOGE("JniCache: GeneratedAudio.<init> not found");
    return false;
  }

  g_cache.string_cls = CacheClass(env, "java/lang/String");
  return g_cache.string_cls != nullptr;
}

void DestroyJniCache(JNIEnv* env) {
  if (g_cache.offline_result_cls)  env->DeleteGlobalRef(g_cache.offline_result_cls);
  if (g_cache.generated_audio_cls) env->DeleteGlobalRef(g_cache.generated_audio_cls);
  if (g_cache.string_cls)          env->DeleteGlobalRef(g_cache.string_cls);
  g_cache = {};
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    LOGE("JNI_OnLoad: GetEnv failed");
    return -1;
  }
  if (!InitJniCache(env)) {
    LOGE("JNI_OnLoad: cache init failed");
    return -1;
  }
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNI_OnUnload(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
    DestroyJniCache(env);
  }
}
