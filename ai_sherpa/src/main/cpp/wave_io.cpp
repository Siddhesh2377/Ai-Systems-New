// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

extern "C" {

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_WaveReader_readWaveFromFile(
    JNIEnv *env, jobject, jstring jfilename) {
  const char *filename = env->GetStringUTFChars(jfilename, nullptr);
  const SherpaOnnxWave *wave = SherpaOnnxReadWave(filename);
  env->ReleaseStringUTFChars(jfilename, filename);

  if (!wave) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to read wave file");
    env->DeleteLocalRef(ex);
    return nullptr;
  }

  jfloatArray samples = env->NewFloatArray(wave->num_samples);
  if (wave->num_samples > 0 && wave->samples) {
    env->SetFloatArrayRegion(samples, 0, wave->num_samples, wave->samples);
  }

  jobject result = env->NewObject(
      g_cache.wave_data_cls, g_cache.wave_data_ctor,
      samples, static_cast<jint>(wave->sample_rate));

  env->DeleteLocalRef(samples);
  SherpaOnnxFreeWave(wave);
  return result;
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_WaveReader_readWaveFromAsset(
    JNIEnv *env, jobject, jobject asset_manager, jstring jfilename) {
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const char *filename = env->GetStringUTFChars(jfilename, nullptr);
  AAsset *asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
  env->ReleaseStringUTFChars(jfilename, filename);

  if (!asset) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to open wave asset");
    env->DeleteLocalRef(ex);
    return nullptr;
  }

  off_t size = AAsset_getLength(asset);
  const char *data = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
  const SherpaOnnxWave *wave = SherpaOnnxReadWaveFromBinaryData(data, static_cast<int32_t>(size));
  AAsset_close(asset);

  if (!wave) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to read wave from asset");
    env->DeleteLocalRef(ex);
    return nullptr;
  }

  jfloatArray samples = env->NewFloatArray(wave->num_samples);
  if (wave->num_samples > 0 && wave->samples) {
    env->SetFloatArrayRegion(samples, 0, wave->num_samples, wave->samples);
  }

  jobject result = env->NewObject(
      g_cache.wave_data_cls, g_cache.wave_data_ctor,
      samples, static_cast<jint>(wave->sample_rate));

  env->DeleteLocalRef(samples);
  SherpaOnnxFreeWave(wave);
  return result;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sherpa_WaveReader_writeWaveToFile(
    JNIEnv *env, jobject, jstring jfilename, jfloatArray jsamples,
    jint sample_rate) {
  const char *filename = env->GetStringUTFChars(jfilename, nullptr);
  jsize len = env->GetArrayLength(jsamples);
  jfloat *data = reinterpret_cast<jfloat *>(
      env->GetPrimitiveArrayCritical(jsamples, nullptr));
  if (!data) {
    env->ReleaseStringUTFChars(jfilename, filename);
    return JNI_FALSE;
  }
  int32_t ok = SherpaOnnxWriteWave(data, static_cast<int32_t>(len),
                                    static_cast<int32_t>(sample_rate), filename);
  env->ReleasePrimitiveArrayCritical(jsamples, data, JNI_ABORT);
  env->ReleaseStringUTFChars(jfilename, filename);
  return ok ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"
