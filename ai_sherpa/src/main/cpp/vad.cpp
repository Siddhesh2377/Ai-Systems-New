// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

static void ReadSileroVadConfig(JNIEnv *env, jobject jobj, jclass cls,
                                 SherpaOnnxSileroVadModelConfig *out) {
  static std::string model = GetStringField(env, jobj, cls, "model");
  out->model = model.c_str();
  out->threshold = GetFloatField(env, jobj, cls, "threshold");
  out->min_silence_duration = GetFloatField(env, jobj, cls, "minSilenceDuration");
  out->min_speech_duration = GetFloatField(env, jobj, cls, "minSpeechDuration");
  out->max_speech_duration = GetFloatField(env, jobj, cls, "maxSpeechDuration");
  out->window_size = GetIntField(env, jobj, cls, "windowSize");
}

static void ReadTenVadConfig(JNIEnv *env, jobject jobj, jclass cls,
                               SherpaOnnxTenVadModelConfig *out) {
  static std::string model = GetStringField(env, jobj, cls, "model");
  out->model = model.c_str();
  out->threshold = GetFloatField(env, jobj, cls, "threshold");
  out->min_silence_duration = GetFloatField(env, jobj, cls, "minSilenceDuration");
  out->min_speech_duration = GetFloatField(env, jobj, cls, "minSpeechDuration");
  out->max_speech_duration = GetFloatField(env, jobj, cls, "maxSpeechDuration");
  out->window_size = GetIntField(env, jobj, cls, "windowSize");
}

static SherpaOnnxVadModelConfig ReadVadConfig(JNIEnv *env, jobject jconfig) {
  SherpaOnnxVadModelConfig cfg{};

  jclass cfg_cls = env->GetObjectClass(jconfig);

  jobject jsilero = GetObjField(env, jconfig, cfg_cls, "sileroVadModelConfig",
                                 "Lcom/dark/ai_sherpa/SileroVadModelConfig;");
  if (jsilero) {
    jclass sc = env->GetObjectClass(jsilero);
    ReadSileroVadConfig(env, jsilero, sc, &cfg.silero_vad);
    env->DeleteLocalRef(sc);
    env->DeleteLocalRef(jsilero);
  }

  jobject jten = GetObjField(env, jconfig, cfg_cls, "tenVadModelConfig",
                              "Lcom/dark/ai_sherpa/TenVadModelConfig;");
  if (jten) {
    jclass tc = env->GetObjectClass(jten);
    ReadTenVadConfig(env, jten, tc, &cfg.ten_vad);
    env->DeleteLocalRef(tc);
    env->DeleteLocalRef(jten);
  }

  static std::string provider = GetStringField(env, jconfig, cfg_cls, "provider");
  cfg.sample_rate = GetIntField(env, jconfig, cfg_cls, "sampleRate");
  cfg.num_threads = GetIntField(env, jconfig, cfg_cls, "numThreads");
  cfg.provider = provider.c_str();
  cfg.debug = GetBoolField(env, jconfig, cfg_cls, "debug") ? 1 : 0;

  env->DeleteLocalRef(cfg_cls);
  return cfg;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_Vad_newFromFile(
    JNIEnv *env, jobject, jobject jconfig, jint buffer_size_in_seconds) {
  SherpaOnnxVadModelConfig cfg = ReadVadConfig(env, jconfig);
  SherpaOnnxVoiceActivityDetector *p =
      SherpaOnnxCreateVoiceActivityDetector(&cfg, static_cast<float>(buffer_size_in_seconds));
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create VoiceActivityDetector");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_Vad_newFromAsset(
    JNIEnv *env, jobject, jobject asset_manager, jobject jconfig,
    jint buffer_size_in_seconds) {
  SherpaOnnxVadModelConfig cfg = ReadVadConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  SherpaOnnxVoiceActivityDetector *p =
      SherpaOnnxCreateVoiceActivityDetectorFromAsset(
          mgr, &cfg, static_cast<float>(buffer_size_in_seconds));
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create VoiceActivityDetector from asset");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyVoiceActivityDetector(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_acceptWaveform(
    JNIEnv *env, jobject, jlong ptr, jfloatArray samples) {
  CHECK_PTR(env, ptr, );
  jsize len = env->GetArrayLength(samples);
  jfloat *data = reinterpret_cast<jfloat *>(
      env->GetPrimitiveArrayCritical(samples, nullptr));
  if (!data) return;
  SherpaOnnxVoiceActivityDetectorAcceptWaveform(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr), data,
      static_cast<int>(len));
  env->ReleasePrimitiveArrayCritical(samples, data, JNI_ABORT);
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sherpa_Vad_empty(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, JNI_TRUE);
  return SherpaOnnxVoiceActivityDetectorEmpty(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr))
      ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_pop(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxVoiceActivityDetectorPop(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_clear(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxVoiceActivityDetectorClear(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_Vad_front(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, nullptr);

  const SherpaOnnxSpeechSegment *seg = SherpaOnnxVoiceActivityDetectorFront(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
  if (!seg) return nullptr;

  jfloatArray samples = env->NewFloatArray(seg->n);
  if (seg->n > 0 && seg->samples) {
    env->SetFloatArrayRegion(samples, 0, seg->n, seg->samples);
  }

  jobject result = env->NewObject(
      g_cache.speech_segment_cls, g_cache.speech_segment_ctor,
      static_cast<jint>(seg->start), samples);

  env->DeleteLocalRef(samples);
  SherpaOnnxDestroySpeechSegment(seg);
  return result;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sherpa_Vad_isSpeechDetected(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, JNI_FALSE);
  return SherpaOnnxVoiceActivityDetectorDetected(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr))
      ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_reset(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxVoiceActivityDetectorReset(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_Vad_flush(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxVoiceActivityDetectorFlush(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr));
}

JNIEXPORT jfloat JNICALL
Java_com_dark_ai_1sherpa_Vad_compute(
    JNIEnv *env, jobject, jlong ptr, jfloatArray samples) {
  CHECK_PTR(env, ptr, 0.f);
  jsize len = env->GetArrayLength(samples);
  jfloat *data = reinterpret_cast<jfloat *>(
      env->GetPrimitiveArrayCritical(samples, nullptr));
  if (!data) return 0.f;
  float prob = SherpaOnnxVadModelComputeProb(
      reinterpret_cast<SherpaOnnxVoiceActivityDetector *>(ptr), data,
      static_cast<int>(len));
  env->ReleasePrimitiveArrayCritical(samples, data, JNI_ABORT);
  return static_cast<jfloat>(prob);
}

} // extern "C"
