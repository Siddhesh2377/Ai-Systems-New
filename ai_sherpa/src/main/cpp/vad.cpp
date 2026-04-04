// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

struct VadCfg {
  std::string silero_model, ten_model, provider;
  SherpaOnnxVadModelConfig cfg{};
};

static VadCfg ReadVadConfig(JNIEnv *env, jobject jconfig) {
  VadCfg h;

  jclass cfg_cls = env->GetObjectClass(jconfig);

  jobject jsilero = GetObjField(env, jconfig, cfg_cls, "sileroVadModelConfig",
                                 "Lcom/dark/ai_sherpa/SileroVadModelConfig;");
  if (jsilero) {
    jclass sc = env->GetObjectClass(jsilero);
    h.silero_model = GetStringField(env, jsilero, sc, "model");
    h.cfg.silero_vad.model = h.silero_model.c_str();
    h.cfg.silero_vad.threshold = GetFloatField(env, jsilero, sc, "threshold");
    h.cfg.silero_vad.min_silence_duration = GetFloatField(env, jsilero, sc, "minSilenceDuration");
    h.cfg.silero_vad.min_speech_duration = GetFloatField(env, jsilero, sc, "minSpeechDuration");
    h.cfg.silero_vad.max_speech_duration = GetFloatField(env, jsilero, sc, "maxSpeechDuration");
    h.cfg.silero_vad.window_size = GetIntField(env, jsilero, sc, "windowSize");
    env->DeleteLocalRef(sc);
    env->DeleteLocalRef(jsilero);
  }

  jobject jten = GetObjField(env, jconfig, cfg_cls, "tenVadModelConfig",
                              "Lcom/dark/ai_sherpa/TenVadModelConfig;");
  if (jten) {
    jclass tc = env->GetObjectClass(jten);
    h.ten_model = GetStringField(env, jten, tc, "model");
    h.cfg.ten_vad.model = h.ten_model.c_str();
    h.cfg.ten_vad.threshold = GetFloatField(env, jten, tc, "threshold");
    h.cfg.ten_vad.min_silence_duration = GetFloatField(env, jten, tc, "minSilenceDuration");
    h.cfg.ten_vad.min_speech_duration = GetFloatField(env, jten, tc, "minSpeechDuration");
    h.cfg.ten_vad.max_speech_duration = GetFloatField(env, jten, tc, "maxSpeechDuration");
    h.cfg.ten_vad.window_size = GetIntField(env, jten, tc, "windowSize");
    env->DeleteLocalRef(tc);
    env->DeleteLocalRef(jten);
  }

  h.provider = GetStringField(env, jconfig, cfg_cls, "provider");
  h.cfg.sample_rate = GetIntField(env, jconfig, cfg_cls, "sampleRate");
  h.cfg.num_threads = GetIntField(env, jconfig, cfg_cls, "numThreads");
  h.cfg.provider = h.provider.c_str();
  h.cfg.debug = GetBoolField(env, jconfig, cfg_cls, "debug") ? 1 : 0;

  env->DeleteLocalRef(cfg_cls);
  return h;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_Vad_newFromFile(
    JNIEnv *env, jobject, jobject jconfig, jint buffer_size_in_seconds) {
  auto h = ReadVadConfig(env, jconfig);
  SherpaOnnxVoiceActivityDetector *p =
      SherpaOnnxCreateVoiceActivityDetector(&h.cfg, static_cast<float>(buffer_size_in_seconds));
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
  auto h = ReadVadConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  SherpaOnnxVoiceActivityDetector *p =
      SherpaOnnxCreateVoiceActivityDetectorFromAsset(
          mgr, &h.cfg, static_cast<float>(buffer_size_in_seconds));
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
