// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

static SherpaOnnxOfflineTtsConfig ReadTtsConfig(JNIEnv *env, jobject jconfig) {
  SherpaOnnxOfflineTtsConfig cfg{};

  jclass cfg_cls = env->GetObjectClass(jconfig);

  jobject jmodel = GetObjField(env, jconfig, cfg_cls, "model",
                                "Lcom/dark/ai_sherpa/OfflineTtsModelConfig;");
  if (jmodel) {
    jclass mc = env->GetObjectClass(jmodel);

    jobject jvits = GetObjField(env, jmodel, mc, "vits",
                                 "Lcom/dark/ai_sherpa/OfflineTtsVitsModelConfig;");
    if (jvits) {
      jclass vc = env->GetObjectClass(jvits);
      static std::string model = GetStringField(env, jvits, vc, "model");
      static std::string lexicon = GetStringField(env, jvits, vc, "lexicon");
      static std::string tokens = GetStringField(env, jvits, vc, "tokens");
      static std::string data_dir = GetStringField(env, jvits, vc, "dataDir");
      static std::string dict_dir = GetStringField(env, jvits, vc, "dictDir");
      cfg.model.vits.model = model.c_str();
      cfg.model.vits.lexicon = lexicon.c_str();
      cfg.model.vits.tokens = tokens.c_str();
      cfg.model.vits.data_dir = data_dir.c_str();
      cfg.model.vits.dict_dir = dict_dir.c_str();
      cfg.model.vits.noise_scale = GetFloatField(env, jvits, vc, "noiseScale");
      cfg.model.vits.noise_scale_w = GetFloatField(env, jvits, vc, "noiseScaleW");
      cfg.model.vits.length_scale = GetFloatField(env, jvits, vc, "lengthScale");
      env->DeleteLocalRef(vc);
      env->DeleteLocalRef(jvits);
    }

    jobject jkokoro = GetObjField(env, jmodel, mc, "kokoro",
                                   "Lcom/dark/ai_sherpa/OfflineTtsKokoroModelConfig;");
    if (jkokoro) {
      jclass kc = env->GetObjectClass(jkokoro);
      static std::string model = GetStringField(env, jkokoro, kc, "model");
      static std::string voices = GetStringField(env, jkokoro, kc, "voices");
      static std::string tokens = GetStringField(env, jkokoro, kc, "tokens");
      static std::string data_dir = GetStringField(env, jkokoro, kc, "dataDir");
      static std::string dict_dir = GetStringField(env, jkokoro, kc, "dictDir");
      cfg.model.kokoro.model = model.c_str();
      cfg.model.kokoro.voices = voices.c_str();
      cfg.model.kokoro.tokens = tokens.c_str();
      cfg.model.kokoro.data_dir = data_dir.c_str();
      cfg.model.kokoro.dict_dir = dict_dir.c_str();
      cfg.model.kokoro.length_scale = GetFloatField(env, jkokoro, kc, "lengthScale");
      env->DeleteLocalRef(kc);
      env->DeleteLocalRef(jkokoro);
    }

    static std::string provider = GetStringField(env, jmodel, mc, "provider");
    cfg.model.num_threads = GetIntField(env, jmodel, mc, "numThreads");
    cfg.model.debug = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
    cfg.model.provider = provider.c_str();

    env->DeleteLocalRef(mc);
    env->DeleteLocalRef(jmodel);
  }

  static std::string rule_fsts = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  static std::string rule_fars = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  cfg.rule_fsts = rule_fsts.c_str();
  cfg.rule_fars = rule_fars.c_str();
  cfg.max_num_sentences = GetIntField(env, jconfig, cfg_cls, "maxNumSentences");

  env->DeleteLocalRef(cfg_cls);
  return cfg;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  SherpaOnnxOfflineTtsConfig cfg = ReadTtsConfig(env, jconfig);
  const SherpaOnnxOfflineTts *p = SherpaOnnxCreateOfflineTts(&cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineTts");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_newFromAsset(
    JNIEnv *env, jobject, jobject asset_manager, jobject jconfig) {
  SherpaOnnxOfflineTtsConfig cfg = ReadTtsConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const SherpaOnnxOfflineTts *p = SherpaOnnxCreateOfflineTtsFromAsset(mgr, &cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineTts from asset");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineTts(
      reinterpret_cast<const SherpaOnnxOfflineTts *>(ptr));
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_getSampleRate(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  return SherpaOnnxOfflineTtsSampleRate(
      reinterpret_cast<const SherpaOnnxOfflineTts *>(ptr));
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_getNumSpeakers(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  return SherpaOnnxOfflineTtsNumSpeakers(
      reinterpret_cast<const SherpaOnnxOfflineTts *>(ptr));
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_generate(
    JNIEnv *env, jobject, jlong ptr, jstring jtext, jint sid, jfloat speed) {
  CHECK_PTR(env, ptr, nullptr);

  const char *text = env->GetStringUTFChars(jtext, nullptr);
  const SherpaOnnxGeneratedAudio *audio = SherpaOnnxOfflineTtsGenerate(
      reinterpret_cast<const SherpaOnnxOfflineTts *>(ptr),
      text, static_cast<int>(sid), static_cast<float>(speed));
  env->ReleaseStringUTFChars(jtext, text);

  if (!audio) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "TTS generation failed");
    env->DeleteLocalRef(ex);
    return nullptr;
  }

  jfloatArray samples = env->NewFloatArray(audio->n);
  if (audio->n > 0 && audio->samples) {
    env->SetFloatArrayRegion(samples, 0, audio->n, audio->samples);
  }

  jobject result = env->NewObject(
      g_cache.generated_audio_cls, g_cache.generated_audio_ctor,
      samples, static_cast<jint>(audio->sample_rate));

  env->DeleteLocalRef(samples);
  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  return result;
}

} // extern "C"
