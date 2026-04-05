// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"

struct TtsCfg {
  std::string vits_model, vits_lexicon, vits_tokens, vits_data_dir, vits_dict_dir;
  std::string kokoro_model, kokoro_voices, kokoro_tokens, kokoro_data_dir, kokoro_dict_dir;
  std::string provider, rule_fsts, rule_fars;
  SherpaOnnxOfflineTtsConfig cfg{};
};

static TtsCfg ReadTtsConfig(JNIEnv *env, jobject jconfig) {
  TtsCfg h;

  jclass cfg_cls = env->GetObjectClass(jconfig);

  jobject jmodel = GetObjField(env, jconfig, cfg_cls, "model",
                                "Lcom/dark/ai_sherpa/OfflineTtsModelConfig;");
  if (jmodel) {
    jclass mc = env->GetObjectClass(jmodel);

    jobject jvits = GetObjField(env, jmodel, mc, "vits",
                                 "Lcom/dark/ai_sherpa/OfflineTtsVitsModelConfig;");
    if (jvits) {
      jclass vc = env->GetObjectClass(jvits);
      h.vits_model = GetStringField(env, jvits, vc, "model");
      h.vits_lexicon = GetStringField(env, jvits, vc, "lexicon");
      h.vits_tokens = GetStringField(env, jvits, vc, "tokens");
      h.vits_data_dir = GetStringField(env, jvits, vc, "dataDir");
      h.vits_dict_dir = GetStringField(env, jvits, vc, "dictDir");
      h.cfg.model.vits.model = h.vits_model.c_str();
      h.cfg.model.vits.lexicon = h.vits_lexicon.c_str();
      h.cfg.model.vits.tokens = h.vits_tokens.c_str();
      h.cfg.model.vits.data_dir = h.vits_data_dir.c_str();
      h.cfg.model.vits.dict_dir = h.vits_dict_dir.c_str();
      h.cfg.model.vits.noise_scale = GetFloatField(env, jvits, vc, "noiseScale");
      h.cfg.model.vits.noise_scale_w = GetFloatField(env, jvits, vc, "noiseScaleW");
      h.cfg.model.vits.length_scale = GetFloatField(env, jvits, vc, "lengthScale");
      env->DeleteLocalRef(vc);
      env->DeleteLocalRef(jvits);
    }

    jobject jkokoro = GetObjField(env, jmodel, mc, "kokoro",
                                   "Lcom/dark/ai_sherpa/OfflineTtsKokoroModelConfig;");
    if (jkokoro) {
      jclass kc = env->GetObjectClass(jkokoro);
      h.kokoro_model = GetStringField(env, jkokoro, kc, "model");
      h.kokoro_voices = GetStringField(env, jkokoro, kc, "voices");
      h.kokoro_tokens = GetStringField(env, jkokoro, kc, "tokens");
      h.kokoro_data_dir = GetStringField(env, jkokoro, kc, "dataDir");
      h.kokoro_dict_dir = GetStringField(env, jkokoro, kc, "dictDir");
      h.cfg.model.kokoro.model = h.kokoro_model.c_str();
      h.cfg.model.kokoro.voices = h.kokoro_voices.c_str();
      h.cfg.model.kokoro.tokens = h.kokoro_tokens.c_str();
      h.cfg.model.kokoro.data_dir = h.kokoro_data_dir.c_str();
      h.cfg.model.kokoro.dict_dir = h.kokoro_dict_dir.c_str();
      h.cfg.model.kokoro.length_scale = GetFloatField(env, jkokoro, kc, "lengthScale");
      env->DeleteLocalRef(kc);
      env->DeleteLocalRef(jkokoro);
    }

    h.provider = GetStringField(env, jmodel, mc, "provider");
    h.cfg.model.num_threads = GetIntField(env, jmodel, mc, "numThreads");
    h.cfg.model.debug = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
    h.cfg.model.provider = h.provider.c_str();

    env->DeleteLocalRef(mc);
    env->DeleteLocalRef(jmodel);
  }

  h.rule_fsts = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  h.rule_fars = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  h.cfg.rule_fsts = h.rule_fsts.c_str();
  h.cfg.rule_fars = h.rule_fars.c_str();
  h.cfg.max_num_sentences = GetIntField(env, jconfig, cfg_cls, "maxNumSentences");

  env->DeleteLocalRef(cfg_cls);
  return h;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  auto h = ReadTtsConfig(env, jconfig);
  const SherpaOnnxOfflineTts *p = SherpaOnnxCreateOfflineTts(&h.cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineTts");
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
  SherpaOnnxGenerationConfig gen_cfg{};
  gen_cfg.sid = static_cast<int>(sid);
  gen_cfg.speed = static_cast<float>(speed);
  const SherpaOnnxGeneratedAudio *audio = SherpaOnnxOfflineTtsGenerateWithConfig(
      reinterpret_cast<const SherpaOnnxOfflineTts *>(ptr),
      text, &gen_cfg, nullptr, nullptr);
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
