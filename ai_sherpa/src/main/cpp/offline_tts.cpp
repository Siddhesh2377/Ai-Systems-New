// Copyright (c) 2025 Dark Matter Labs
#include <algorithm>
#include <cstring>

#define TN_MODULE TN_MODULE_AI_SHERPA
#define TN_TAG    "ai_sherpa"
#include <tn_security/tn_security_macros.h>

#include "jni_cache.h"
#include "jni_common.h"
#include "sherpa-onnx/c-api/c-api.h"

namespace {

// See note on OfflineCfg — same RAII pattern.
struct TtsCfg {
  std::string vits_model, vits_lexicon, vits_tokens, vits_data_dir, vits_dict_dir;
  std::string kokoro_model, kokoro_voices, kokoro_tokens, kokoro_data_dir, kokoro_dict_dir;
  std::string provider, rule_fsts, rule_fars;
  SherpaOnnxOfflineTtsConfig cfg{};
};

void ReadVits(JNIEnv* env, jobject jmodel, jclass mc, TtsCfg& h) {
  jobject j = GetObjField(env, jmodel, mc, "vits",
                          "Lcom/dark/ai_sherpa/OfflineTtsVitsModelConfig;");
  if (!j) return;
  jclass c = env->GetObjectClass(j);
  h.vits_model    = GetStringField(env, j, c, "model");
  h.vits_lexicon  = GetStringField(env, j, c, "lexicon");
  h.vits_tokens   = GetStringField(env, j, c, "tokens");
  h.vits_data_dir = GetStringField(env, j, c, "dataDir");
  h.vits_dict_dir = GetStringField(env, j, c, "dictDir");
  h.cfg.model.vits.model         = h.vits_model.c_str();
  h.cfg.model.vits.lexicon       = h.vits_lexicon.c_str();
  h.cfg.model.vits.tokens        = h.vits_tokens.c_str();
  h.cfg.model.vits.data_dir      = h.vits_data_dir.c_str();
  h.cfg.model.vits.dict_dir      = h.vits_dict_dir.c_str();
  h.cfg.model.vits.noise_scale   = GetFloatField(env, j, c, "noiseScale");
  h.cfg.model.vits.noise_scale_w = GetFloatField(env, j, c, "noiseScaleW");
  h.cfg.model.vits.length_scale  = GetFloatField(env, j, c, "lengthScale");
  env->DeleteLocalRef(c);
  env->DeleteLocalRef(j);
}

void ReadKokoro(JNIEnv* env, jobject jmodel, jclass mc, TtsCfg& h) {
  jobject j = GetObjField(env, jmodel, mc, "kokoro",
                          "Lcom/dark/ai_sherpa/OfflineTtsKokoroModelConfig;");
  if (!j) return;
  jclass c = env->GetObjectClass(j);
  h.kokoro_model    = GetStringField(env, j, c, "model");
  h.kokoro_voices   = GetStringField(env, j, c, "voices");
  h.kokoro_tokens   = GetStringField(env, j, c, "tokens");
  h.kokoro_data_dir = GetStringField(env, j, c, "dataDir");
  h.kokoro_dict_dir = GetStringField(env, j, c, "dictDir");
  h.cfg.model.kokoro.model        = h.kokoro_model.c_str();
  h.cfg.model.kokoro.voices       = h.kokoro_voices.c_str();
  h.cfg.model.kokoro.tokens       = h.kokoro_tokens.c_str();
  h.cfg.model.kokoro.data_dir     = h.kokoro_data_dir.c_str();
  h.cfg.model.kokoro.dict_dir     = h.kokoro_dict_dir.c_str();
  h.cfg.model.kokoro.length_scale = GetFloatField(env, j, c, "lengthScale");
  env->DeleteLocalRef(c);
  env->DeleteLocalRef(j);
}

TtsCfg ReadTtsConfig(JNIEnv* env, jobject jconfig) {
  TtsCfg h;
  jclass cfg_cls = env->GetObjectClass(jconfig);

  if (jobject jmodel = GetObjField(env, jconfig, cfg_cls, "model",
                                   "Lcom/dark/ai_sherpa/OfflineTtsModelConfig;")) {
    jclass mc = env->GetObjectClass(jmodel);
    ReadVits(env, jmodel, mc, h);
    ReadKokoro(env, jmodel, mc, h);

    h.provider                 = GetStringField(env, jmodel, mc, "provider");
    h.cfg.model.num_threads    = std::max(1, (int)GetIntField(env, jmodel, mc, "numThreads"));
    h.cfg.model.debug          = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
    h.cfg.model.provider       = h.provider.c_str();

    env->DeleteLocalRef(mc);
    env->DeleteLocalRef(jmodel);
  }

  h.rule_fsts = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  h.rule_fars = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  h.cfg.rule_fsts        = h.rule_fsts.c_str();
  h.cfg.rule_fars        = h.rule_fars.c_str();
  h.cfg.max_num_sentences = std::max(1, (int)GetIntField(env, jconfig, cfg_cls, "maxNumSentences"));

  env->DeleteLocalRef(cfg_cls);
  return h;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_newFromFile(
    JNIEnv* env, jobject, jobject jconfig) {
  auto h = ReadTtsConfig(env, jconfig);

  tn_sec_set_op("OfflineTts.newFromFile");
  TN_D("op-detail: vits=%s kokoro=%s tokens=%s threads=%d",
       h.vits_model.empty()   ? "-" : h.vits_model.c_str(),
       h.kokoro_model.empty() ? "-" : h.kokoro_model.c_str(),
       h.vits_tokens.empty()  ? h.kokoro_tokens.c_str() : h.vits_tokens.c_str(),
       h.cfg.model.num_threads);

  const SherpaOnnxOfflineTts* p = SherpaOnnxCreateOfflineTts(&h.cfg);
  if (!p) {
    TN_ERR_FIX(TN_CODE_MODEL_LOAD_FAIL, TN_STAGE_LOAD,
        "Verify the TTS model files, tokens, and free memory.",
        "%s",
        "SherpaOnnxCreateOfflineTts returned null. "
        "Likely missing/incompatible model, tokens, or insufficient memory.");
    ThrowIllegalState(env, "Failed to create OfflineTts");
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_delete(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineTts(reinterpret_cast<const SherpaOnnxOfflineTts*>(ptr));
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_getSampleRate(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  return SherpaOnnxOfflineTtsSampleRate(
      reinterpret_cast<const SherpaOnnxOfflineTts*>(ptr));
}

JNIEXPORT jint JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_getNumSpeakers(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  return SherpaOnnxOfflineTtsNumSpeakers(
      reinterpret_cast<const SherpaOnnxOfflineTts*>(ptr));
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_OfflineTts_generate(
    JNIEnv* env, jobject, jlong ptr, jstring jtext, jint sid, jfloat speed) {
  CHECK_PTR(env, ptr, nullptr);
  if (!jtext) {
    ThrowIllegalState(env, "TTS text is null");
    return nullptr;
  }

  const char* text = env->GetStringUTFChars(jtext, nullptr);
  const size_t text_len = text ? std::strlen(text) : 0;

  tn_sec_set_op("OfflineTts.generate");
  TN_D("op-detail: sid=%d speed=%.2f text_len=%zu",
       (int)sid, (float)speed, text_len);

  SherpaOnnxGenerationConfig gen_cfg{};
  gen_cfg.sid   = static_cast<int>(sid);
  gen_cfg.speed = static_cast<float>(speed);

  const SherpaOnnxGeneratedAudio* audio = SherpaOnnxOfflineTtsGenerateWithConfig(
      reinterpret_cast<const SherpaOnnxOfflineTts*>(ptr),
      text, &gen_cfg, nullptr, nullptr);

  if (text) env->ReleaseStringUTFChars(jtext, text);

  if (!audio) {
    TN_ERR_FIX(TN_CODE_DECODE_FAIL, TN_STAGE_TTS_GENERATE,
        "Check that sid is in [0, numSpeakers) and text is non-empty.",
        "%s",
        "TTS generation returned null. "
        "Possibly invalid speaker id, empty text, or out of memory.");
    ThrowIllegalState(env, "TTS generation failed");
    return nullptr;
  }

  jfloatArray samples = env->NewFloatArray(audio->n);
  if (audio->n > 0 && audio->samples) {
    env->SetFloatArrayRegion(samples, 0, audio->n, audio->samples);
  }

  jobject result = env->NewObject(g_cache.generated_audio_cls,
                                  g_cache.generated_audio_ctor,
                                  samples, static_cast<jint>(audio->sample_rate));

  env->DeleteLocalRef(samples);
  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  return result;
}

}  // extern "C"
