// Copyright (c) 2025 Dark Matter Labs
#include <algorithm>

#define TN_MODULE TN_MODULE_AI_SHERPA
#define TN_TAG    "ai_sherpa"
#include <tn_security/tn_security_macros.h>

#include "jni_cache.h"
#include "jni_common.h"
#include "sherpa-onnx/c-api/c-api.h"

namespace {

// Holds owned std::strings whose .c_str() pointers populate the C struct.
// The C struct only stores const char* — strings must outlive the
// SherpaOnnxCreateOfflineRecognizer call, hence this RAII bundle.
struct OfflineCfg {
  std::string decoding_method, hotwords_file, rule_fsts, rule_fars;
  std::string tokens, provider, model_type, modeling_unit, bpe_vocab;
  std::string trans_enc, trans_dec, trans_joi;
  std::string paraformer_model, nemo_ctc_model, tdnn_model;
  std::string whisper_enc, whisper_dec, whisper_lang, whisper_task;
  std::string lm_model;
  std::string hr_lexicon, hr_rule_fsts;
  SherpaOnnxOfflineRecognizerConfig cfg{};
};

void ReadFeatConfig(JNIEnv* env, jobject jcfg, jclass cfg_cls, OfflineCfg& h) {
  jobject jfeat = GetObjField(env, jcfg, cfg_cls, "featConfig",
                              "Lcom/dark/ai_sherpa/FeatureConfig;");
  if (!jfeat) return;
  jclass fc = env->GetObjectClass(jfeat);
  h.cfg.feat_config.sample_rate = GetIntField(env, jfeat, fc, "sampleRate");
  h.cfg.feat_config.feature_dim = GetIntField(env, jfeat, fc, "featureDim");
  env->DeleteLocalRef(fc);
  env->DeleteLocalRef(jfeat);
}

void ReadModelConfig(JNIEnv* env, jobject jcfg, jclass cfg_cls, OfflineCfg& h) {
  jobject jmodel = GetObjField(env, jcfg, cfg_cls, "modelConfig",
                               "Lcom/dark/ai_sherpa/OfflineModelConfig;");
  if (!jmodel) return;
  jclass mc = env->GetObjectClass(jmodel);

  if (jobject j = GetObjField(env, jmodel, mc, "transducer",
                              "Lcom/dark/ai_sherpa/OfflineTransducerModelConfig;")) {
    jclass c = env->GetObjectClass(j);
    h.trans_enc = GetStringField(env, j, c, "encoder");
    h.trans_dec = GetStringField(env, j, c, "decoder");
    h.trans_joi = GetStringField(env, j, c, "joiner");
    h.cfg.model_config.transducer.encoder = h.trans_enc.c_str();
    h.cfg.model_config.transducer.decoder = h.trans_dec.c_str();
    h.cfg.model_config.transducer.joiner  = h.trans_joi.c_str();
    env->DeleteLocalRef(c);
    env->DeleteLocalRef(j);
  }

  if (jobject j = GetObjField(env, jmodel, mc, "paraformer",
                              "Lcom/dark/ai_sherpa/OfflineParaformerModelConfig;")) {
    jclass c = env->GetObjectClass(j);
    h.paraformer_model = GetStringField(env, j, c, "model");
    h.cfg.model_config.paraformer.model = h.paraformer_model.c_str();
    env->DeleteLocalRef(c);
    env->DeleteLocalRef(j);
  }

  if (jobject j = GetObjField(env, jmodel, mc, "nemoCtc",
                              "Lcom/dark/ai_sherpa/OfflineNemoEncDecCtcModelConfig;")) {
    jclass c = env->GetObjectClass(j);
    h.nemo_ctc_model = GetStringField(env, j, c, "model");
    h.cfg.model_config.nemo_ctc.model = h.nemo_ctc_model.c_str();
    env->DeleteLocalRef(c);
    env->DeleteLocalRef(j);
  }

  if (jobject j = GetObjField(env, jmodel, mc, "whisper",
                              "Lcom/dark/ai_sherpa/OfflineWhisperModelConfig;")) {
    jclass c = env->GetObjectClass(j);
    h.whisper_enc  = GetStringField(env, j, c, "encoder");
    h.whisper_dec  = GetStringField(env, j, c, "decoder");
    h.whisper_lang = GetStringField(env, j, c, "language");
    h.whisper_task = GetStringField(env, j, c, "task");
    h.cfg.model_config.whisper.encoder      = h.whisper_enc.c_str();
    h.cfg.model_config.whisper.decoder      = h.whisper_dec.c_str();
    h.cfg.model_config.whisper.language     = h.whisper_lang.c_str();
    h.cfg.model_config.whisper.task         = h.whisper_task.c_str();
    h.cfg.model_config.whisper.tail_paddings = GetIntField(env, j, c, "tailPaddings");
    env->DeleteLocalRef(c);
    env->DeleteLocalRef(j);
  }

  if (jobject j = GetObjField(env, jmodel, mc, "tdnn",
                              "Lcom/dark/ai_sherpa/OfflineTdnnModelConfig;")) {
    jclass c = env->GetObjectClass(j);
    h.tdnn_model = GetStringField(env, j, c, "model");
    h.cfg.model_config.tdnn.model = h.tdnn_model.c_str();
    env->DeleteLocalRef(c);
    env->DeleteLocalRef(j);
  }

  h.tokens        = GetStringField(env, jmodel, mc, "tokens");
  h.provider      = GetStringField(env, jmodel, mc, "provider");
  h.model_type    = GetStringField(env, jmodel, mc, "modelType");
  h.modeling_unit = GetStringField(env, jmodel, mc, "modelingUnit");
  h.bpe_vocab     = GetStringField(env, jmodel, mc, "bpeVocab");

  // Defensive clamp — Kotlin already requires >= 1, but a malicious
  // proguard-stripped consumer could still pass 0/negative.
  int n_threads = GetIntField(env, jmodel, mc, "numThreads");
  h.cfg.model_config.num_threads   = std::max(1, n_threads);
  h.cfg.model_config.tokens        = h.tokens.c_str();
  h.cfg.model_config.debug         = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
  h.cfg.model_config.provider      = h.provider.c_str();
  h.cfg.model_config.model_type    = h.model_type.c_str();
  h.cfg.model_config.modeling_unit = h.modeling_unit.c_str();
  h.cfg.model_config.bpe_vocab     = h.bpe_vocab.c_str();

  env->DeleteLocalRef(mc);
  env->DeleteLocalRef(jmodel);
}

void ReadLmConfig(JNIEnv* env, jobject jcfg, jclass cfg_cls, OfflineCfg& h) {
  jobject j = GetObjField(env, jcfg, cfg_cls, "lmConfig",
                          "Lcom/dark/ai_sherpa/OfflineLMConfig;");
  if (!j) return;
  jclass c = env->GetObjectClass(j);
  h.lm_model = GetStringField(env, j, c, "model");
  h.cfg.lm_config.model = h.lm_model.c_str();
  h.cfg.lm_config.scale = GetFloatField(env, j, c, "scale");
  env->DeleteLocalRef(c);
  env->DeleteLocalRef(j);
}

void ReadHrConfig(JNIEnv* env, jobject jcfg, jclass cfg_cls, OfflineCfg& h) {
  jobject j = GetObjField(env, jcfg, cfg_cls, "hr",
                          "Lcom/dark/ai_sherpa/HomophoneReplacerConfig;");
  if (!j) return;
  jclass c = env->GetObjectClass(j);
  h.hr_lexicon   = GetStringField(env, j, c, "lexicon");
  h.hr_rule_fsts = GetStringField(env, j, c, "ruleFsts");
  h.cfg.hr.lexicon   = h.hr_lexicon.c_str();
  h.cfg.hr.rule_fsts = h.hr_rule_fsts.c_str();
  env->DeleteLocalRef(c);
  env->DeleteLocalRef(j);
}

OfflineCfg ReadOfflineConfig(JNIEnv* env, jobject jconfig) {
  OfflineCfg h;
  jclass cfg_cls = env->GetObjectClass(jconfig);

  ReadFeatConfig(env, jconfig, cfg_cls, h);
  ReadModelConfig(env, jconfig, cfg_cls, h);
  ReadLmConfig(env, jconfig, cfg_cls, h);
  ReadHrConfig(env, jconfig, cfg_cls, h);

  h.decoding_method = GetStringField(env, jconfig, cfg_cls, "decodingMethod");
  h.hotwords_file   = GetStringField(env, jconfig, cfg_cls, "hotwordsFile");
  h.rule_fsts       = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  h.rule_fars       = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  h.cfg.decoding_method   = h.decoding_method.c_str();
  h.cfg.max_active_paths  = std::max(1, (int)GetIntField(env, jconfig, cfg_cls, "maxActivePaths"));
  h.cfg.hotwords_file     = h.hotwords_file.c_str();
  h.cfg.hotwords_score    = GetFloatField(env, jconfig, cfg_cls, "hotwordsScore");
  h.cfg.rule_fsts         = h.rule_fsts.c_str();
  h.cfg.rule_fars         = h.rule_fars.c_str();
  h.cfg.blank_penalty     = GetFloatField(env, jconfig, cfg_cls, "blankPenalty");

  env->DeleteLocalRef(cfg_cls);
  return h;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_newFromFile(
    JNIEnv* env, jobject, jobject jconfig) {
  auto h = ReadOfflineConfig(env, jconfig);

  tn_sec_set_op("OfflineRecognizer.newFromFile");
  TN_D("op-detail: tokens=%s whisper_enc=%s whisper_dec=%s nemo=%s threads=%d",
       h.tokens.c_str(),
       h.whisper_enc.empty() ? "-" : h.whisper_enc.c_str(),
       h.whisper_dec.empty() ? "-" : h.whisper_dec.c_str(),
       h.nemo_ctc_model.empty() ? "-" : h.nemo_ctc_model.c_str(),
       h.cfg.model_config.num_threads);

  const SherpaOnnxOfflineRecognizer* p = SherpaOnnxCreateOfflineRecognizer(&h.cfg);
  if (!p) {
    TN_ERR_FIX(TN_CODE_MODEL_LOAD_FAIL, TN_STAGE_LOAD,
        "Verify the model file, tokens, and free memory.",
        "%s",
        "SherpaOnnxCreateOfflineRecognizer returned null. "
        "Likely missing/corrupt model file, mismatched tokens, or insufficient memory.");
    ThrowIllegalState(env, "Failed to create OfflineRecognizer");
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_delete(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineRecognizer(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer*>(ptr));
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_createStream(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  const SherpaOnnxOfflineStream* stream = SherpaOnnxCreateOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer*>(ptr));
  if (!stream) {
    ThrowIllegalState(env, "Failed to create OfflineStream");
    return 0;
  }
  return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineStream_acceptWaveform(
    JNIEnv* env, jobject, jlong ptr, jint sample_rate, jfloatArray samples) {
  CHECK_PTR(env, ptr, );
  if (!samples) return;
  jsize len = env->GetArrayLength(samples);
  if (len <= 0) return;

  // GetPrimitiveArrayCritical avoids a copy for large audio buffers — but the
  // JNI critical region must stay short. AcceptWaveform copies into the
  // recognizer's own ring buffer synchronously, so this is safe.
  auto* data = static_cast<jfloat*>(env->GetPrimitiveArrayCritical(samples, nullptr));
  if (!data) return;
  SherpaOnnxAcceptWaveformOffline(
      reinterpret_cast<const SherpaOnnxOfflineStream*>(ptr),
      static_cast<int>(sample_rate), data, static_cast<int>(len));
  env->ReleasePrimitiveArrayCritical(samples, data, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineStream_delete(JNIEnv* env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineStream*>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_decode(
    JNIEnv* env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, );
  CHECK_PTR(env, stream_ptr, );
  SherpaOnnxDecodeOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer*>(ptr),
      reinterpret_cast<const SherpaOnnxOfflineStream*>(stream_ptr));
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_getResult(
    JNIEnv* env, jobject, jlong /*ptr*/, jlong stream_ptr) {
  CHECK_PTR(env, stream_ptr, nullptr);

  const SherpaOnnxOfflineRecognizerResult* r = SherpaOnnxGetOfflineStreamResult(
      reinterpret_cast<const SherpaOnnxOfflineStream*>(stream_ptr));
  if (!r) return nullptr;

  jstring text = env->NewStringUTF(r->text ? r->text : "");

  const int n = r->count;
  jobjectArray tokens = env->NewObjectArray(n, g_cache.string_cls, nullptr);
  for (int i = 0; i < n; ++i) {
    jstring t = env->NewStringUTF(r->tokens_arr && r->tokens_arr[i] ? r->tokens_arr[i] : "");
    env->SetObjectArrayElement(tokens, i, t);
    env->DeleteLocalRef(t);
  }

  jfloatArray timestamps = env->NewFloatArray(n);
  if (n > 0 && r->timestamps) {
    env->SetFloatArrayRegion(timestamps, 0, n, r->timestamps);
  }

  jobject result = env->NewObject(g_cache.offline_result_cls,
                                  g_cache.offline_result_ctor,
                                  text, tokens, timestamps);

  env->DeleteLocalRef(text);
  env->DeleteLocalRef(tokens);
  env->DeleteLocalRef(timestamps);

  SherpaOnnxDestroyOfflineRecognizerResult(r);
  return result;
}

}  // extern "C"
