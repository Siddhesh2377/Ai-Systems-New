// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

struct OfflineCfg {
  std::string decoding_method, hotwords_file, rule_fsts, rule_fars;
  std::string tokens, provider, model_type, modeling_unit, bpe_vocab;
  std::string trans_enc, trans_dec, trans_joi;
  std::string paraformer_model, nemo_ctc_model;
  std::string whisper_enc, whisper_dec, whisper_lang, whisper_task;
  std::string tdnn_model;
  std::string nemo_trans_enc, nemo_trans_dec, nemo_trans_joi;
  std::string lm_model;
  std::string hr_lexicon, hr_rule_fsts;
  SherpaOnnxOfflineRecognizerConfig cfg{};
};

static OfflineCfg ReadOfflineConfig(JNIEnv *env, jobject jconfig) {
  OfflineCfg h;

  jclass cfg_cls = env->GetObjectClass(jconfig);

  // feat config
  jobject jfeat = GetObjField(env, jconfig, cfg_cls, "featConfig",
                               "Lcom/dark/ai_sherpa/FeatureConfig;");
  if (jfeat) {
    jclass fc = env->GetObjectClass(jfeat);
    h.cfg.feat_config.sample_rate = GetIntField(env, jfeat, fc, "sampleRate");
    h.cfg.feat_config.feature_dim = GetIntField(env, jfeat, fc, "featureDim");
    h.cfg.feat_config.dither = GetFloatField(env, jfeat, fc, "dither");
    env->DeleteLocalRef(fc);
    env->DeleteLocalRef(jfeat);
  }

  // model config
  jobject jmodel = GetObjField(env, jconfig, cfg_cls, "modelConfig",
                                "Lcom/dark/ai_sherpa/OfflineModelConfig;");
  if (jmodel) {
    jclass mc = env->GetObjectClass(jmodel);

    jobject jtrans = GetObjField(env, jmodel, mc, "transducer",
                                  "Lcom/dark/ai_sherpa/OfflineTransducerModelConfig;");
    if (jtrans) {
      jclass tc = env->GetObjectClass(jtrans);
      h.trans_enc = GetStringField(env, jtrans, tc, "encoder");
      h.trans_dec = GetStringField(env, jtrans, tc, "decoder");
      h.trans_joi = GetStringField(env, jtrans, tc, "joiner");
      h.cfg.model_config.transducer.encoder = h.trans_enc.c_str();
      h.cfg.model_config.transducer.decoder = h.trans_dec.c_str();
      h.cfg.model_config.transducer.joiner = h.trans_joi.c_str();
      env->DeleteLocalRef(tc);
      env->DeleteLocalRef(jtrans);
    }

    jobject jpara = GetObjField(env, jmodel, mc, "paraformer",
                                 "Lcom/dark/ai_sherpa/OfflineParaformerModelConfig;");
    if (jpara) {
      jclass pc = env->GetObjectClass(jpara);
      h.paraformer_model = GetStringField(env, jpara, pc, "model");
      h.cfg.model_config.paraformer.model = h.paraformer_model.c_str();
      env->DeleteLocalRef(pc);
      env->DeleteLocalRef(jpara);
    }

    jobject jnemo = GetObjField(env, jmodel, mc, "nemoCtc",
                                 "Lcom/dark/ai_sherpa/OfflineNemoEncDecCtcModelConfig;");
    if (jnemo) {
      jclass nc = env->GetObjectClass(jnemo);
      h.nemo_ctc_model = GetStringField(env, jnemo, nc, "model");
      h.cfg.model_config.nemo_ctc.model = h.nemo_ctc_model.c_str();
      env->DeleteLocalRef(nc);
      env->DeleteLocalRef(jnemo);
    }

    jobject jwhisper = GetObjField(env, jmodel, mc, "whisper",
                                    "Lcom/dark/ai_sherpa/OfflineWhisperModelConfig;");
    if (jwhisper) {
      jclass wc = env->GetObjectClass(jwhisper);
      h.whisper_enc = GetStringField(env, jwhisper, wc, "encoder");
      h.whisper_dec = GetStringField(env, jwhisper, wc, "decoder");
      h.whisper_lang = GetStringField(env, jwhisper, wc, "language");
      h.whisper_task = GetStringField(env, jwhisper, wc, "task");
      h.cfg.model_config.whisper.encoder = h.whisper_enc.c_str();
      h.cfg.model_config.whisper.decoder = h.whisper_dec.c_str();
      h.cfg.model_config.whisper.language = h.whisper_lang.c_str();
      h.cfg.model_config.whisper.task = h.whisper_task.c_str();
      h.cfg.model_config.whisper.tail_paddings = GetIntField(env, jwhisper, wc, "tailPaddings");
      env->DeleteLocalRef(wc);
      env->DeleteLocalRef(jwhisper);
    }

    jobject jtdnn = GetObjField(env, jmodel, mc, "tdnn",
                                 "Lcom/dark/ai_sherpa/OfflineTdnnModelConfig;");
    if (jtdnn) {
      jclass tc = env->GetObjectClass(jtdnn);
      h.tdnn_model = GetStringField(env, jtdnn, tc, "model");
      h.cfg.model_config.tdnn.model = h.tdnn_model.c_str();
      env->DeleteLocalRef(tc);
      env->DeleteLocalRef(jtdnn);
    }

    jobject jnemo_t = GetObjField(env, jmodel, mc, "nemoTransducer",
                                   "Lcom/dark/ai_sherpa/OfflineNemoEncDecRnntModelConfig;");
    if (jnemo_t) {
      jclass ntc = env->GetObjectClass(jnemo_t);
      h.nemo_trans_enc = GetStringField(env, jnemo_t, ntc, "encoder");
      h.nemo_trans_dec = GetStringField(env, jnemo_t, ntc, "decoder");
      h.nemo_trans_joi = GetStringField(env, jnemo_t, ntc, "joiner");
      h.cfg.model_config.nemo_transducer.encoder = h.nemo_trans_enc.c_str();
      h.cfg.model_config.nemo_transducer.decoder = h.nemo_trans_dec.c_str();
      h.cfg.model_config.nemo_transducer.joiner = h.nemo_trans_joi.c_str();
      env->DeleteLocalRef(ntc);
      env->DeleteLocalRef(jnemo_t);
    }

    h.tokens = GetStringField(env, jmodel, mc, "tokens");
    h.provider = GetStringField(env, jmodel, mc, "provider");
    h.model_type = GetStringField(env, jmodel, mc, "modelType");
    h.modeling_unit = GetStringField(env, jmodel, mc, "modelingUnit");
    h.bpe_vocab = GetStringField(env, jmodel, mc, "bpeVocab");
    h.cfg.model_config.tokens = h.tokens.c_str();
    h.cfg.model_config.num_threads = GetIntField(env, jmodel, mc, "numThreads");
    h.cfg.model_config.debug = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
    h.cfg.model_config.provider = h.provider.c_str();
    h.cfg.model_config.model_type = h.model_type.c_str();
    h.cfg.model_config.modeling_unit = h.modeling_unit.c_str();
    h.cfg.model_config.bpe_vocab = h.bpe_vocab.c_str();

    env->DeleteLocalRef(mc);
    env->DeleteLocalRef(jmodel);
  }

  // lm config
  jobject jlm = GetObjField(env, jconfig, cfg_cls, "lmConfig",
                              "Lcom/dark/ai_sherpa/OfflineLMConfig;");
  if (jlm) {
    jclass lc = env->GetObjectClass(jlm);
    h.lm_model = GetStringField(env, jlm, lc, "model");
    h.cfg.lm_config.model = h.lm_model.c_str();
    h.cfg.lm_config.scale = GetFloatField(env, jlm, lc, "scale");
    env->DeleteLocalRef(lc);
    env->DeleteLocalRef(jlm);
  }

  // hr config
  jobject jhr = GetObjField(env, jconfig, cfg_cls, "hr",
                             "Lcom/dark/ai_sherpa/HomophoneReplacerConfig;");
  if (jhr) {
    jclass hc = env->GetObjectClass(jhr);
    h.hr_lexicon = GetStringField(env, jhr, hc, "lexicon");
    h.hr_rule_fsts = GetStringField(env, jhr, hc, "ruleFsts");
    h.cfg.hr.lexicon = h.hr_lexicon.c_str();
    h.cfg.hr.rule_fsts = h.hr_rule_fsts.c_str();
    env->DeleteLocalRef(hc);
    env->DeleteLocalRef(jhr);
  }

  h.decoding_method = GetStringField(env, jconfig, cfg_cls, "decodingMethod");
  h.hotwords_file = GetStringField(env, jconfig, cfg_cls, "hotwordsFile");
  h.rule_fsts = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  h.rule_fars = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  h.cfg.decoding_method = h.decoding_method.c_str();
  h.cfg.max_active_paths = GetIntField(env, jconfig, cfg_cls, "maxActivePaths");
  h.cfg.hotwords_file = h.hotwords_file.c_str();
  h.cfg.hotwords_score = GetFloatField(env, jconfig, cfg_cls, "hotwordsScore");
  h.cfg.rule_fsts = h.rule_fsts.c_str();
  h.cfg.rule_fars = h.rule_fars.c_str();
  h.cfg.blank_penalty = GetFloatField(env, jconfig, cfg_cls, "blankPenalty");

  env->DeleteLocalRef(cfg_cls);
  return h;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  auto h = ReadOfflineConfig(env, jconfig);
  const SherpaOnnxOfflineRecognizer *p = SherpaOnnxCreateOfflineRecognizer(&h.cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineRecognizer");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_newFromAsset(
    JNIEnv *env, jobject, jobject asset_manager, jobject jconfig) {
  auto h = ReadOfflineConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const SherpaOnnxOfflineRecognizer *p =
      SherpaOnnxCreateOfflineRecognizerFromAsset(mgr, &h.cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineRecognizer from asset");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineRecognizer(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer *>(ptr));
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_createStream(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, 0);
  const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer *>(ptr));
  if (!stream) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OfflineStream");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineStream_acceptWaveform(
    JNIEnv *env, jobject, jlong ptr, jint sample_rate, jfloatArray samples) {
  CHECK_PTR(env, ptr, );
  jsize len = env->GetArrayLength(samples);
  jfloat *data = reinterpret_cast<jfloat *>(
      env->GetPrimitiveArrayCritical(samples, nullptr));
  if (!data) return;
  SherpaOnnxAcceptWaveformOffline(
      reinterpret_cast<const SherpaOnnxOfflineStream *>(ptr),
      static_cast<int>(sample_rate), data, static_cast<int>(len));
  env->ReleasePrimitiveArrayCritical(samples, data, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineStream_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineStream *>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_decode(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, );
  CHECK_PTR(env, stream_ptr, );
  SherpaOnnxDecodeOfflineStream(
      reinterpret_cast<const SherpaOnnxOfflineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOfflineStream *>(stream_ptr));
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_getResult(
    JNIEnv *env, jobject, jlong stream_ptr) {
  CHECK_PTR(env, stream_ptr, nullptr);

  const SherpaOnnxOfflineRecognizerResult *r = SherpaOnnxGetOfflineStreamResult(
      reinterpret_cast<const SherpaOnnxOfflineStream *>(stream_ptr));
  if (!r) return nullptr;

  jstring text = env->NewStringUTF(r->text ? r->text : "");

  int n_tokens = r->count;
  jobjectArray tokens = env->NewObjectArray(n_tokens, g_cache.string_cls, nullptr);
  for (int i = 0; i < n_tokens; ++i) {
    jstring t = env->NewStringUTF(r->tokens[i] ? r->tokens[i] : "");
    env->SetObjectArrayElement(tokens, i, t);
    env->DeleteLocalRef(t);
  }

  jfloatArray timestamps = env->NewFloatArray(n_tokens);
  if (r->timestamps && n_tokens > 0) {
    env->SetFloatArrayRegion(timestamps, 0, n_tokens, r->timestamps);
  }

  jobject result = env->NewObject(
      g_cache.offline_result_cls, g_cache.offline_result_ctor,
      text, tokens, timestamps);

  env->DeleteLocalRef(text);
  env->DeleteLocalRef(tokens);
  env->DeleteLocalRef(timestamps);

  SherpaOnnxDestroyOfflineRecognizerResult(r);
  return result;
}

} // extern "C"
