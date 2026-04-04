// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cstring>

static void ReadEndpointRule(JNIEnv *env, jobject rule, jclass cls,
                              SherpaOnnxEndpointRule *out) {
  out->must_contain_nonsilence = GetBoolField(env, rule, cls, "mustContainNonSilence") ? 1 : 0;
  out->min_trailing_silence = GetFloatField(env, rule, cls, "minTrailingSilence");
  out->min_utterance_length = GetFloatField(env, rule, cls, "minUtteranceLength");
}

struct OnlineCfg {
  std::string decoding_method, hotwords_file, rule_fsts, rule_fars;
  std::string tokens, provider, model_type, modeling_unit, bpe_vocab;
  std::string trans_enc, trans_dec, trans_joi;
  std::string para_enc, para_dec;
  std::string zf_model, nemo_model;
  std::string lm_model;
  std::string ctc_graph;
  std::string hr_lexicon, hr_rule_fsts;
  SherpaOnnxOnlineRecognizerConfig cfg{};
};

static OnlineCfg ReadOnlineConfig(JNIEnv *env, jobject jconfig) {
  OnlineCfg h;

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
                                "Lcom/dark/ai_sherpa/OnlineModelConfig;");
  if (jmodel) {
    jclass mc = env->GetObjectClass(jmodel);

    jobject jtrans = GetObjField(env, jmodel, mc, "transducer",
                                  "Lcom/dark/ai_sherpa/OnlineTransducerModelConfig;");
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
                                 "Lcom/dark/ai_sherpa/OnlineParaformerModelConfig;");
    if (jpara) {
      jclass pc = env->GetObjectClass(jpara);
      h.para_enc = GetStringField(env, jpara, pc, "encoder");
      h.para_dec = GetStringField(env, jpara, pc, "decoder");
      h.cfg.model_config.paraformer.encoder = h.para_enc.c_str();
      h.cfg.model_config.paraformer.decoder = h.para_dec.c_str();
      env->DeleteLocalRef(pc);
      env->DeleteLocalRef(jpara);
    }

    jobject jzf = GetObjField(env, jmodel, mc, "zipformer2Ctc",
                               "Lcom/dark/ai_sherpa/OnlineZipformer2CtcModelConfig;");
    if (jzf) {
      jclass zc = env->GetObjectClass(jzf);
      h.zf_model = GetStringField(env, jzf, zc, "model");
      h.cfg.model_config.zipformer2_ctc.model = h.zf_model.c_str();
      env->DeleteLocalRef(zc);
      env->DeleteLocalRef(jzf);
    }

    jobject jnemo = GetObjField(env, jmodel, mc, "neMoCtc",
                                 "Lcom/dark/ai_sherpa/OnlineNeMoCtcModelConfig;");
    if (jnemo) {
      jclass nc = env->GetObjectClass(jnemo);
      h.nemo_model = GetStringField(env, jnemo, nc, "model");
      h.cfg.model_config.nemo_ctc.model = h.nemo_model.c_str();
      env->DeleteLocalRef(nc);
      env->DeleteLocalRef(jnemo);
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
                              "Lcom/dark/ai_sherpa/OnlineLMConfig;");
  if (jlm) {
    jclass lc = env->GetObjectClass(jlm);
    h.lm_model = GetStringField(env, jlm, lc, "model");
    h.cfg.lm_config.model = h.lm_model.c_str();
    h.cfg.lm_config.scale = GetFloatField(env, jlm, lc, "scale");
    env->DeleteLocalRef(lc);
    env->DeleteLocalRef(jlm);
  }

  // endpoint config
  jobject jep = GetObjField(env, jconfig, cfg_cls, "endpointConfig",
                             "Lcom/dark/ai_sherpa/EndpointConfig;");
  if (jep) {
    jclass ec = env->GetObjectClass(jep);

    jobject jr1 = GetObjField(env, jep, ec, "rule1", "Lcom/dark/ai_sherpa/EndpointRule;");
    jobject jr2 = GetObjField(env, jep, ec, "rule2", "Lcom/dark/ai_sherpa/EndpointRule;");
    jobject jr3 = GetObjField(env, jep, ec, "rule3", "Lcom/dark/ai_sherpa/EndpointRule;");
    if (jr1) {
      jclass rc = env->GetObjectClass(jr1);
      ReadEndpointRule(env, jr1, rc, &h.cfg.endpoint_config.rule1);
      if (jr2) ReadEndpointRule(env, jr2, rc, &h.cfg.endpoint_config.rule2);
      if (jr3) ReadEndpointRule(env, jr3, rc, &h.cfg.endpoint_config.rule3);
      env->DeleteLocalRef(rc);
    }
    if (jr1) env->DeleteLocalRef(jr1);
    if (jr2) env->DeleteLocalRef(jr2);
    if (jr3) env->DeleteLocalRef(jr3);
    env->DeleteLocalRef(ec);
    env->DeleteLocalRef(jep);
  }

  // ctc fst decoder
  jobject jctc = GetObjField(env, jconfig, cfg_cls, "ctcFstDecoderConfig",
                              "Lcom/dark/ai_sherpa/OnlineCtcFstDecoderConfig;");
  if (jctc) {
    jclass cc = env->GetObjectClass(jctc);
    h.ctc_graph = GetStringField(env, jctc, cc, "graph");
    h.cfg.ctc_fst_decoder_config.graph = h.ctc_graph.c_str();
    h.cfg.ctc_fst_decoder_config.max_active = GetIntField(env, jctc, cc, "maxActive");
    env->DeleteLocalRef(cc);
    env->DeleteLocalRef(jctc);
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
  h.cfg.enable_endpoint = GetBoolField(env, jconfig, cfg_cls, "enableEndpoint") ? 1 : 0;

  env->DeleteLocalRef(cfg_cls);
  return h;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  auto h = ReadOnlineConfig(env, jconfig);
  const SherpaOnnxOnlineRecognizer *p = SherpaOnnxCreateOnlineRecognizer(&h.cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OnlineRecognizer");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_newFromAsset(
    JNIEnv *env, jobject, jobject asset_manager, jobject jconfig) {
  auto h = ReadOnlineConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const SherpaOnnxOnlineRecognizer *p =
      SherpaOnnxCreateOnlineRecognizerFromAsset(mgr, &h.cfg);
  if (!p) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OnlineRecognizer from asset");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(p);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOnlineRecognizer(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr));
}

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_createStream(
    JNIEnv *env, jobject, jlong ptr, jstring hotwords) {
  CHECK_PTR(env, ptr, 0);
  const SherpaOnnxOnlineRecognizer *recognizer =
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr);
  const SherpaOnnxOnlineStream *stream;
  if (hotwords) {
    const char *hw = env->GetStringUTFChars(hotwords, nullptr);
    stream = SherpaOnnxCreateOnlineStreamWithHotwords(recognizer, hw);
    env->ReleaseStringUTFChars(hotwords, hw);
  } else {
    stream = SherpaOnnxCreateOnlineStream(recognizer);
  }
  if (!stream) {
    jclass ex = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(ex, "Failed to create OnlineStream");
    env->DeleteLocalRef(ex);
    return 0;
  }
  return reinterpret_cast<jlong>(stream);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_reset(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, );
  CHECK_PTR(env, stream_ptr, );
  SherpaOnnxOnlineRecognizerReset(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOnlineStream *>(stream_ptr));
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_isReady(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, JNI_FALSE);
  CHECK_PTR(env, stream_ptr, JNI_FALSE);
  return SherpaOnnxOnlineRecognizerIsReady(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOnlineStream *>(stream_ptr))
      ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_isEndpoint(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, JNI_FALSE);
  CHECK_PTR(env, stream_ptr, JNI_FALSE);
  return SherpaOnnxOnlineRecognizerIsEndpoint(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOnlineStream *>(stream_ptr))
      ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_decode(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, );
  CHECK_PTR(env, stream_ptr, );
  SherpaOnnxDecodeOnlineStream(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOnlineStream *>(stream_ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_decodeStreams(
    JNIEnv *env, jobject, jlong ptr, jlongArray stream_ptrs) {
  CHECK_PTR(env, ptr, );
  jsize n = env->GetArrayLength(stream_ptrs);
  if (n == 0) return;

  jlong *raw = reinterpret_cast<jlong *>(
      env->GetPrimitiveArrayCritical(stream_ptrs, nullptr));
  if (!raw) return;

  auto **streams = new const SherpaOnnxOnlineStream *[n];
  for (jsize i = 0; i < n; ++i) {
    streams[i] = reinterpret_cast<const SherpaOnnxOnlineStream *>(raw[i]);
  }
  env->ReleasePrimitiveArrayCritical(stream_ptrs, raw, JNI_ABORT);

  SherpaOnnxDecodeMultipleOnlineStreams(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      streams, static_cast<int>(n));
  delete[] streams;
}

JNIEXPORT jobject JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_getResult(
    JNIEnv *env, jobject, jlong ptr, jlong stream_ptr) {
  CHECK_PTR(env, ptr, nullptr);
  CHECK_PTR(env, stream_ptr, nullptr);

  const SherpaOnnxOnlineRecognizerResult *r = SherpaOnnxGetOnlineStreamResult(
      reinterpret_cast<const SherpaOnnxOnlineRecognizer *>(ptr),
      reinterpret_cast<const SherpaOnnxOnlineStream *>(stream_ptr));
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

  jfloatArray words_arr = env->NewFloatArray(0);

  jobject result = env->NewObject(
      g_cache.online_result_cls, g_cache.online_result_ctor,
      text, tokens, timestamps, words_arr);

  env->DeleteLocalRef(text);
  env->DeleteLocalRef(tokens);
  env->DeleteLocalRef(timestamps);
  env->DeleteLocalRef(words_arr);

  SherpaOnnxDestroyOnlineRecognizerResult(r);
  return result;
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineStream_acceptWaveform(
    JNIEnv *env, jobject, jlong ptr, jint sample_rate, jfloatArray samples) {
  CHECK_PTR(env, ptr, );
  jsize len = env->GetArrayLength(samples);
  jfloat *data = reinterpret_cast<jfloat *>(
      env->GetPrimitiveArrayCritical(samples, nullptr));
  if (!data) return;
  SherpaOnnxOnlineStreamAcceptWaveform(
      reinterpret_cast<const SherpaOnnxOnlineStream *>(ptr),
      static_cast<int>(sample_rate), data, static_cast<int>(len));
  env->ReleasePrimitiveArrayCritical(samples, data, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineStream_inputFinished(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxOnlineStreamInputFinished(
      reinterpret_cast<const SherpaOnnxOnlineStream *>(ptr));
}

JNIEXPORT void JNICALL
Java_com_dark_ai_1sherpa_OnlineStream_delete(
    JNIEnv *env, jobject, jlong ptr) {
  CHECK_PTR(env, ptr, );
  SherpaOnnxDestroyOnlineStream(
      reinterpret_cast<const SherpaOnnxOnlineStream *>(ptr));
}

} // extern "C"
