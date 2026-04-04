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

static SherpaOnnxOnlineRecognizerConfig ReadOnlineConfig(JNIEnv *env, jobject jconfig) {
  SherpaOnnxOnlineRecognizerConfig cfg{};

  jclass cfg_cls = env->GetObjectClass(jconfig);

  // feat config
  jobject jfeat = GetObjField(env, jconfig, cfg_cls, "featConfig",
                               "Lcom/dark/ai_sherpa/FeatureConfig;");
  if (jfeat) {
    jclass fc = env->GetObjectClass(jfeat);
    cfg.feat_config.sample_rate = GetIntField(env, jfeat, fc, "sampleRate");
    cfg.feat_config.feature_dim = GetIntField(env, jfeat, fc, "featureDim");
    cfg.feat_config.dither = GetFloatField(env, jfeat, fc, "dither");
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
      static std::string enc = GetStringField(env, jtrans, tc, "encoder");
      static std::string dec = GetStringField(env, jtrans, tc, "decoder");
      static std::string joi = GetStringField(env, jtrans, tc, "joiner");
      cfg.model_config.transducer.encoder = enc.c_str();
      cfg.model_config.transducer.decoder = dec.c_str();
      cfg.model_config.transducer.joiner = joi.c_str();
      env->DeleteLocalRef(tc);
      env->DeleteLocalRef(jtrans);
    }

    jobject jpara = GetObjField(env, jmodel, mc, "paraformer",
                                 "Lcom/dark/ai_sherpa/OnlineParaformerModelConfig;");
    if (jpara) {
      jclass pc = env->GetObjectClass(jpara);
      static std::string enc = GetStringField(env, jpara, pc, "encoder");
      static std::string dec = GetStringField(env, jpara, pc, "decoder");
      cfg.model_config.paraformer.encoder = enc.c_str();
      cfg.model_config.paraformer.decoder = dec.c_str();
      env->DeleteLocalRef(pc);
      env->DeleteLocalRef(jpara);
    }

    jobject jzf = GetObjField(env, jmodel, mc, "zipformer2Ctc",
                               "Lcom/dark/ai_sherpa/OnlineZipformer2CtcModelConfig;");
    if (jzf) {
      jclass zc = env->GetObjectClass(jzf);
      static std::string m = GetStringField(env, jzf, zc, "model");
      cfg.model_config.zipformer2_ctc.model = m.c_str();
      env->DeleteLocalRef(zc);
      env->DeleteLocalRef(jzf);
    }

    jobject jnemo = GetObjField(env, jmodel, mc, "neMoCtc",
                                 "Lcom/dark/ai_sherpa/OnlineNeMoCtcModelConfig;");
    if (jnemo) {
      jclass nc = env->GetObjectClass(jnemo);
      static std::string m = GetStringField(env, jnemo, nc, "model");
      cfg.model_config.nemo_ctc.model = m.c_str();
      env->DeleteLocalRef(nc);
      env->DeleteLocalRef(jnemo);
    }

    static std::string tokens = GetStringField(env, jmodel, mc, "tokens");
    static std::string provider = GetStringField(env, jmodel, mc, "provider");
    static std::string model_type = GetStringField(env, jmodel, mc, "modelType");
    static std::string modeling_unit = GetStringField(env, jmodel, mc, "modelingUnit");
    static std::string bpe_vocab = GetStringField(env, jmodel, mc, "bpeVocab");
    cfg.model_config.tokens = tokens.c_str();
    cfg.model_config.num_threads = GetIntField(env, jmodel, mc, "numThreads");
    cfg.model_config.debug = GetBoolField(env, jmodel, mc, "debug") ? 1 : 0;
    cfg.model_config.provider = provider.c_str();
    cfg.model_config.model_type = model_type.c_str();
    cfg.model_config.modeling_unit = modeling_unit.c_str();
    cfg.model_config.bpe_vocab = bpe_vocab.c_str();

    env->DeleteLocalRef(mc);
    env->DeleteLocalRef(jmodel);
  }

  // lm config
  jobject jlm = GetObjField(env, jconfig, cfg_cls, "lmConfig",
                              "Lcom/dark/ai_sherpa/OnlineLMConfig;");
  if (jlm) {
    jclass lc = env->GetObjectClass(jlm);
    static std::string m = GetStringField(env, jlm, lc, "model");
    cfg.lm_config.model = m.c_str();
    cfg.lm_config.scale = GetFloatField(env, jlm, lc, "scale");
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
      ReadEndpointRule(env, jr1, rc, &cfg.endpoint_config.rule1);
      if (jr2) ReadEndpointRule(env, jr2, rc, &cfg.endpoint_config.rule2);
      if (jr3) ReadEndpointRule(env, jr3, rc, &cfg.endpoint_config.rule3);
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
    static std::string graph = GetStringField(env, jctc, cc, "graph");
    cfg.ctc_fst_decoder_config.graph = graph.c_str();
    cfg.ctc_fst_decoder_config.max_active = GetIntField(env, jctc, cc, "maxActive");
    env->DeleteLocalRef(cc);
    env->DeleteLocalRef(jctc);
  }

  // hr config
  jobject jhr = GetObjField(env, jconfig, cfg_cls, "hr",
                             "Lcom/dark/ai_sherpa/HomophoneReplacerConfig;");
  if (jhr) {
    jclass hc = env->GetObjectClass(jhr);
    static std::string lex = GetStringField(env, jhr, hc, "lexicon");
    static std::string rfsts = GetStringField(env, jhr, hc, "ruleFsts");
    cfg.hr.lexicon = lex.c_str();
    cfg.hr.rule_fsts = rfsts.c_str();
    env->DeleteLocalRef(hc);
    env->DeleteLocalRef(jhr);
  }

  static std::string decoding_method = GetStringField(env, jconfig, cfg_cls, "decodingMethod");
  static std::string hotwords_file = GetStringField(env, jconfig, cfg_cls, "hotwordsFile");
  static std::string rule_fsts = GetStringField(env, jconfig, cfg_cls, "ruleFsts");
  static std::string rule_fars = GetStringField(env, jconfig, cfg_cls, "ruleFars");
  cfg.decoding_method = decoding_method.c_str();
  cfg.max_active_paths = GetIntField(env, jconfig, cfg_cls, "maxActivePaths");
  cfg.hotwords_file = hotwords_file.c_str();
  cfg.hotwords_score = GetFloatField(env, jconfig, cfg_cls, "hotwordsScore");
  cfg.rule_fsts = rule_fsts.c_str();
  cfg.rule_fars = rule_fars.c_str();
  cfg.blank_penalty = GetFloatField(env, jconfig, cfg_cls, "blankPenalty");
  cfg.enable_endpoint = GetBoolField(env, jconfig, cfg_cls, "enableEndpoint") ? 1 : 0;

  env->DeleteLocalRef(cfg_cls);
  return cfg;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OnlineRecognizer_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  SherpaOnnxOnlineRecognizerConfig cfg = ReadOnlineConfig(env, jconfig);
  const SherpaOnnxOnlineRecognizer *p = SherpaOnnxCreateOnlineRecognizer(&cfg);
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
  SherpaOnnxOnlineRecognizerConfig cfg = ReadOnlineConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const SherpaOnnxOnlineRecognizer *p =
      SherpaOnnxCreateOnlineRecognizerFromAsset(mgr, &cfg);
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
