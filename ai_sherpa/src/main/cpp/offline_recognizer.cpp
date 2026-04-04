// Copyright (c) 2025 Dark Matter Labs
#include "jni_common.h"
#include "jni_cache.h"
#include "sherpa-onnx/c-api/c-api.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

static SherpaOnnxOfflineRecognizerConfig ReadOfflineConfig(JNIEnv *env, jobject jconfig) {
  SherpaOnnxOfflineRecognizerConfig cfg{};

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
                                "Lcom/dark/ai_sherpa/OfflineModelConfig;");
  if (jmodel) {
    jclass mc = env->GetObjectClass(jmodel);

    jobject jtrans = GetObjField(env, jmodel, mc, "transducer",
                                  "Lcom/dark/ai_sherpa/OfflineTransducerModelConfig;");
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
                                 "Lcom/dark/ai_sherpa/OfflineParaformerModelConfig;");
    if (jpara) {
      jclass pc = env->GetObjectClass(jpara);
      static std::string m = GetStringField(env, jpara, pc, "model");
      cfg.model_config.paraformer.model = m.c_str();
      env->DeleteLocalRef(pc);
      env->DeleteLocalRef(jpara);
    }

    jobject jnemo = GetObjField(env, jmodel, mc, "nemoCtc",
                                 "Lcom/dark/ai_sherpa/OfflineNemoEncDecCtcModelConfig;");
    if (jnemo) {
      jclass nc = env->GetObjectClass(jnemo);
      static std::string m = GetStringField(env, jnemo, nc, "model");
      cfg.model_config.nemo_ctc.model = m.c_str();
      env->DeleteLocalRef(nc);
      env->DeleteLocalRef(jnemo);
    }

    jobject jwhisper = GetObjField(env, jmodel, mc, "whisper",
                                    "Lcom/dark/ai_sherpa/OfflineWhisperModelConfig;");
    if (jwhisper) {
      jclass wc = env->GetObjectClass(jwhisper);
      static std::string enc = GetStringField(env, jwhisper, wc, "encoder");
      static std::string dec = GetStringField(env, jwhisper, wc, "decoder");
      static std::string lang = GetStringField(env, jwhisper, wc, "language");
      static std::string task = GetStringField(env, jwhisper, wc, "task");
      cfg.model_config.whisper.encoder = enc.c_str();
      cfg.model_config.whisper.decoder = dec.c_str();
      cfg.model_config.whisper.language = lang.c_str();
      cfg.model_config.whisper.task = task.c_str();
      cfg.model_config.whisper.tail_paddings = GetIntField(env, jwhisper, wc, "tailPaddings");
      env->DeleteLocalRef(wc);
      env->DeleteLocalRef(jwhisper);
    }

    jobject jtdnn = GetObjField(env, jmodel, mc, "tdnn",
                                 "Lcom/dark/ai_sherpa/OfflineTdnnModelConfig;");
    if (jtdnn) {
      jclass tc = env->GetObjectClass(jtdnn);
      static std::string m = GetStringField(env, jtdnn, tc, "model");
      cfg.model_config.tdnn.model = m.c_str();
      env->DeleteLocalRef(tc);
      env->DeleteLocalRef(jtdnn);
    }

    jobject jnemo_t = GetObjField(env, jmodel, mc, "nemoTransducer",
                                   "Lcom/dark/ai_sherpa/OfflineNemoEncDecRnntModelConfig;");
    if (jnemo_t) {
      jclass ntc = env->GetObjectClass(jnemo_t);
      static std::string enc = GetStringField(env, jnemo_t, ntc, "encoder");
      static std::string dec = GetStringField(env, jnemo_t, ntc, "decoder");
      static std::string joi = GetStringField(env, jnemo_t, ntc, "joiner");
      cfg.model_config.nemo_transducer.encoder = enc.c_str();
      cfg.model_config.nemo_transducer.decoder = dec.c_str();
      cfg.model_config.nemo_transducer.joiner = joi.c_str();
      env->DeleteLocalRef(ntc);
      env->DeleteLocalRef(jnemo_t);
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
                              "Lcom/dark/ai_sherpa/OfflineLMConfig;");
  if (jlm) {
    jclass lc = env->GetObjectClass(jlm);
    static std::string m = GetStringField(env, jlm, lc, "model");
    cfg.lm_config.model = m.c_str();
    cfg.lm_config.scale = GetFloatField(env, jlm, lc, "scale");
    env->DeleteLocalRef(lc);
    env->DeleteLocalRef(jlm);
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

  env->DeleteLocalRef(cfg_cls);
  return cfg;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_dark_ai_1sherpa_OfflineRecognizer_newFromFile(
    JNIEnv *env, jobject, jobject jconfig) {
  SherpaOnnxOfflineRecognizerConfig cfg = ReadOfflineConfig(env, jconfig);
  const SherpaOnnxOfflineRecognizer *p = SherpaOnnxCreateOfflineRecognizer(&cfg);
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
  SherpaOnnxOfflineRecognizerConfig cfg = ReadOfflineConfig(env, jconfig);
  AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
  const SherpaOnnxOfflineRecognizer *p =
      SherpaOnnxCreateOfflineRecognizerFromAsset(mgr, &cfg);
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
