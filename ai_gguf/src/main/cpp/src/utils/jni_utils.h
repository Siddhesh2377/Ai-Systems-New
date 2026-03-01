#pragma once
#include <jni.h>
#include <string>
#include <vector>

namespace jni {

std::string to_string(JNIEnv* env, jstring s);
jstring to_jstring(JNIEnv* env, const std::string& s);

struct Callback {
    JNIEnv* env = nullptr;
    jobject obj = nullptr;
    jmethodID on_token     = nullptr;
    jmethodID on_tool_call = nullptr;
    jmethodID on_done      = nullptr;
    jmethodID on_error     = nullptr;
    jmethodID on_metrics   = nullptr;

    bool init(JNIEnv* e, jobject callback);
    void token(const std::string& text);
    void tool_call(const std::string& name, const std::string& args);
    void done();
    void error(const std::string& msg);
    void metrics(float tps, float ttft_ms, float total_ms,
                 int n_eval, int n_pred,
                 float model_mb, float ctx_mb, float peak_mb, float mem_pct);
};

} // namespace jni
