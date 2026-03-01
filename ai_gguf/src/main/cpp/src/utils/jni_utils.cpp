#include "jni_utils.h"
#include <vector>

namespace jni {

std::string to_string(JNIEnv* env, jstring s) {
    if (!s) return "";
    const jchar* chars = env->GetStringChars(s, nullptr);
    jsize len = env->GetStringLength(s);
    std::string result;
    result.reserve(len * 3);
    for (jsize i = 0; i < len; i++) {
        jchar c = chars[i];
        if (c < 0x80) {
            result += (char)c;
        } else if (c < 0x800) {
            result += (char)(0xC0 | (c >> 6));
            result += (char)(0x80 | (c & 0x3F));
        } else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < len) {
            jchar c2 = chars[i + 1];
            if (c2 >= 0xDC00 && c2 <= 0xDFFF) {
                uint32_t cp = 0x10000 + ((c - 0xD800) << 10) + (c2 - 0xDC00);
                result += (char)(0xF0 | (cp >> 18));
                result += (char)(0x80 | ((cp >> 12) & 0x3F));
                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                result += (char)(0x80 | (cp & 0x3F));
                i++;
            }
        } else {
            result += (char)(0xE0 | (c >> 12));
            result += (char)(0x80 | ((c >> 6) & 0x3F));
            result += (char)(0x80 | (c & 0x3F));
        }
    }
    env->ReleaseStringChars(s, chars);
    return result;
}

jstring to_jstring(JNIEnv* env, const std::string& s) {
    if (s.empty()) return env->NewStringUTF("");

    bool ascii = true;
    for (unsigned char c : s) {
        if (c >= 0x80) { ascii = false; break; }
    }
    if (ascii) return env->NewStringUTF(s.c_str());

    std::vector<jchar> buf;
    buf.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint32_t cp;
        uint8_t b = (uint8_t)s[i];
        if (b < 0x80) {
            cp = b; i += 1;
        } else if ((b & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((b & 0x1F) << 6) | (s[i+1] & 0x3F); i += 2;
        } else if ((b & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((b & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); i += 3;
        } else if ((b & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((b & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); i += 4;
        } else {
            cp = 0xFFFD; i += 1;
        }
        if (cp < 0x10000) {
            buf.push_back((jchar)cp);
        } else {
            cp -= 0x10000;
            buf.push_back((jchar)(0xD800 + (cp >> 10)));
            buf.push_back((jchar)(0xDC00 + (cp & 0x3FF)));
        }
    }
    return env->NewString(buf.data(), (jsize)buf.size());
}

bool Callback::init(JNIEnv* e, jobject callback) {
    env = e;
    obj = callback;
    if (!callback) return false;
    jclass cls = env->GetObjectClass(callback);
    if (!cls) return false;
    on_token     = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
    on_tool_call = env->GetMethodID(cls, "onToolCall", "(Ljava/lang/String;Ljava/lang/String;)V");
    on_done      = env->GetMethodID(cls, "onDone", "()V");
    on_error     = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
    on_metrics   = env->GetMethodID(cls, "onMetrics", "(FFFIIFFFF)V");
    env->DeleteLocalRef(cls);
    return on_token && on_done;
}

void Callback::token(const std::string& text) {
    if (!on_token || text.empty()) return;
    jstring js = to_jstring(env, text);
    if (js) {
        env->CallVoidMethod(obj, on_token, js);
        env->DeleteLocalRef(js);
    }
}

void Callback::tool_call(const std::string& name, const std::string& args) {
    if (!on_tool_call) return;
    jstring jn = env->NewStringUTF(name.c_str());
    jstring ja = to_jstring(env, args);
    env->CallVoidMethod(obj, on_tool_call, jn, ja);
    env->DeleteLocalRef(jn);
    if (ja) env->DeleteLocalRef(ja);
}

void Callback::done() {
    if (on_done) env->CallVoidMethod(obj, on_done);
}

void Callback::error(const std::string& msg) {
    if (!on_error) return;
    jstring js = env->NewStringUTF(msg.c_str());
    env->CallVoidMethod(obj, on_error, js);
    env->DeleteLocalRef(js);
}

void Callback::metrics(float tps, float ttft_ms, float total_ms,
                       int n_eval, int n_pred,
                       float model_mb, float ctx_mb, float peak_mb, float mem_pct) {
    if (!on_metrics) return;
    env->CallVoidMethod(obj, on_metrics, tps, ttft_ms, total_ms,
                        n_eval, n_pred, model_mb, ctx_mb, peak_mb, mem_pct);
}

} // namespace jni
