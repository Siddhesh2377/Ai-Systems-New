// JNI bridge — Kotlin facade ↔ tn_security C core.
//
// The native sink (set once at init) forwards every event to a static method
// on TnSecurity. Kotlin then fans events out to all registered TnSink
// listeners.

#include <tn_security/tn_security.h>

#include <jni.h>
#include <csignal>
#include <cstring>
#include <android/log.h>

namespace {

JavaVM*  g_vm                = nullptr;
jclass   g_cls_TnSecurity    = nullptr;   // global ref
jmethodID g_mid_onNativeEvent = nullptr;

// java.lang.String + ctor for lenient byte->String decoding. NewStringUTF
// requires *Modified* UTF-8 with strict validation: invalid byte sequences,
// truncated multi-byte chars, or 4-byte UTF-8 sequences abort the process.
// Upstream logs (llama.cpp dumping tokenizer merges with truncated multi-byte
// chars) routinely produce such bytes. Going through String(byte[], "UTF-8")
// uses java.nio.charset.CharsetDecoder which replaces invalid sequences with
// U+FFFD instead of aborting.
jclass    g_cls_String       = nullptr;
jmethodID g_ctor_String_bytes_charset = nullptr;
jstring   g_charset_utf8     = nullptr;   // global ref to "UTF-8" string

const char* kJavaClass = "com/dark/tn_security/TnSecurity";

// Signature mirrors the C sink fn:
//   onNativeEvent(int kind, int level, int module, int code, int stage,
//                 String tag, String opId, String file, int line, String func,
//                 String message, String suggestion, long timestampMs, int tid)
const char* kSig =
    "(IIIIILjava/lang/String;Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;"
    "Ljava/lang/String;Ljava/lang/String;JI)V";

struct AttachScope {
    JNIEnv* env       = nullptr;
    bool    attached  = false;
    AttachScope() {
        if (!g_vm) return;
        if (g_vm->GetEnv((void**)&env, JNI_VERSION_1_6) == JNI_OK) return;
        if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) attached = true;
    }
    ~AttachScope() {
        if (attached && g_vm) g_vm->DetachCurrentThread();
    }
};

// Build a jstring from a possibly-malformed UTF-8 byte sequence. Bypasses
// NewStringUTF's strict Modified UTF-8 validator by going through
// String(byte[], Charset) which is lenient. Returns nullptr if `s` is null
// or any allocation fails; caller must free the local ref.
jstring to_jstring(JNIEnv* env, const char* s) {
    if (!s) return nullptr;
    if (!g_cls_String || !g_ctor_String_bytes_charset || !g_charset_utf8) {
        // Init failed somehow — fall back to NewStringUTF and pray.
        return env->NewStringUTF(s);
    }
    const jsize len = (jsize)::strlen(s);
    jbyteArray arr = env->NewByteArray(len);
    if (!arr) {
        if (env->ExceptionCheck()) env->ExceptionClear();
        return nullptr;
    }
    if (len > 0) {
        env->SetByteArrayRegion(arr, 0, len, reinterpret_cast<const jbyte*>(s));
    }
    jstring js = (jstring)env->NewObject(
        g_cls_String, g_ctor_String_bytes_charset, arr, g_charset_utf8);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        js = nullptr;
    }
    env->DeleteLocalRef(arr);
    return js;
}

void native_sink(int kind, int level, int module, int code, int stage,
                 const char* tag, const char* op_id, const char* file,
                 int line, const char* func, const char* message,
                 const char* suggestion, int64_t timestamp_ms, int32_t tid,
                 void* /*user*/) {
    AttachScope sc;
    if (!sc.env || !g_cls_TnSecurity || !g_mid_onNativeEvent) return;
    JNIEnv* env = sc.env;

    jstring j_tag        = to_jstring(env, tag);
    jstring j_op_id      = to_jstring(env, op_id);
    jstring j_file       = to_jstring(env, file);
    jstring j_func       = to_jstring(env, func);
    jstring j_message    = to_jstring(env, message);
    jstring j_suggestion = to_jstring(env, suggestion);

    env->CallStaticVoidMethod(g_cls_TnSecurity, g_mid_onNativeEvent,
                              (jint)kind, (jint)level, (jint)module,
                              (jint)code, (jint)stage,
                              j_tag, j_op_id, j_file, (jint)line, j_func,
                              j_message, j_suggestion,
                              (jlong)timestamp_ms, (jint)tid);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }

    if (j_tag)        env->DeleteLocalRef(j_tag);
    if (j_op_id)      env->DeleteLocalRef(j_op_id);
    if (j_file)       env->DeleteLocalRef(j_file);
    if (j_func)       env->DeleteLocalRef(j_func);
    if (j_message)    env->DeleteLocalRef(j_message);
    if (j_suggestion) env->DeleteLocalRef(j_suggestion);
}

} // namespace


extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_vm = vm;
    JNIEnv* env = nullptr;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    jclass local = env->FindClass(kJavaClass);
    if (!local) return JNI_ERR;
    g_cls_TnSecurity = (jclass)env->NewGlobalRef(local);
    env->DeleteLocalRef(local);

    g_mid_onNativeEvent = env->GetStaticMethodID(g_cls_TnSecurity, "onNativeEvent", kSig);
    if (!g_mid_onNativeEvent) return JNI_ERR;

    jclass strLocal = env->FindClass("java/lang/String");
    if (strLocal) {
        g_cls_String = (jclass)env->NewGlobalRef(strLocal);
        env->DeleteLocalRef(strLocal);
        g_ctor_String_bytes_charset =
            env->GetMethodID(g_cls_String, "<init>", "([BLjava/lang/String;)V");
        jstring localUtf8 = env->NewStringUTF("UTF-8");
        if (localUtf8) {
            g_charset_utf8 = (jstring)env->NewGlobalRef(localUtf8);
            env->DeleteLocalRef(localUtf8);
        }
    }
    if (env->ExceptionCheck()) env->ExceptionClear();

    return JNI_VERSION_1_6;
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeInit(JNIEnv*, jobject) {
    tn_sec_init();
    tn_sec_set_sink(native_sink, nullptr);
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeShutdown(JNIEnv*, jobject) {
    tn_sec_set_sink(nullptr, nullptr);
    tn_sec_shutdown();
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeSetCrashFilePattern(
        JNIEnv* env, jobject, jstring jpattern) {
    if (!jpattern) { tn_sec_set_crash_file_pattern(nullptr); return; }
    const char* s = env->GetStringUTFChars(jpattern, nullptr);
    tn_sec_set_crash_file_pattern(s);
    env->ReleaseStringUTFChars(jpattern, s);
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeInstallSignalHandlers(JNIEnv*, jobject) {
    tn_sec_install_signal_handlers();
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeSetOp(JNIEnv* env, jobject, jstring jop) {
    if (!jop) { tn_sec_set_op(nullptr); return; }
    const char* s = env->GetStringUTFChars(jop, nullptr);
    tn_sec_set_op(s);
    env->ReleaseStringUTFChars(jop, s);
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeClearOp(JNIEnv*, jobject) {
    tn_sec_clear_op();
}

JNIEXPORT jstring JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeCurrentOp(JNIEnv* env, jobject) {
    const char* s = tn_sec_current_op();
    return s ? env->NewStringUTF(s) : nullptr;
}

// Kotlin-side log/error emission — flows through the same pipeline (ring +
// sink). This makes Kotlin Log.d/e/i/w replacements visible to crash dumps.

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeLog(
        JNIEnv* env, jobject, jint level, jint module,
        jstring jtag, jstring jop, jstring jfile, jint line, jstring jfunc,
        jstring jmsg) {
    const char* tag  = jtag  ? env->GetStringUTFChars(jtag,  nullptr) : nullptr;
    const char* op   = jop   ? env->GetStringUTFChars(jop,   nullptr) : nullptr;
    const char* file = jfile ? env->GetStringUTFChars(jfile, nullptr) : nullptr;
    const char* func = jfunc ? env->GetStringUTFChars(jfunc, nullptr) : nullptr;
    const char* msg  = jmsg  ? env->GetStringUTFChars(jmsg,  nullptr) : "";

    tn_sec_log((tn_level)level, (tn_module)module, tag, op, file, line, func, "%s", msg);

    if (tag)  env->ReleaseStringUTFChars(jtag,  tag);
    if (op)   env->ReleaseStringUTFChars(jop,   op);
    if (file) env->ReleaseStringUTFChars(jfile, file);
    if (func) env->ReleaseStringUTFChars(jfunc, func);
    if (jmsg) env->ReleaseStringUTFChars(jmsg,  msg);
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeEmitError(
        JNIEnv* env, jobject, jint module, jint code, jint stage,
        jstring jop, jstring jfile, jint line, jstring jfunc,
        jstring jsuggestion, jstring jmsg) {
    const char* op         = jop         ? env->GetStringUTFChars(jop,         nullptr) : nullptr;
    const char* file       = jfile       ? env->GetStringUTFChars(jfile,       nullptr) : nullptr;
    const char* func       = jfunc       ? env->GetStringUTFChars(jfunc,       nullptr) : nullptr;
    const char* suggestion = jsuggestion ? env->GetStringUTFChars(jsuggestion, nullptr) : nullptr;
    const char* msg        = jmsg        ? env->GetStringUTFChars(jmsg,        nullptr) : "";

    tn_error_init init = {
        (tn_module)module, (tn_code)code, (tn_stage)stage,
        op, file, line, func, suggestion,
    };
    tn_sec_emit_error(&init, "%s", msg);

    if (op)         env->ReleaseStringUTFChars(jop,         op);
    if (file)       env->ReleaseStringUTFChars(jfile,       file);
    if (func)       env->ReleaseStringUTFChars(jfunc,       func);
    if (suggestion) env->ReleaseStringUTFChars(jsuggestion, suggestion);
    if (jmsg)       env->ReleaseStringUTFChars(jmsg,        msg);
}

JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeEmitCancellation(
        JNIEnv* env, jobject, jint module, jstring jop, jstring jreason) {
    const char* op     = jop     ? env->GetStringUTFChars(jop,     nullptr) : nullptr;
    const char* reason = jreason ? env->GetStringUTFChars(jreason, nullptr) : nullptr;

    tn_sec_emit_cancellation((tn_module)module, op, reason);

    if (op)     env->ReleaseStringUTFChars(jop,     op);
    if (reason) env->ReleaseStringUTFChars(jreason, reason);
}

JNIEXPORT jstring JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeModuleSlug(JNIEnv* env, jobject, jint module) {
    return env->NewStringUTF(tn_sec_module_slug((tn_module)module));
}

JNIEXPORT jstring JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeSignalName(JNIEnv* env, jobject, jint sig) {
    return env->NewStringUTF(tn_sec_signal_name(sig));
}

JNIEXPORT jint JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeApiVersion(JNIEnv*, jobject) {
    return TN_SECURITY_API_VERSION;
}

// Test endpoint — used by instrumentation tests in Phase 8 to deliberately
// raise a signal and verify crash-file capture. Hidden from public Kotlin
// surface; called only via reflection in tests.
JNIEXPORT void JNICALL
Java_com_dark_tn_1security_TnSecurity_nativeRaiseSignalForTest(
        JNIEnv*, jobject, jint sig) {
    raise(sig);
}

} // extern "C"
