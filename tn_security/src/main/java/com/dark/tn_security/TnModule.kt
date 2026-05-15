package com.dark.tn_security

/**
 * Source module of a log/error/crash. Numeric values mirror `tn_module` in
 * tn_security.h — never renumber, only append. The string `slug` matches the
 * C-side `tn_sec_module_slug()` and is used in crash filenames.
 */
enum class TnModule(val value: Int, val slug: String) {
    UNKNOWN     (0,   "unknown"),
    TN_SECURITY (1,   "tn_security"),

    LLAMA_CPP   (10,  "llama.cpp"),
    GGML        (11,  "ggml"),
    SHERPA_ONNX (20,  "sherpa-onnx"),
    ONNX_RT     (21,  "onnxruntime"),
    MNN         (30,  "MNN"),
    QNN         (31,  "QNN"),

    GGUF_LIB    (100, "gguf_lib"),
    AI_SHERPA   (101, "ai_sherpa"),
    AI_SD       (102, "ai_sd"),

    TN_SERVICE  (200, "tn_service"),
    TN_APP      (201, "tn_app"),
    TN_PLUGIN   (202, "tn_plugin"),
    TN_HXS      (203, "tn_hxs");

    companion object {
        fun fromInt(v: Int): TnModule = entries.firstOrNull { it.value == v } ?: UNKNOWN
    }
}
