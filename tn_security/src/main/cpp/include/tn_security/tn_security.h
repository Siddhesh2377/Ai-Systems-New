// tn_security — unified diagnostic + error capture for all on-device SDKs.
//
// One process, one sink fan-out, one event format. Every log line and every
// error from every native lib (llama.cpp / ggml / sherpa-onnx / MNN) and every
// SDK (gguf_lib / ai_sherpa / ai_sd) and every Kotlin layer flows through this
// file's API.
//
// Numeric enum values are STABLE across versions. Add new values at the end of
// the enum; never renumber. The Kotlin side mirrors these by value.

#ifndef TN_SECURITY_H
#define TN_SECURITY_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TN_SECURITY_API_VERSION 1

// Default-visibility export annotation. The tn_security .so is compiled with
// -fvisibility=hidden so internals stay private; every public symbol below
// is explicitly re-exported via TN_API.
#if !defined(TN_API)
#  if defined(_WIN32)
#    define TN_API __declspec(dllexport)
#  else
#    define TN_API __attribute__((visibility("default")))
#  endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Enums (stable)
// ─────────────────────────────────────────────────────────────────────────────

typedef enum tn_level {
    TN_LEVEL_TRACE = 0,
    TN_LEVEL_DEBUG = 1,
    TN_LEVEL_INFO  = 2,
    TN_LEVEL_WARN  = 3,
    TN_LEVEL_ERROR = 4,
    TN_LEVEL_FATAL = 5,
} tn_level;

typedef enum tn_module {
    TN_MODULE_UNKNOWN     = 0,
    TN_MODULE_TN_SECURITY = 1,
    // upstream native libs
    TN_MODULE_LLAMA_CPP   = 10,
    TN_MODULE_GGML        = 11,
    TN_MODULE_SHERPA_ONNX = 20,
    TN_MODULE_ONNX_RT     = 21,
    TN_MODULE_MNN         = 30,
    TN_MODULE_QNN         = 31,
    // our SDKs
    TN_MODULE_GGUF_LIB    = 100,
    TN_MODULE_AI_SHERPA   = 101,
    TN_MODULE_AI_SD       = 102,
    // tool-neuron host
    TN_MODULE_TN_SERVICE  = 200,
    TN_MODULE_TN_APP      = 201,
    TN_MODULE_TN_PLUGIN   = 202,
    TN_MODULE_TN_HXS      = 203,
} tn_module;

typedef enum tn_code {
    TN_CODE_OK                      = 0,
    TN_CODE_UNKNOWN                 = 1,
    TN_CODE_CANCELLED               = 2,
    TN_CODE_INVALID_PARAM           = 3,
    TN_CODE_NOT_READY               = 4,
    // resource (100–199)
    TN_CODE_OOM                     = 100,
    TN_CODE_DISK_FULL               = 101,
    TN_CODE_RESOURCE_EXHAUSTED      = 102,
    TN_CODE_THREAD_POOL_FULL        = 103,
    // io (200–299)
    TN_CODE_IO_FAIL                 = 200,
    TN_CODE_FILE_NOT_FOUND          = 201,
    TN_CODE_FILE_CORRUPT            = 202,
    TN_CODE_PERMISSION_DENIED       = 203,
    TN_CODE_NETWORK_FAIL            = 204,
    // model (300–399)
    TN_CODE_MODEL_LOAD_FAIL         = 300,
    TN_CODE_MODEL_ARCH_UNSUPPORTED  = 301,
    TN_CODE_MODEL_TEMPLATE_INVALID  = 302,
    TN_CODE_CONTEXT_OVERFLOW        = 303,
    TN_CODE_MMAP_FAIL               = 304,
    TN_CODE_QUANT_UNSUPPORTED       = 305,
    // inference (400–499)
    TN_CODE_DECODE_FAIL             = 400,
    TN_CODE_TOKENIZE_FAIL           = 401,
    TN_CODE_SAMPLE_FAIL             = 402,
    TN_CODE_PROJECTOR_MISMATCH      = 403,
    TN_CODE_KV_CACHE_FAIL           = 404,
    TN_CODE_GRAPH_BUILD_FAIL        = 405,
    // backend / hardware (500–599)
    TN_CODE_BACKEND_INIT_FAIL       = 500,
    TN_CODE_QNN_HTP_UNAVAILABLE     = 501,
    TN_CODE_SOC_INCOMPATIBLE        = 502,
    TN_CODE_GPU_UNAVAILABLE         = 503,
    TN_CODE_MNN_INIT_FAIL           = 504,
    TN_CODE_ZSTD_PATCH_FAIL         = 505,
    // ipc / service (600–699)
    TN_CODE_AIDL_DEAD_OBJECT        = 600,
    TN_CODE_AIDL_TIMEOUT            = 601,
    TN_CODE_AIDL_TRANSACTION_LARGE  = 602,
    TN_CODE_SERVICE_BIND_FAIL       = 603,
    // plugin (700–799)
    TN_CODE_PLUGIN_API_MISMATCH     = 700,
    TN_CODE_PLUGIN_CLASS_NOT_FOUND  = 701,
    TN_CODE_PLUGIN_INIT_FAIL        = 702,
    TN_CODE_PLUGIN_EXEC_FAIL        = 703,
    // native crash (900–999)
    TN_CODE_NATIVE_CRASH            = 900,
    TN_CODE_NATIVE_ABORT            = 901,
} tn_code;

typedef enum tn_stage {
    TN_STAGE_UNSPECIFIED    = 0,
    TN_STAGE_INIT           = 10,
    TN_STAGE_LOAD           = 20,
    TN_STAGE_WARMUP         = 21,
    // text generation
    TN_STAGE_TOKENIZE       = 30,
    TN_STAGE_PROMPT_EVAL    = 40,
    TN_STAGE_DECODE         = 41,
    TN_STAGE_SAMPLE         = 42,
    TN_STAGE_DETOKENIZE     = 43,
    // vlm
    TN_STAGE_VLM_PROJECT    = 50,
    TN_STAGE_VLM_DECODE_IMG = 51,
    TN_STAGE_VLM_TOKENIZE   = 52,
    // speech
    TN_STAGE_STT_DECODE     = 60,
    TN_STAGE_TTS_GENERATE   = 61,
    TN_STAGE_AUDIO_ACCEPT   = 62,
    // diffusion
    TN_STAGE_SD_UNET        = 70,
    TN_STAGE_SD_CLIP        = 71,
    TN_STAGE_SD_VAE         = 72,
    TN_STAGE_SD_SCHEDULER   = 73,
    TN_STAGE_SD_UPSCALE     = 74,
    TN_STAGE_SD_SEGMENT     = 75,
    TN_STAGE_SD_INPAINT     = 76,
    TN_STAGE_SD_DEPTH       = 77,
    TN_STAGE_SD_STYLE       = 78,
    // rag
    TN_STAGE_RAG_INGEST     = 80,
    TN_STAGE_RAG_EMBED      = 81,
    TN_STAGE_RAG_QUERY      = 82,
    // plugin
    TN_STAGE_PLUGIN_LOAD    = 90,
    TN_STAGE_PLUGIN_EXEC    = 91,
    // runtime setup / asset
    TN_STAGE_ASSET_COPY     = 100,
    TN_STAGE_ASSET_EXTRACT  = 101,
    TN_STAGE_ASSET_PATCH    = 102,
} tn_stage;

// ─────────────────────────────────────────────────────────────────────────────
//  Sink — single C function pointer. The JNI bridge sets this to forward
//  events to all registered Kotlin sinks. Set once at init.
// ─────────────────────────────────────────────────────────────────────────────

// kind: 0=log, 1=error, 2=cancellation, 3=crash (signal handler synthesizes
// crash events out-of-band and writes them to a file — the sink will see them
// when the file is drained next session).
typedef void (*tn_sec_sink_fn)(
    int         kind,
    int         level,        // tn_level (for logs) or 0
    int         module,       // tn_module
    int         code,         // tn_code (for errors) or 0
    int         stage,        // tn_stage (for errors) or 0
    const char* tag,          // may be NULL
    const char* op_id,        // may be NULL
    const char* file,         // call-site source file, may be NULL
    int         line,         // call-site line, 0 if unknown
    const char* func,         // call-site function, may be NULL
    const char* message,      // formatted message
    const char* suggestion,   // user-actionable fix hint, may be NULL
    int64_t     timestamp_ms, // wall-clock
    int32_t     tid,          // gettid()
    void*       user_data);

TN_API void tn_sec_set_sink(tn_sec_sink_fn fn, void* user_data);

// ─────────────────────────────────────────────────────────────────────────────
//  Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

TN_API void tn_sec_init(void);
TN_API void tn_sec_shutdown(void);

// Pattern uses %m (module name slug), %p (pid), %t (epoch ms). E.g.
//   "/data/data/.../files/tn_security/crash_%m_%p_%t.json"
// Pass NULL to disable.
TN_API void tn_sec_set_crash_file_pattern(const char* pattern);

// Install signal handlers (SIGSEGV/SIGABRT/SIGBUS/SIGILL/SIGFPE) that write
// crash JSON to the configured pattern, then re-raise so the kernel tombstone
// still happens. Safe to call multiple times; subsequent calls no-op.
TN_API void tn_sec_install_signal_handlers(void);

// ─────────────────────────────────────────────────────────────────────────────
//  Op tracking (thread-local). Any error/log emitted while an op is active
//  gets the op_id stamped. Set at the entry of a "user operation" (a generate
//  call, a model load, etc.) and clear at exit.
// ─────────────────────────────────────────────────────────────────────────────

TN_API void        tn_sec_set_op(const char* op_id);
TN_API void        tn_sec_clear_op(void);
TN_API const char* tn_sec_current_op(void);   // returns NULL or thread-local

// ─────────────────────────────────────────────────────────────────────────────
//  Logging
// ─────────────────────────────────────────────────────────────────────────────

TN_API void tn_sec_log(tn_level   level,
                tn_module  module,
                const char* tag,
                const char* op_id,
                const char* file,
                int         line,
                const char* func,
                const char* fmt, ...) __attribute__((format(printf, 8, 9)));

TN_API void tn_sec_log_v(tn_level   level,
                  tn_module  module,
                  const char* tag,
                  const char* op_id,
                  const char* file,
                  int         line,
                  const char* func,
                  const char* fmt,
                  va_list     args);

// ─────────────────────────────────────────────────────────────────────────────
//  Errors
// ─────────────────────────────────────────────────────────────────────────────

typedef struct tn_error_init {
    tn_module   module;
    tn_code     code;
    tn_stage    stage;
    const char* op_id;      // may be NULL → uses thread-local
    const char* file;       // may be NULL
    int         line;
    const char* func;       // may be NULL
    const char* suggestion; // user-actionable, may be NULL
} tn_error_init;

TN_API void tn_sec_emit_error(const tn_error_init* init,
                       const char*          fmt, ...)
    __attribute__((format(printf, 2, 3)));

// ─────────────────────────────────────────────────────────────────────────────
//  Cancellation — distinct from errors; UI filters these out of "errors" feed.
// ─────────────────────────────────────────────────────────────────────────────

TN_API void tn_sec_emit_cancellation(tn_module   module,
                              const char* op_id,
                              const char* reason);

// ─────────────────────────────────────────────────────────────────────────────
//  Module name slug — for crash file naming. e.g. TN_MODULE_GGUF_LIB → "gguf_lib"
// ─────────────────────────────────────────────────────────────────────────────

TN_API const char* tn_sec_module_slug(tn_module module);

// ─────────────────────────────────────────────────────────────────────────────
//  Signal name — for crash logging. e.g. SIGSEGV → "SIGSEGV"
// ─────────────────────────────────────────────────────────────────────────────

TN_API const char* tn_sec_signal_name(int sig);

#ifdef __cplusplus
}
#endif

#endif // TN_SECURITY_H
