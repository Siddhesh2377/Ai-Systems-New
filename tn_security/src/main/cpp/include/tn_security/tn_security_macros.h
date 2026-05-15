// tn_security_macros.h — call-site macros for logging and emitting errors.
//
// Each .cpp file SHOULD define TN_MODULE before including this header:
//
//     #define TN_MODULE TN_MODULE_GGUF_LIB
//     #include <tn_security/tn_security_macros.h>
//
// Optionally also TN_TAG for a default tag:
//
//     #define TN_TAG "gguf"
//
// Then in code:
//
//     TN_I("loaded model: %s", path);
//     TN_E("decode failed: rc=%d", rc);
//     TN_ERR(TN_CODE_DECODE_FAIL, TN_STAGE_DECODE, "llama_decode rc=%d", rc);

#ifndef TN_SECURITY_MACROS_H
#define TN_SECURITY_MACROS_H

#include <tn_security/tn_security.h>

#ifdef __cplusplus
#include <cstddef>
#endif

#ifndef TN_MODULE
#define TN_MODULE TN_MODULE_UNKNOWN
#endif

#ifndef TN_TAG
#define TN_TAG nullptr
#endif

// ─── Logging ────────────────────────────────────────────────────────────────

#define TN_LOG(level, tag, fmt, ...) \
    tn_sec_log((level), (TN_MODULE), (tag), tn_sec_current_op(), \
               __FILE__, __LINE__, __func__, (fmt), ##__VA_ARGS__)

#define TN_T(fmt, ...) TN_LOG(TN_LEVEL_TRACE, TN_TAG, fmt, ##__VA_ARGS__)
#define TN_D(fmt, ...) TN_LOG(TN_LEVEL_DEBUG, TN_TAG, fmt, ##__VA_ARGS__)
#define TN_I(fmt, ...) TN_LOG(TN_LEVEL_INFO,  TN_TAG, fmt, ##__VA_ARGS__)
#define TN_W(fmt, ...) TN_LOG(TN_LEVEL_WARN,  TN_TAG, fmt, ##__VA_ARGS__)
#define TN_E(fmt, ...) TN_LOG(TN_LEVEL_ERROR, TN_TAG, fmt, ##__VA_ARGS__)
#define TN_F(fmt, ...) TN_LOG(TN_LEVEL_FATAL, TN_TAG, fmt, ##__VA_ARGS__)

// ─── Errors ─────────────────────────────────────────────────────────────────

#define TN_ERR(code, stage, fmt, ...) do {                                \
    tn_error_init _tn_ei = { (TN_MODULE), (code), (stage),                \
                              tn_sec_current_op(),                         \
                              __FILE__, __LINE__, __func__, nullptr };     \
    tn_sec_emit_error(&_tn_ei, (fmt), ##__VA_ARGS__);                      \
} while (0)

#define TN_ERR_FIX(code, stage, suggestion, fmt, ...) do {                \
    tn_error_init _tn_ei = { (TN_MODULE), (code), (stage),                \
                              tn_sec_current_op(),                         \
                              __FILE__, __LINE__, __func__, (suggestion) };\
    tn_sec_emit_error(&_tn_ei, (fmt), ##__VA_ARGS__);                      \
} while (0)

// ─── Cancellation ───────────────────────────────────────────────────────────

#define TN_CANCEL(reason) \
    tn_sec_emit_cancellation((TN_MODULE), tn_sec_current_op(), (reason))

// ─── Op scope (RAII in C++) ─────────────────────────────────────────────────
// Use in C++ to automatically clear an op-id when a scope exits:
//
//     {
//         TN_OP_SCOPE("generate-42");
//         // ... any errors here are stamped with op_id "generate-42"
//     }   // automatically cleared

#ifdef __cplusplus
namespace tn_security_detail {
struct TnOpScope {
    bool prev_was_set;
    char prev[128];
    explicit TnOpScope(const char* op) {
        const char* p = tn_sec_current_op();
        prev_was_set = (p != nullptr);
        if (prev_was_set) {
            size_t n = 0;
            while (p[n] && n < sizeof(prev) - 1) { prev[n] = p[n]; ++n; }
            prev[n] = '\0';
        }
        tn_sec_set_op(op);
    }
    ~TnOpScope() {
        if (prev_was_set) tn_sec_set_op(prev);
        else              tn_sec_clear_op();
    }
};
} // namespace tn_security_detail

#define TN_OP_SCOPE_CONCAT_(a, b) a##b
#define TN_OP_SCOPE_CONCAT(a, b)  TN_OP_SCOPE_CONCAT_(a, b)
#define TN_OP_SCOPE(op_id) \
    ::tn_security_detail::TnOpScope TN_OP_SCOPE_CONCAT(_tn_op_, __LINE__)(op_id)
#endif

#endif // TN_SECURITY_MACROS_H
