#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TN_ERR_NONE             = 0,
    TN_ERR_UNKNOWN          = 1,
    TN_ERR_OOM              = 2,
    TN_ERR_INVALID_PARAM    = 3,
    TN_ERR_MODEL_LOAD       = 4,
    TN_ERR_MODEL_INCOMPAT   = 5,
    TN_ERR_TEMPLATE         = 6,
    TN_ERR_GENERATION       = 7,
    TN_ERR_IO               = 8,
    TN_ERR_NATIVE_CRASH     = 9,
};

void tn_error_init(void);
void tn_error_set_crash_log_path(const char* path);

void tn_error_set_op(const char* op, const char* detail);
void tn_error_clear_op(void);

void tn_error_set_last(int code, const char* category, const char* message);
void tn_error_clear_last(void);

const char* tn_error_get_last_json(void);

const char* tn_error_signal_name(int sig);

#ifdef __cplusplus
}
#endif
