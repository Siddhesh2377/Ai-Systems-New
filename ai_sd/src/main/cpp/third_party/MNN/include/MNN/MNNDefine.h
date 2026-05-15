//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

#ifdef MNN_USE_LOGCAT
#if defined(__OHOS__)
#include <hilog/log.h>
#define MNN_ERROR(format, ...) {char logtmp[4096]; snprintf(logtmp, 4096, format, ##__VA_ARGS__); OH_LOG_Print(LOG_APP, LOG_ERROR, LOG_DOMAIN, "MNNJNI", (const char*)logtmp);}
#define MNN_PRINT(format, ...) {char logtmp[4096]; snprintf(logtmp, 4096, format, ##__VA_ARGS__); OH_LOG_Print(LOG_APP, LOG_DEBUG, LOG_DOMAIN, "MNNJNI", (const char*)logtmp);}
#else
// Route MNN's internal log macros through tn_security so they get attributed
// to TN_MODULE_MNN (30) instead of disappearing into logcat unstructured. The
// __has_include guard preserves the original android_log_print path if the
// header isn't visible (e.g. MNN built standalone, outside ai_sd's CMake).
#if __has_include(<tn_security/tn_security.h>)
#include <tn_security/tn_security.h>
#define MNN_ERROR(format, ...) tn_sec_log(TN_LEVEL_ERROR, TN_MODULE_MNN, "MNNJNI", \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) tn_sec_log(TN_LEVEL_INFO,  TN_MODULE_MNN, "MNNJNI", \
    tn_sec_current_op(), __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#else
#include <android/log.h>
#define MNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#endif
#endif
#elif defined MNN_BUILD_FOR_IOS
// on iOS, stderr prints to XCode debug area and syslog prints Console. You need both.
#include <syslog.h>
#define MNN_PRINT(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#else
#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define MNN_ASSERT(x)
#endif

#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define MNN_CHECK(success, log) \
if(!(success)){ \
MNN_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

#if defined(_MSC_VER)
#if defined(BUILDING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllexport)
#elif defined(USING_MNN_DLL)
#define MNN_PUBLIC __declspec(dllimport)
#else
#define MNN_PUBLIC
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif

#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define MNN_VERSION_MAJOR 3
#define MNN_VERSION_MINOR 2
#define MNN_VERSION_PATCH 1
#define MNN_VERSION STR(MNN_VERSION_MAJOR) "." STR(MNN_VERSION_MINOR) "." STR(MNN_VERSION_PATCH)
#endif /* MNNDefine_h */
