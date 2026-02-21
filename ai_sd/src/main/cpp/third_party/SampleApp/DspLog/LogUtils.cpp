//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "HAP_farf.h"
#include "LogUtils.hpp"

#define PRINTLEN 1024

void qnn::log::utils::logDefaultCallback(const char* fmt,
                                         QnnLog_Level_t level,
                                         uint64_t timestamp,
                                         va_list argp) {
  char buffer[PRINTLEN] = "";
  std::lock_guard<std::mutex> lock(sg_logUtilMutex);
  vsnprintf(buffer, sizeof(buffer), fmt, argp);
  FARF(ALWAYS, "[%x] %s", level, buffer);
}