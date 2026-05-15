cmake_minimum_required(VERSION 3.22.1)
project("ai_sherpa")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Avoid -ffast-math: ONNX Runtime expects strict IEEE-754 NaN propagation
# in some operators (notably softmax masking with -inf), which fast-math
# breaks. -O3 + NEON is enough for sherpa-onnx's call sites.
if(ANDROID_ABI STREQUAL "arm64-v8a")
  add_compile_options(-march=armv8-a+simd -O3)
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
  add_compile_options(-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -O3)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(SHERPA_ONNX_DIR /home/home/dev/include/sherpa-onnx)
set(ORT_DIR         /home/home/dev/include/ort-android-1.24.3)

# sherpa-onnx looks for these env vars to skip its own ORT build.
set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "${ORT_DIR}/headers")
set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR}     "${ORT_DIR}/${ANDROID_ABI}")
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)

set(SHERPA_ONNX_ENABLE_JNI                  OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION  OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_BINARY               OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_C_API                ON  CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TTS                  ON  CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PYTHON               OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PORTAUDIO            OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_WEBSOCKET            OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TESTS                OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS                       OFF CACHE BOOL "" FORCE)

find_package(tn_security REQUIRED CONFIG)

# Propagate tn_security headers into the sherpa-onnx subbuild — the upstream
# fork's csrc/macros.h now includes <tn_security/tn_security.h> so its
# SHERPA_ONNX_LOGE macro can route into the unified sink. Unresolved
# tn_sec_log symbols in the static archive are satisfied when libai_sherpa.so
# links against tn_security::tn_security below.
get_target_property(_tn_sec_inc tn_security::tn_security INTERFACE_INCLUDE_DIRECTORIES)
if(_tn_sec_inc)
  include_directories(${_tn_sec_inc})
endif()

add_subdirectory(${SHERPA_ONNX_DIR} ${CMAKE_BINARY_DIR}/sherpa-onnx)

add_library(ai_sherpa SHARED
  jni_cache.cpp
  offline_recognizer.cpp
  offline_tts.cpp
)

target_include_directories(ai_sherpa PRIVATE
  ${SHERPA_ONNX_DIR}
  ${ORT_DIR}/headers
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ai_sherpa PRIVATE
  sherpa-onnx-c-api
  sherpa-onnx-core
  tn_security::tn_security
  android
  log
)

# 16KB page size required by Android 15+ on devices with that page size
# (e.g. Pixel 8a). Future Android targets will reject .so files without it.
target_link_options(ai_sherpa PRIVATE
  "-Wl,-z,max-page-size=16384"
  "-Wl,-z,common-page-size=16384"
)
