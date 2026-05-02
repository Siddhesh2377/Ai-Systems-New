cmake_minimum_required(VERSION 3.22.1)
project("ai_sherpa")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARM release optimizations
if(ANDROID_ABI STREQUAL "arm64-v8a")
  add_compile_options(-march=armv8-a+simd -O3 -ffast-math -fno-finite-math-only)
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
  add_compile_options(-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp -O3 -ffast-math -fno-finite-math-only)
endif()

# LTO for release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(SHERPA_ONNX_DIR /home/home/dev/include/sherpa-onnx)
set(ORT_DIR /home/home/dev/include/ort-android-1.24.3)

# Point sherpa-onnx at the prebuilt ORT Android .so
set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "${ORT_DIR}/headers")
set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR} "${ORT_DIR}/${ANDROID_ABI}")
set(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ON CACHE BOOL "" FORCE)

# sherpa-onnx build options
set(SHERPA_ONNX_ENABLE_JNI OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${SHERPA_ONNX_DIR} ${CMAKE_BINARY_DIR}/sherpa-onnx)

add_library(ai_sherpa SHARED
  jni_cache.cpp
  online_recognizer.cpp
  offline_recognizer.cpp
  offline_tts.cpp
  vad.cpp
  wave_io.cpp
  error_tracker.cpp
  error_jni.cpp
)

target_include_directories(ai_sherpa PRIVATE
  ${SHERPA_ONNX_DIR}
  ${ORT_DIR}/headers
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ai_sherpa PRIVATE
  sherpa-onnx-c-api
  sherpa-onnx-core
  android
  log
)

target_link_options(ai_sherpa PRIVATE
  "-Wl,-z,max-page-size=16384"
  "-Wl,-z,common-page-size=16384"
)
