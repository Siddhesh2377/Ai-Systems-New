cmake_minimum_required(VERSION 3.22.1)
project("ai_sherpa")

set(SHERPA_ONNX_DIR /home/home/dev/include/sherpa-onnx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID_ABI STREQUAL "arm64-v8a")
  add_compile_options(-march=armv8-a -O3)
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
  add_compile_options(-march=armv7-a -mfpu=neon -O3)
endif()

add_library(ai_sherpa SHARED
  jni_cache.cpp
  online_recognizer.cpp
  offline_recognizer.cpp
  offline_tts.cpp
  vad.cpp
  wave_io.cpp
)

target_include_directories(ai_sherpa PRIVATE
  ${SHERPA_ONNX_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ai_sherpa
  sherpa-onnx-core
  android
  log
)
