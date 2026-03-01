cmake_minimum_required(VERSION 3.22.1)
project(ai_gguf LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMACPP_DIR /home/home/dev/lib/llama.cpp-custom)
if(NOT EXISTS "${LLAMACPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Could not find llama.cpp in ${LLAMACPP_DIR}")
endif()

set(GGML_PAGE_SIZE 16384 CACHE STRING "GGML page size for Android 16KB support")
add_compile_definitions(GGML_PAGE_SIZE=${GGML_PAGE_SIZE})

set(LLAMA_MOBILE ON CACHE BOOL "Enable mobile optimizations")
add_compile_definitions(LLAMA_MOBILE)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

if(ANDROID)
    set(OpenCL_INCLUDE_DIR "${CMAKE_SYSROOT}/usr/include" CACHE PATH "OpenCL headers")
    set(OpenCL_LIBRARY "${CMAKE_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/libOpenCL.so" CACHE FILEPATH "OpenCL ICD loader")
endif()

add_subdirectory(${LLAMACPP_DIR} llama-build)

set(SRC_FILES
    src/ai_gguf.cpp
    src/utils/jni_utils.cpp
    src/cpu/cpu_helper.cpp
)

include_directories(${LLAMACPP_DIR})
include_directories(${LLAMACPP_DIR}/include)

add_library(ai_gguf SHARED ${SRC_FILES})

set(NDK_CPUFEATURES_DIR ${ANDROID_NDK}/sources/android/cpufeatures)
add_library(cpufeatures STATIC ${NDK_CPUFEATURES_DIR}/cpu-features.c)
target_include_directories(cpufeatures PUBLIC ${NDK_CPUFEATURES_DIR})

target_include_directories(ai_gguf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ai_gguf
    PRIVATE llama-engine
    PRIVATE llama
    PRIVATE ggml
    PRIVATE ggml-base
    PRIVATE cpufeatures
    PRIVATE android
    PRIVATE log
)

target_link_options(ai_gguf PRIVATE -Wl,-z,max-page-size=16384)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

message(STATUS "=== ai_gguf build type: ${CMAKE_BUILD_TYPE} ===")
message(STATUS "=== Building for ABI: ${ANDROID_ABI} ===")
message(STATUS "=== GGML_PAGE_SIZE: ${GGML_PAGE_SIZE} ===")
