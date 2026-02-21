cmake_minimum_required(VERSION 3.22.1)
project(ai_sd LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# QNN SDK (Qualcomm Neural Network)
# ============================================================================
set(QNN_SDK_ROOT /home/home/dev/lib/qairt)
if(NOT EXISTS "${QNN_SDK_ROOT}/include")
    message(FATAL_ERROR "QNN SDK not found at ${QNN_SDK_ROOT}")
endif()

# ============================================================================
# 16KB page alignment for Android (matching ai_gguf)
# ============================================================================
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# ============================================================================
# Compiler flags (matching xororz's build)
# ============================================================================
add_compile_options(
    -O3
    -Wall
    -fno-rtti
    -fPIC
    -fvisibility=hidden
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=for-loop-analysis")

# Platform libs
set(PLATFORM_LIBS log android dl m c GLESv2 EGL)

# ============================================================================
# Third-party: ZSTD
# ============================================================================
set(ZSTD_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zstd)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${ZSTD_ROOT_DIR}/build/cmake zstd EXCLUDE_FROM_ALL)
set(ZSTD_INCLUDE_DIR ${ZSTD_ROOT_DIR}/lib)

# ============================================================================
# Third-party: MNN
# ============================================================================
set(MNN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID ON CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT OFF CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "" FORCE)
add_subdirectory(${MNN_ROOT_DIR} mnn_build EXCLUDE_FROM_ALL)

# ============================================================================
# Third-party: tokenizers-cpp
# ============================================================================
set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp)
if(NOT DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(CMAKE_ANDROID_ARCH_ABI "arm64-v8a" CACHE STRING "Android ABI" FORCE)
endif()
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI ${CMAKE_ANDROID_ARCH_ABI} CACHE STRING "Android ABI" FORCE)
endif()
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

# ============================================================================
# Third-party: xtensor (header-only with XSIMD)
# ============================================================================
set(XTENSOR_USE_XSIMD ON)

# ============================================================================
# QNN SampleApp sources (patched for SD pipeline use)
# ============================================================================
set(SAMPLE_APP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/SampleApp)
set(SAMPLE_APP_LOG ${SAMPLE_APP_DIR}/Log)

if(WIN32)
    set(HOST_TAG "windows-x86_64")
else()
    set(HOST_TAG "linux-x86_64")
endif()
set(SAMPLE_APP_PAL_LINUX ${SAMPLE_APP_DIR}/PAL/src/linux)
set(SAMPLE_APP_PAL_COMMON ${SAMPLE_APP_DIR}/PAL/src/common)
set(SAMPLE_APP_UTILS ${SAMPLE_APP_DIR}/Utils)
set(SAMPLE_APP_WRAPPER ${SAMPLE_APP_DIR}/WrapperUtils)

file(GLOB SAMPLE_APP_SOURCES
    "${SAMPLE_APP_DIR}/QnnSampleApp.cpp"
    "${SAMPLE_APP_LOG}/*.cpp"
    "${SAMPLE_APP_PAL_LINUX}/*.cpp"
    "${SAMPLE_APP_PAL_COMMON}/*.cpp"
    "${SAMPLE_APP_UTILS}/*.cpp"
    "${SAMPLE_APP_WRAPPER}/*.cpp"
)

# ============================================================================
# Include directories
# ============================================================================
set(PACKAGE_INCLUDES
    # MNN
    ${MNN_ROOT_DIR}/include

    # QNN SDK
    ${QNN_SDK_ROOT}/include/QNN

    # QNN SampleApp
    ${SAMPLE_APP_DIR}
    ${SAMPLE_APP_DIR}/CachingUtil
    ${SAMPLE_APP_DIR}/Log
    ${SAMPLE_APP_DIR}/PAL/include
    ${SAMPLE_APP_DIR}/Utils
    ${SAMPLE_APP_DIR}/WrapperUtils

    # Our source
    ${CMAKE_CURRENT_SOURCE_DIR}/src

    # Header-only third-party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtl/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xtensor-blas/include
    ${ZSTD_INCLUDE_DIR}
)

# ============================================================================
# Source files
# ============================================================================
set(AI_SD_SOURCES
    # JNI entry point
    src/ai_sd_jni.cpp

    # State management
    src/state/diffusion_state.cpp

    # JNI utilities
    src/utils/jni_utils.cpp

    # Core inference pipeline (ported from xororz)
    src/pipeline/diffusion_pipeline.cpp
)

# ============================================================================
# Build SHARED library (not executable!)
# ============================================================================
# OLD (xororz): add_executable(libstable_diffusion_core.so ...)
# NEW:          add_library(ai_sd SHARED ...)
add_library(ai_sd SHARED
    ${AI_SD_SOURCES}
    ${SAMPLE_APP_SOURCES}
)

target_include_directories(ai_sd PRIVATE ${PACKAGE_INCLUDES})

target_compile_definitions(ai_sd PRIVATE
    XTENSOR_USE_XSIMD
    QNN_API=__attribute__\(\(visibility\(\"default\"\)\)\)
)

# ============================================================================
# Link libraries
# ============================================================================
target_link_libraries(ai_sd PRIVATE
    ${PLATFORM_LIBS}
    MNN
    tokenizers_cpp
    libzstd
)

# 16KB page alignment
target_link_options(ai_sd PRIVATE -Wl,-z,max-page-size=16384)

# Strip in release
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(ai_sd PRIVATE "-Wl,-s")
endif()

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

message(STATUS "=== ai_sd build type: ${CMAKE_BUILD_TYPE} ===")
message(STATUS "=== Building for ABI: ${ANDROID_ABI} ===")
message(STATUS "=== QNN SDK: ${QNN_SDK_ROOT} ===")
