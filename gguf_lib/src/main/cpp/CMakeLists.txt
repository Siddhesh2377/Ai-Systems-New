cmake_minimum_required(VERSION 3.22.1)
project("gguf_lib")

# Force Release optimizations even in debug APK builds
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

# Global optimization flags — applied to ALL targets (llama.cpp + JNI bridge).
# Using add_compile_options because set(CMAKE_CXX_FLAGS_RELEASE ...) gets
# overridden by the NDK toolchain after our CMakeLists.txt runs.
#
# -ffp-contract=fast: fuse a*b+c into single FMA instruction (ARM has native fmla)
# -fno-math-errno: skip setting errno after math ops (Android never checks it)
# -fno-signed-zeros: treat -0.0 as +0.0 (allows more FP optimizations)
# -fno-trapping-math: assume FP ops don't trap (ARM doesn't trap by default)
# -fvisibility=hidden: hide all symbols, reduces .so size + eliminates PLT overhead
# -fomit-frame-pointer: free x29 register for general use
# -ffunction-sections/-fdata-sections: put each function/data in its own section
#   so --gc-sections can strip unreferenced code (critical for dead code elimination)
add_compile_options(
    -ffp-contract=fast
    -fno-math-errno
    -fno-signed-zeros
    -fno-trapping-math
    -fvisibility=hidden
    -fomit-frame-pointer
    -ffunction-sections
    -fdata-sections
)

# Path to llama.cpp
set(LLAMA_DIR "/home/home/dev/include/llama.cpp")

# Build llama.cpp as static libraries
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Enable multimodal (VLM) support
set(LLAMA_MULTIMODAL ON CACHE BOOL "" FORCE)

# Enable GGML CPU optimizations
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_LTO ON CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Use baseline armv8-a for maximum device compatibility.
    # KleidiAI handles runtime dispatch to dotprod/i8mm/SVE kernels on capable devices.
    # armv8.6-a caused SIGILL crashes on devices without i8mm/dotprod (Cortex-A76, A77, etc.)
    set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "" FORCE)
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
endif()

# Linker optimizations:
# --gc-sections: strip unreferenced functions/data (works with -ffunction-sections)
# --icf=safe: merge identical code sections (safe = only merge if addresses aren't taken)
# -z,max-page-size=16384: Android 15+ 16KB page size support
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe -Wl,-z,max-page-size=16384")

add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)

# VLM (Vision Language Model) sources
set(VLM_DIR ${LLAMA_DIR}/engine/vlm)
set(VLM_SOURCES
    ${VLM_DIR}/clip.cpp
    ${VLM_DIR}/mtmd.cpp
    ${VLM_DIR}/mtmd-helper.cpp
    ${VLM_DIR}/mtmd-audio.cpp
    ${VLM_DIR}/models/cogvlm.cpp
    ${VLM_DIR}/models/conformer.cpp
    ${VLM_DIR}/models/glm4v.cpp
    ${VLM_DIR}/models/internvl.cpp
    ${VLM_DIR}/models/kimik25.cpp
    ${VLM_DIR}/models/kimivl.cpp
    ${VLM_DIR}/models/llama4.cpp
    ${VLM_DIR}/models/llava.cpp
    ${VLM_DIR}/models/minicpmv.cpp
    ${VLM_DIR}/models/mobilenetv5.cpp
    ${VLM_DIR}/models/nemotron-v2-vl.cpp
    ${VLM_DIR}/models/paddleocr.cpp
    ${VLM_DIR}/models/pixtral.cpp
    ${VLM_DIR}/models/qwen2vl.cpp
    ${VLM_DIR}/models/qwen3vl.cpp
    ${VLM_DIR}/models/siglip.cpp
    ${VLM_DIR}/models/whisper-enc.cpp
    ${VLM_DIR}/models/youtuvl.cpp
)

# JNI bridge + engine libraries
add_library(${CMAKE_PROJECT_NAME} SHARED
    gguf_lib.cpp
    ${LLAMA_DIR}/engine/tool-manager.cpp
    ${LLAMA_DIR}/engine/thread-engine.cpp
    ${LLAMA_DIR}/engine/rag-engine.cpp
    ${VLM_SOURCES})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/vendor
    ${LLAMA_DIR}/engine
    ${LLAMA_DIR}/engine/vlm
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# JNI bridge flags (inherits global C/CXX flags)
# Can't use -fno-exceptions or -fno-rtti — nlohmann::json and JNI error handling need them
# -Wno-cast-qual: suppress stb_image/miniaudio vendored code warnings from VLM
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -O3 -DNDEBUG -Wno-cast-qual)

target_link_libraries(${CMAKE_PROJECT_NAME}
    llama
    common
    android
    log)
