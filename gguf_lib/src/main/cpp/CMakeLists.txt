cmake_minimum_required(VERSION 3.22.1)
project("gguf_lib")

# Force release-grade optimizations even in debug APK builds — debug llama.cpp
# is too slow to be usable on-device.
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

# Use add_compile_options() (not CMAKE_CXX_FLAGS_RELEASE) — the NDK toolchain
# overrides the latter after our CMakeLists.txt runs.
#
# -ffp-contract=fast       fuse a*b+c into a single FMA (ARM has native fmla)
# -fno-math-errno          skip setting errno after libm calls (Android ignores it)
# -fno-signed-zeros        treat -0.0 as +0.0
# -fno-trapping-math       FP ops never trap on ARM
# -fvisibility=hidden      hide all symbols, no PLT overhead, smaller .so
# -fomit-frame-pointer     free x29 for general use
# -ffunction-sections /
# -fdata-sections          enable --gc-sections dead-code stripping
#
# Do NOT add -ffast-math — it implies -fassociative-math, which reorders FP
# accumulation and breaks quantized inference accuracy.
# Do NOT add -fno-exceptions or -fno-rtti — gguf_lib.cpp has try/catch and
# nlohmann::json needs RTTI.
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

# Path to the upstream llama.cpp checkout. Update this if you relocate llama.cpp.
set(LLAMA_DIR "/home/home/dev/include/llama.cpp")

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(LLAMA_MULTIMODAL     ON  CACHE BOOL "" FORCE)
set(GGML_CPU             ON  CACHE BOOL "" FORCE)
set(GGML_LTO             ON  CACHE BOOL "" FORCE)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Baseline armv8-a — KleidiAI dispatches dotprod/i8mm/SVE per-device at runtime.
    # armv8.6-a as a global target caused SIGILL on Cortex-A76/A77 without i8mm.
    set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "" FORCE)
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
else()
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
endif()

# --gc-sections          strip unreferenced functions/data
# --icf=safe             merge identical code (only when addresses aren't taken)
# -z,max-page-size=16384 16KB page size support for Android 15+
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe -Wl,-z,max-page-size=16384")

add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)

set(RAG_INGEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rag_ingest)
set(PDFIUM_DIR     ${RAG_INGEST_DIR}/third_party/pdfium)

add_library(pdfium SHARED IMPORTED)
set_target_properties(pdfium PROPERTIES
    IMPORTED_LOCATION ${PDFIUM_DIR}/${ANDROID_ABI}/libpdfium.so)

set(MINIZ_SOURCES ${RAG_INGEST_DIR}/third_party/miniz/miniz.c)
set_source_files_properties(${MINIZ_SOURCES} PROPERTIES LANGUAGE C)

set(RAG_INGEST_SOURCES
    ${RAG_INGEST_DIR}/rag_ingest.cpp
    ${RAG_INGEST_DIR}/rag_ingest_html.cpp
    ${RAG_INGEST_DIR}/rag_ingest_zip.cpp
    ${RAG_INGEST_DIR}/rag_ingest_pdf.cpp
    ${MINIZ_SOURCES})

set(TEXT_DIGEST_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/text_digest)
set(TEXT_DIGEST_SOURCES ${TEXT_DIGEST_DIR}/text_digest.cpp)

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

add_library(${CMAKE_PROJECT_NAME} SHARED
    gguf_lib.cpp
    error_tracker.cpp
    ${LLAMA_DIR}/engine/tn-log.cpp
    ${LLAMA_DIR}/engine/tool-manager.cpp
    ${LLAMA_DIR}/engine/thread-engine.cpp
    ${LLAMA_DIR}/engine/rag-engine.cpp
    ${RAG_INGEST_SOURCES}
    ${TEXT_DIGEST_SOURCES}
    ${VLM_SOURCES})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/vendor
    ${LLAMA_DIR}/engine
    ${LLAMA_DIR}/engine/vlm
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RAG_INGEST_DIR}
    ${TEXT_DIGEST_DIR}
)

# -Wno-cast-qual silences vendored stb_image / miniaudio warnings used by VLM.
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -O3 -DNDEBUG -Wno-cast-qual)

target_link_libraries(${CMAKE_PROJECT_NAME}
    llama
    common
    pdfium
    android
    log)
