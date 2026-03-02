cmake_minimum_required(VERSION 3.22.1)
project(ai_chatterbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

set(SRC_FILES
    src/chatterbox_jni.cpp
    src/bpe_tokenizer.cpp
    src/chatterbox_engine.cpp
)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SRC_FILES})

# ── ONNX Runtime (manually linked from AAR) ─────────────────────
# The onnxruntime-android AAR does not include prefab metadata,
# so we create an IMPORTED target manually.
# The .so is extracted from the AAR; at runtime the AAR's jni/ copy is used.
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/${ANDROID_ABI}/libonnxruntime.so
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/onnxruntime
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE android
    PRIVATE log
    PRIVATE onnxruntime
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,--gc-sections
)

message(STATUS "=== ai_chatterbox build: ${CMAKE_BUILD_TYPE} | ABI: ${ANDROID_ABI} ===")
message(STATUS "=== ORT .so: ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib/${ANDROID_ABI}/libonnxruntime.so ===")
