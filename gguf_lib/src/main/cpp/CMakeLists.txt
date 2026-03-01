cmake_minimum_required(VERSION 3.22.1)
project("gguf_lib")

# Path to llama.cpp
set(LLAMA_DIR "/home/home/dev/include/llama.cpp")

# Build llama.cpp as part of this project
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama)
add_subdirectory(${LLAMA_DIR}/engine ${CMAKE_CURRENT_BINARY_DIR}/engine)

# JNI bridge
add_library(${CMAKE_PROJECT_NAME} SHARED
    gguf_lib.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/engine
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    tn-engine
    llama
    common
    android
    log)
