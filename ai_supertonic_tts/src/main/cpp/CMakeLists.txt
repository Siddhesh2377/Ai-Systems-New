cmake_minimum_required(VERSION 3.22.1)
project(ai_supertonic_tts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

set(SRC_FILES
        src/supertonic_jni.cpp
        src/audio/wav_encoder.cpp
)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SRC_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME}
        PRIVATE android
        PRIVATE log
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE -Wl,-z,max-page-size=16384)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

message(STATUS "=== ai_supertonic_tts build type: ${CMAKE_BUILD_TYPE} ===")
message(STATUS "=== Building for ABI: ${ANDROID_ABI} ===")
