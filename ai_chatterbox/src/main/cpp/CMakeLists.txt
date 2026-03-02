cmake_minimum_required(VERSION 3.22.1)
project(ai_chatterbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

set(SRC_FILES
    src/chatterbox_jni.cpp
)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SRC_FILES})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE android
    PRIVATE log
)

target_link_options(${CMAKE_PROJECT_NAME} PRIVATE
    -Wl,-z,max-page-size=16384
    -Wl,--gc-sections
)

message(STATUS "=== ai_chatterbox build: ${CMAKE_BUILD_TYPE} | ABI: ${ANDROID_ABI} ===")
