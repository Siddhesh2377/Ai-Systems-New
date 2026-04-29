cmake_minimum_required(VERSION 3.22.1)
project("ai_rmg" CXX)

set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT DEFINED RMG_ROOT)
    set(RMG_ROOT "/home/home/CLionProjects/rm-graph" CACHE PATH "rm-graph source root")
endif()
if(NOT EXISTS "${RMG_ROOT}/src/engine.h" OR NOT EXISTS "${RMG_ROOT}/format/rmg.h")
    message(FATAL_ERROR
        "RMG_ROOT='${RMG_ROOT}' does not contain rm-graph sources. "
        "Pass -DRMG_ROOT=<path> pointing at the rm-graph repository root.")
endif()

# add_compile_options is required: NDK toolchain overrides CMAKE_CXX_FLAGS_RELEASE.
# -ffp-contract=fast / -fno-{math-errno,signed-zeros,trapping-math} is the safe FP
# subset; do NOT switch to -ffast-math (its -fassociative-math reorders
# accumulation and breaks quantized inference).
add_compile_options(
        -O3
        -DNDEBUG
        -ffp-contract=fast
        -fno-math-errno
        -fno-signed-zeros
        -fno-trapping-math
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -fomit-frame-pointer
        -ffunction-sections
        -fdata-sections
        -fstack-protector-strong
        -Wall
        -Wextra
)

if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(-march=armv8.2-a+fp16+fp16fml+dotprod+crypto+sha2)
endif()

# -z,max-page-size=16384 needed for Android 15+ (16KB pages).
set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe -Wl,-z,max-page-size=16384")

add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_rmg.cpp
        ${RMG_ROOT}/src/log.cc
        ${RMG_ROOT}/src/cpu_features.cc
        ${RMG_ROOT}/src/rmg_loader.cc
        ${RMG_ROOT}/src/kernels.cc
        ${RMG_ROOT}/src/engine.cc)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${RMG_ROOT}/src
        ${RMG_ROOT}/format)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        log)
