cmake_minimum_required(VERSION 3.18)
project(ai_sd C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(QNN_SDK_ROOT /home/home/dev/arms/qairt/2.39.0.250926)

# Create qnnlibs directory in jniLibs instead of build directory
set(JNILIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
make_directory(${JNILIBS_DIR})

# Copy QNN libraries to jniLibs
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV68Stub.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV79Stub.so DESTINATION ${JNILIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so DESTINATION ${JNILIBS_DIR})

# Copy Hexagon DSP libraries
set(HEXAGON_LIBS_DIR ${JNILIBS_DIR}/dsp)
make_directory(${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81.so DESTINATION ${HEXAGON_LIBS_DIR})
file(COPY ${QNN_SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so DESTINATION ${HEXAGON_LIBS_DIR})

if(WIN32)
    set(PLATFORM_NAME "windows")
    set(HOST_TAG "windows-x86_64")
else()
    set(PLATFORM_NAME "linux")
    set(HOST_TAG "linux-x86_64")
endif()

set(OUTPUT_SUFFIX ${CMAKE_ANDROID_ARCH_ABI})
set(PLATFORM_LIBS log android dl m c GLESv2 EGL)
set(ANDROID_TOOLCHAIN_ROOT ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/${HOST_TAG} CACHE PATH "Android NDK root path")

add_compile_options(-O3 -Wall -fno-rtti -fPIC -fvisibility=hidden)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=for-loop-analysis")

set(ZSTD_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/zstd)
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "Build zstd programs" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "Build static zstd library" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "Enable multithreading support" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "Build zstd tests" FORCE)
add_subdirectory(${ZSTD_ROOT_DIR}/build/cmake zstd EXCLUDE_FROM_ALL)
set(ZSTD_INCLUDE_DIR ${ZSTD_ROOT_DIR}/lib)

set(MNN_ROOT_DIR 3rdparty/MNN)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_LOW_MEMORY ON CACHE BOOL "" FORCE)
set(MNN_CPU_WEIGHT_DEQUANT_GEMM ON CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID ON CACHE BOOL "" FORCE)
set(MNN_BUILD_FOR_ANDROID_COMMAND ON CACHE BOOL "" FORCE)
set(MNN_USE_LOGCAT OFF CACHE BOOL "" FORCE)
set(MNN_OPENCL ON CACHE BOOL "" FORCE)
set(MNN_SUPPORT_TRANSFORMER_FUSE ON CACHE BOOL "" FORCE)
add_subdirectory(3rdparty/MNN)

set(XTENSOR_USE_XSIMD ON)

set(PACKAGE_INCLUDES
        ${MNN_ROOT_DIR}/include
        ${QNN_SDK_ROOT}/include/QNN
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/include/nlohmann
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/json/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xtensor/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xtl/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xsimd/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/xtensor-blas/include
        ${ZSTD_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/CachingUtil
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/Log
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/PAL/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/Utils
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src/WrapperUtils
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -llog")
set(TOKENZIER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tokenizers-cpp)
if(NOT DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(CMAKE_ANDROID_ARCH_ABI "arm64-v8a" CACHE STRING "Android ABI" FORCE)
endif()
if(NOT DEFINED ANDROID_ABI)
    set(ANDROID_ABI ${CMAKE_ANDROID_ARCH_ABI} CACHE STRING "Android ABI" FORCE)
endif()
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/SampleApp/src)
set(SRC_DIR_LOG ${SRC_DIR}/Log)
set(SRC_DIR_PAL_LINUX ${SRC_DIR}/PAL/src/linux)
set(SRC_DIR_PAL_COMMON ${SRC_DIR}/PAL/src/common)
set(SRC_DIR_UTILS ${SRC_DIR}/Utils)
set(SRC_DIR_WRAPPER_UTILS ${SRC_DIR}/WrapperUtils)

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
        "${SRC_DIR}/QnnSampleApp.cpp"
        "${SRC_DIR_LOG}/*.cpp"
        "${SRC_DIR_PAL_LINUX}/*.cpp"
        "${SRC_DIR_PAL_COMMON}/*.cpp"
        "${SRC_DIR_UTILS}/*.cpp"
        "${SRC_DIR_WRAPPER_UTILS}/*.cpp"
)

if(QNN_DEBUG_ENABLE)
    add_compile_options(-O0 -g)
    add_definitions(-DQNN_API=)
else()
    add_definitions(-DQNN_API=__attribute__\(\(visibility\(\"default\"\)\)\))
endif()

add_library(${PROJECT_NAME} SHARED ${SOURCES})

if(NOT QNN_DEBUG_ENABLE)
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-s")
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${PACKAGE_INCLUDES})

# Remove this - let Gradle handle output location
# set_target_properties(${PROJECT_NAME} PROPERTIES
#         LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${OUTPUT_SUFFIX}"
# )

target_compile_definitions(${PROJECT_NAME} PRIVATE XTENSOR_USE_XSIMD)

target_link_libraries(${PROJECT_NAME} PRIVATE ${PLATFORM_LIBS} MNN tokenizers_cpp libzstd)