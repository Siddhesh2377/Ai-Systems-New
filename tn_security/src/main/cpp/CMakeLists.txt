cmake_minimum_required(VERSION 3.22.1)
project("tn_security")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -fvisibility=hidden
    -ffunction-sections
    -fdata-sections
    -fstack-protector-strong
)

set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=safe -Wl,-z,max-page-size=16384")

add_library(${CMAKE_PROJECT_NAME} SHARED
    tn_security.cpp
    tn_security_jni.cpp)

target_include_directories(${CMAKE_PROJECT_NAME}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -O2 -DNDEBUG)

target_link_libraries(${CMAKE_PROJECT_NAME} android log)
