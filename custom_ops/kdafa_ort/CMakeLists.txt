# custom_ops/kdafa_ort/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

project(kdafa_custom_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_ROOT)
    message(FATAL_ERROR "Please set -DORT_ROOT=path/to/onnxruntime")
endif()

add_library(kdafa_custom_ops SHARED
    kdafa_afa_op.cpp
)

set_target_properties(kdafa_custom_ops PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_include_directories(kdafa_custom_ops PRIVATE
    ${ORT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(WIN32)
    target_compile_definitions(kdafa_custom_ops PRIVATE NOMINMAX)
    target_link_libraries(kdafa_custom_ops PRIVATE
        ${ORT_ROOT}/lib/onnxruntime.lib
    )
else()
    target_link_libraries(kdafa_custom_ops PRIVATE
        ${ORT_ROOT}/lib/libonnxruntime.so
    )
endif()

if(MSVC)
    target_compile_options(kdafa_custom_ops PRIVATE /utf-8)
endif()