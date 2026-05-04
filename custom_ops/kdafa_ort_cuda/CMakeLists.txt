cmake_minimum_required(VERSION 3.18)

project(kdafa_custom_ops_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_ROOT)
    message(FATAL_ERROR "Please set -DORT_ROOT=path/to/onnxruntime")
endif()

find_package(CUDAToolkit REQUIRED)

# GTX 1660 Super는 Turing, compute capability 7.5
# 다른 GPU면 86, 89 등으로 바꾸면 됨
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(kdafa_custom_ops_cuda SHARED
    kdafa_afa_cuda_op.cu
)

set_target_properties(kdafa_custom_ops_cuda PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(kdafa_custom_ops_cuda PRIVATE
    ${ORT_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kdafa_custom_ops_cuda PRIVATE
    ${ORT_ROOT}/lib/onnxruntime.lib
    CUDA::cudart
    CUDA::cufft
)

if(MSVC)
    target_compile_options(kdafa_custom_ops_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()