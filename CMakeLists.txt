cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(rt LANGUAGES CXX CUDA)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(rt raytracing.cu)
include_directories(rt Common)
