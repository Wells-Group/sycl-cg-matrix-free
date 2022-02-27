cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME sycl)
project(${PROJECT_NAME})

# Set C++ standard
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(supported_implementations hipSYCL LLVM)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)
if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(STATUS "No SYCL implementation specified.")
  set(SYCL_IMPL "LLVM")
endif()

# Check C++ flags for the benchmark
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

CHECK_CXX_COMPILER_FLAG("-Ofast" COMPILER_SUPPORTS_OFAST)
if(COMPILER_SUPPORTS_OFAST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
endif()

find_package(benchmark REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

if(SYCL_IMPL STREQUAL "LLVM")
    CHECK_CXX_COMPILER_FLAG("-fsycl" COMPILER_SUPPORTS_FSYCL)
    if(COMPILER_SUPPORTS_FSYCL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-unnamed-lambda")
        if(SYCL_TARGET STREQUAL "GPU")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda-sycldevice")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=spir64_x86_64")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")
        target_compile_definitions(${PROJECT_NAME} PRIVATE __LLVM_SYCL__)
    else()
        message(FATAL_ERROR "Compiler doest not support fsycl.")
    endif()
endif()

if(SYCL_IMPL STREQUAL "hipSYCL")
    find_package(hipSYCL REQUIRED)
    add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES bs.cpp)
    if(SYCL_TARGET STREQUAL "GPU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-path=${CUDA_PATH}")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic")
target_link_libraries(${PROJECT_NAME} PRIVATE benchmark::benchmark)

message(STATUS ${CMAKE_CXX_FLAGS})