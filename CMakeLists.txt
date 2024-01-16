cmake_minimum_required(VERSION 3.5)

project(
  cuda_gemm_benchmark
  DESCRIPTION "gemm benchmark"
  LANGUAGES CXX)

if(NOT DEFINED CUDA_ARCH)
  message(FATAL_ERROR "CUDA_ARCH macro is not defined. Please set CUDA_ARCH.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  message("CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT")
  set(CMAKE_INSTALL_PREFIX
      "install"
      CACHE PATH "CMAKE_INSTALL_PREFIX" FORCE)
endif()

# Options
option(USE_SYSTEM_LIBS "use system libs or not" OFF)

# Find cuda SDK
find_package(CUDA REQUIRED)

# googletest
include(FetchContent)
if(NOT googletest_POPULATED)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/googletest)
  # https://cmake.org/cmake/help/latest/policy/CMP0077.html
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(INSTALL_GTEST OFF)
  FetchContent_MakeAvailable(googletest)
  FetchContent_GetProperties(googletest)

endif()

# googlebenchmark
if(NOT googlebenchmark_POPULATED)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.1
    GIT_SHALLOW TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/googlebenchmark)
  set(BENCHMARK_ENABLE_TESTING OFF)
  set(BENCHMARK_ENABLE_INSTALL OFF)
  set(HAVE_GNU_POSIX_REGEX OFF)
  set(BENCHMARK_INSTALL_DOCS OFF)
  FetchContent_MakeAvailable(googlebenchmark)
  FetchContent_GetProperties(googlebenchmark)
endif()

enable_testing()

include_directories(${googlebenchmark_BINARY_DIR}/include)
include_directories(${googlebenchmark_SOURCE_DIR}/include)

include_directories(benchmarks)

add_subdirectory(benchmarks)
