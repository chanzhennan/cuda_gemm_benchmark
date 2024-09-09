// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#pragma once

#include <benchmark/benchmark.h>
#include <cublas.h>
#include <cuda_runtime.h>

#include <iostream>
// Helper macro to create a main routine in a test that runs the benchmarks

void GPUInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << prop.name << std::endl;
    std::cout << "  Memory: " << prop.totalGlobalMem / (1024 * 1024) << "MB"
              << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << std::endl;
  }
}

#define CUDA_BENCHMARK_MAIN()                                           \
  int main(int argc, char** argv) {                                     \
    GPUInfo();                                                          \
    ::benchmark::Initialize(&argc, argv);                               \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
    ::benchmark::Shutdown();                                            \
    return 0;                                                           \
  }                                                                     \
  int main(int, char**)
