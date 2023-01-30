// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "fix_block_size/fix_block_size.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class FixBlockSize : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    cudaMemcpy(dA, A, sizeof(T) * dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(T) * dataSize, cudaMemcpyHostToDevice);

    // call kernel
    GEMM4<TPB>(dA, dB, dC, M, N, K);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);
    M = state.range(0);
    N = state.range(0);
    K = state.range(0);

    // Populate array
    cudaMallocHost(&A, sizeof(T) * dataSize);
    cudaMallocHost(&B, sizeof(T) * dataSize);
    cudaMallocHost(&C, sizeof(T) * dataSize);
    cudaMalloc((void **)&dA, sizeof(T) * dataSize);
    cudaMalloc((void **)&dB, sizeof(T) * dataSize);
    cudaMalloc((void **)&dC, sizeof(T) * dataSize);

    cudabm::genRandom(A, dataSize);
    cudabm::genRandom(B, dataSize);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
  }

  double getDataSize() { return (double)dataSize; }

 private:
  T *dA, *A;
  T *dB, *B;
  T *dC, *C;
  int M, N, K;
  long int dataSize;
};

#define BENCHMARK_GEMM4_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(FixBlockSize, name, dType)               \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(FixBlockSize, name)                             \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(1024, 2048);

#define BENCHMARK_GEMM4_OP_TYPE(dType) BENCHMARK_GEMM4_OP(Gemm_##dType, dType)

BENCHMARK_GEMM4_OP_TYPE(float)
// BENCHMARK_GEMM4_OP_TYPE(int)
