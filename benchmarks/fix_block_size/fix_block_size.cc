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
    // call kernel
    GEMM4<TPB>(dA, dB, dC, M, N, K);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);
    M = state.range(0);
    N = state.range(0);
    K = state.range(0);

    // Populate array
    cudaMallocManaged((void **)&dA, sizeof(T) * dataSize);
    cudaMallocManaged((void **)&dB, sizeof(T) * dataSize);
    cudaMallocManaged((void **)&dC, sizeof(T) * dataSize);
    cudaMallocManaged((void **)&testC, sizeof(T) * dataSize);

    cudabm::genRandom(dA, dataSize);
    cudabm::genRandom(dB, dataSize);

    cudabm::Gemm(dA, dB, testC, M, N, K);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    if (!cudabm::Equal<T>(M * N, dC, testC, 1e-2))
      throw std::runtime_error("Value diff occur in unroll");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(testC);
  }

  double getDataSize() { return (double)dataSize; }

 private:
  T *dA, *dB;
  T *testC, *dC;
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
