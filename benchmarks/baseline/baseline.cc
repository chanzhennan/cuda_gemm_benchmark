// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "baseline/baseline.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class Baseline : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    GEMM<TPB>(dA, dB, dC, M, N, K);

    // for (int i = 0; i < M; i++)
    // {
    //   for (int j = 0; j < N; j++)
    //   {
    //     std::cout << dC[i * M + j] << " ";
    //   }

    //   std::cout << "\n";
    // }
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    dataSize = state.range(0) * state.range(0);
    M = state.range(0);
    N = state.range(0);
    K = state.range(0);

    // Populate array
    cudaMallocManaged(&dA, sizeof(T) * dataSize);
    cudaMallocManaged(&dB, sizeof(T) * dataSize);
    cudaMallocManaged(&dC, sizeof(T) * dataSize);
    cudaMallocManaged(&testC, sizeof(T) * dataSize);

    cudabm::genRandom(dA, dataSize);
    cudabm::genRandom(dB, dataSize);

    cudabm::Gemm(dA, dB, testC, M, N, K);
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    if (!cudabm::Equal<T>(M * N, dC, testC, 1e-2))
      throw std::runtime_error("Value diff occur in baseline");

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

#define BENCHMARK_GEMM1_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(Baseline, name, dType)                   \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize();                           \
    st.counters["FLOPS"] = benchmark::Counter{                         \
        getDataSize(), benchmark::Counter::kIsIterationInvariantRate}; \
  }                                                                    \
  BENCHMARK_REGISTER_F(Baseline, name)                                 \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Iterations(1)                                                  \
      ->Range(1024, 2048);

#define BENCHMARK_GEMM1_OP_TYPE(dType) BENCHMARK_GEMM1_OP(Gemm_##dType, dType)

BENCHMARK_GEMM1_OP_TYPE(float)
// BENCHMARK_GEMM1_OP_TYPE(int)
