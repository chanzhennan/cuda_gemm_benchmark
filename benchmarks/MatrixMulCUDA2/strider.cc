// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA2/strider.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class Strider : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    GEMM2<BLOCKSIZE>(dA, dB, dC, state.range(0), state.range(0),
                     state.range(0));
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    // Populate array
    long int dsize = state.range(0) * state.range(0);
    cudaMallocManaged(&dA, sizeof(T) * dsize);
    cudaMallocManaged(&dB, sizeof(T) * dsize);
    cudaMallocManaged(&dC, sizeof(T) * dsize);
    cudaMallocManaged(&testC, sizeof(T) * dsize);

    cudabm::genRandom(dA, dsize);
    cudabm::genRandom(dB, dsize);

    // for test M, N, K = state.range(0)
    cudabm::Gemm(dA, dB, testC, state.range(0), state.range(0), state.range(0));
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    // if (!cudabm::Equal<T>(M * N, dC, testC, 1e-2))
    //   throw std::runtime_error("Value diff occur in Naive");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(testC);
  }

  double getDataSize(const ::benchmark::State &state) {
    // datasize = 2 * M * N
    return (double)(2 * state.range(0) * state.range(0));
  }

  double getFlops(const ::benchmark::State &state) {
    // flops =  2 * M * N * K / s
    return (double)(2 * state.range(0) * state.range(0) * state.range(0));
  }

 private:
  T *dA, *dB;
  T *testC, *dC;
  long int dataSize;
  long int flops;
};

#define BENCHMARK_GEMM2_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(Strider, name, dType)                    \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["FLOPS"] =                                             \
        benchmark::Counter(getFlops(st), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(Strider, name)                                  \
      ->Unit(benchmark::kMillisecond)                                  \
      ->RangeMultiplier(2)                                             \
      ->Range(2048, 4096);

#define BENCHMARK_GEMM2_OP_TYPE(dType) BENCHMARK_GEMM2_OP(Gemm_##dType, dType)

BENCHMARK_GEMM2_OP_TYPE(float)
// BENCHMARK_GEMM2_OP_TYPE(int)
