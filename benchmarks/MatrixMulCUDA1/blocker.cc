// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA1/blocker.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class Blocker : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    GEMM1<BLOCKSIZE>(dA, dB, dC, state.range(0), state.range(1),
                     state.range(2));
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    // Populate array
    unsigned long M = (unsigned long)state.range(0);
    unsigned long N = (unsigned long)state.range(1);
    unsigned long K = (unsigned long)state.range(2);

    cudaMallocManaged(&dA, sizeof(T) * M * K);
    cudaMallocManaged(&dB, sizeof(T) * K * N);
    cudaMallocManaged(&dC, sizeof(T) * M * N);
    cudaMallocManaged(&testC, sizeof(T) * M * N);

    cudabm::genRandom(dA, M * K);
    cudabm::genRandom(dB, K * N);
  }

  void verify(const ::benchmark::State &st) {
    // for test M, N, K = state.range(0)
    cudabm::Gemm(dA, dB, testC, st.range(0), st.range(1), st.range(2));
    if (!cudabm::Equal<T>(st.range(0) * st.range(1), dC, testC, 1e-2))
      throw std::runtime_error("Value diff occur in Block");
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    verify(st);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(testC);
  }

  double getDataSize(const ::benchmark::State &state) {
    // datasize = 2 * M * N
    return (double)(2 * state.range(0) * state.range(1));
  }

  double getFlops(const ::benchmark::State &state) {
    // flops =  2 * M * N * K / s
    return (double)(2 * state.range(0) * state.range(1) * state.range(2));
  }

 private:
  T *dA, *dB;
  T *testC, *dC;
  long int dataSize;
  long int flops;
};

#define BENCHMARK_GEMM1_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(Blocker, name, dType)                    \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["FLOPS"] =                                             \
        benchmark::Counter(getFlops(st), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(Blocker, name)                                  \
      ->Unit(benchmark::kMillisecond)                                  \
      ->ArgsProduct({{4096}, {4096}, {4096}});

#define BENCHMARK_GEMM1_OP_TYPE(dType) BENCHMARK_GEMM1_OP(Gemm_##dType, dType)

BENCHMARK_GEMM1_OP_TYPE(float)
