// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA9/cublas.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/utils.h"

template <typename T>
class Cublas : public benchmark::Fixture {
 public:
  void callKernel(benchmark::State &state) {
    // call kernel
    GEMM9(dA, dB, dC, state.range(0), state.range(1), state.range(2),
          blas_handle);
  }

  void SetUp(const ::benchmark::State &state) BENCHMARK_OVERRIDE {
    // Populate array
    unsigned long M = (unsigned long)state.range(0);
    unsigned long N = (unsigned long)state.range(1);
    unsigned long K = (unsigned long)state.range(2);
    unsigned long asize = M * K;
    unsigned long bsize = K * N;
    unsigned long csize = M * N;

    cudaMallocManaged(&dA, sizeof(T) * asize);
    cudaMallocManaged(&dB, sizeof(T) * bsize);
    cudaMallocManaged(&dC, sizeof(T) * csize);
    cudaMallocManaged(&testC, sizeof(T) * csize);

    cublasCreate(&blas_handle);

    cudabm::genOnes(dA, asize);
    cudabm::genOnes(dB, bsize);
    cudabm::Gemm(dA, dB, testC, state.range(0), state.range(1), state.range(2));
  }

  void verify(const ::benchmark::State &st) {
    cudabm::Gemm(dA, dB, testC, st.range(0), st.range(1), st.range(2));
    if (!cudabm::Equal<T>(st.range(0) * st.range(1), dC, testC, 1e-2))
      throw std::runtime_error("Value diff occur in Dense");
  }

  void TearDown(const ::benchmark::State &st) BENCHMARK_OVERRIDE {
    // verify(st);

    cublasDestroy(blas_handle);
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
  cublasHandle_t blas_handle;
};

#define BENCHMARK_GEMM9_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(Cublas, name, dType)                     \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["FLOPS"] =                                             \
        benchmark::Counter(getFlops(st), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(Cublas, name)                                   \
      ->Unit(benchmark::kMillisecond)                                  \
      ->ArgsProduct({{1, 2}, {4096, 16384}, {4096, 16384}});

#define BENCHMARK_GEMM9_OP_TYPE(dType) BENCHMARK_GEMM9_OP(Gemm_##dType, dType)

BENCHMARK_GEMM9_OP_TYPE(float)
