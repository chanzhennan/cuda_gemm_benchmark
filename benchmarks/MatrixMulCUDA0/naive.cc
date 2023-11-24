// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA0/naive.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basegemm.h"

template <typename T>
class Naive : public BaseGemm {
 public:
  void callKernel(benchmark::State& state) override {
    GEMM0<TPB>(BaseGemm::getDeviceA(), BaseGemm::getDeviceB(),
               BaseGemm::getDeviceC(), state.range(0), state.range(1),
               state.range(2));
  }

  void myprint2(benchmark::State& state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);

    // float* a = BaseGemm::getDeviceA();
    // float* b = BaseGemm::getDeviceB();
    float* c = BaseGemm::getDeviceC();

    printf("\n");
    printf("\n");
    printf("(cc) m * n \n\n");
    for (int j = 0; j < 8; j++) {
      for (int i = 0; i < 8; i++) {
        printf("%.2f ", c[j * n + i]);
      }
      printf("\n");
    }
  }

  void myprint1(benchmark::State& state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);

    float* a = BaseGemm::getDeviceA();
    float* b = BaseGemm::getDeviceB();

    printf("\n");
    printf("\n");
    printf("(aa) m * k \n\n");
    for (int j = 0; j < 128; j++) {
      printf("%.2f ", a[j * k]);
    }
    printf("\n");

    printf("\n\n (bb) k * n \n\n");
    for (int j = 0; j < 128; j++) {
      printf("%.2f ", b[j]);
    }
    printf("\n");
  }
};

#define BENCHMARK_GEMM0_OP(name, dType)                                      \
  BENCHMARK_TEMPLATE_DEFINE_F(Naive, name, dType)                            \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    myprint2(st);                                                            \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(Naive, name)                                          \
      ->Unit(benchmark::kMillisecond)                                        \
      ->Iterations(1)                                                        \
      ->ArgsProduct({{4096}, {4096}, {4096}});

#define BENCHMARK_GEMM0_OP_TYPE(dType) BENCHMARK_GEMM0_OP(Gemm_##dType, dType)

BENCHMARK_GEMM0_OP_TYPE(float)
