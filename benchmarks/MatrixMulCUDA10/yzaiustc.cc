// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA10/yzaiustc.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basegemm.h"

template <typename T>
class Yzaiustc : public BaseGemm {
 public:
  void callKernel(benchmark::State &state) override {
    GEMM10(BaseGemm::getDeviceA(), BaseGemm::getDeviceB(),
           BaseGemm::getDeviceC(), state.range(0), state.range(1),
           state.range(2));
  }
};

#define BENCHMARK_GEMM10_OP(name, dType)                                     \
  BENCHMARK_TEMPLATE_DEFINE_F(Yzaiustc, name, dType)                         \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(Yzaiustc, name)                                       \
      ->Unit(benchmark::kMillisecond)                                        \
      ->ArgsProduct({{5120}, {4096}, {4096}});

#define BENCHMARK_GEMM10_OP_TYPE(dType) BENCHMARK_GEMM10_OP(Gemm_##dType, dType)

BENCHMARK_GEMM10_OP_TYPE(float)
