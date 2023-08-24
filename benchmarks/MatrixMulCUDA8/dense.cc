// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA8/dense.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basegemm.h"

template <typename T>
class Dense : public BaseGemm {
 public:
  void callKernel(benchmark::State &state) override {
    GEMM8(BaseGemm::getDeviceA(), BaseGemm::getDeviceB(),
          BaseGemm::getDeviceC(), state.range(0), state.range(1),
          state.range(2));
  }
};

#define BENCHMARK_GEMM9_OP(name, dType)                                      \
  BENCHMARK_TEMPLATE_DEFINE_F(Dense, name, dType)                            \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(Dense, name)                                          \
      ->Unit(benchmark::kMillisecond)                                        \
      ->ArgsProduct({{5120}, {4096}, {4096}});

#define BENCHMARK_GEMM9_OP_TYPE(dType) BENCHMARK_GEMM9_OP(Gemm_##dType, dType)

BENCHMARK_GEMM9_OP_TYPE(float)