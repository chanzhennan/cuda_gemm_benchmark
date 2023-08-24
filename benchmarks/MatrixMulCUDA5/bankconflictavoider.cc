// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
#include "MatrixMulCUDA5/bankconflictavoider.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basegemm.h"

template <typename T>
class BcAvoider : public BaseGemm {
 public:
  void callKernel(benchmark::State &state) override {
    GEMM5(BaseGemm::getDeviceA(), BaseGemm::getDeviceB(),
          BaseGemm::getDeviceC(), state.range(0), state.range(1),
          state.range(2));
  }
};

#define BENCHMARK_GEMM5_OP(name, dType)                                \
  BENCHMARK_TEMPLATE_DEFINE_F(BcAvoider, name, dType)                  \
  (benchmark::State & st) {                                            \
    for (auto _ : st) {                                                \
      callKernel(st);                                                  \
    }                                                                  \
    st.counters["DATASIZE"] = getDataSize(st);                         \
    st.counters["FLOPS"] =                                             \
        benchmark::Counter(getFlops(st), benchmark::Counter::kIsRate); \
  }                                                                    \
  BENCHMARK_REGISTER_F(BcAvoider, name)                                \
      ->Unit(benchmark::kMillisecond)                                  \
      ->ArgsProduct({{1, 2}, {4096, 16384}, {4096, 16384}});

#define BENCHMARK_GEMM5_OP_TYPE(dType) BENCHMARK_GEMM5_OP(Gemm_##dType, dType)

BENCHMARK_GEMM5_OP_TYPE(float)
