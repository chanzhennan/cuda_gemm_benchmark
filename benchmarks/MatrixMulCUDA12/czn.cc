// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.
// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include "MatrixMulCUDA12/czn.cuh"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bm_lib/basegemm.h"

template <typename T>
class Czn : public BaseGemm {
 public:
  void callKernel(benchmark::State &state) override {
    GEMM12(BaseGemm::getDeviceA(), BaseGemm::getDeviceB(),
           BaseGemm::getDeviceC(), state.range(0), state.range(1),
           state.range(2));
  }


  void myprint(benchmark::State &state)
  {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);


    float* b = BaseGemm::getDeviceB();
    float* a = BaseGemm::getDeviceA();
    
    printf("bbbbbbbbb\n\n");  
    for (int i = 0; i < 128; i++)
    {
      printf("%.2f ", b[i]);
    }
    printf("\n\n");  


    printf("aaaaa ");
    for (int i = 0; i < 128; i++)
    {
      printf("%.2f ", a[i * k]);
    }
    printf("\n\n");  
    
  }
  

};


#define BENCHMARK_GEMM12_OP(name, dType)                                     \
  BENCHMARK_TEMPLATE_DEFINE_F(Czn, name, dType)                              \
  (benchmark::State & st) {                                                  \
    for (auto _ : st) {                                                      \
      callKernel(st);                                                        \
    }                                                                        \
    myprint(st);                                                                \
    double iter = st.iterations();                                           \
    st.counters["operation"] = getFlops(st) * iter;                          \
    st.counters["TFlops"] = benchmark::Counter((getFlops(st) * iter / 1e12), \
                                               benchmark::Counter::kIsRate); \
  }                                                                          \
  BENCHMARK_REGISTER_F(Czn, name)                                            \
      ->Unit(benchmark::kMillisecond)                                        \
      ->Iterations(1)                                                        \
      ->ArgsProduct({{5120}, {4096}, {4096}});

#define BENCHMARK_GEMM12_OP_TYPE(dType) BENCHMARK_GEMM12_OP(Gemm_##dType, dType)

BENCHMARK_GEMM12_OP_TYPE(float)
