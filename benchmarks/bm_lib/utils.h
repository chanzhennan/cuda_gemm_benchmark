// Copyright (c) 2023 Zhennanc Ltd. All rights reserved.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define checkCuda(call)                                                 \
  do {                                                                  \
    cudaError_t status = (call);                                        \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " (" \
                << status << ") at " << __FILE__ << ":" << __LINE__     \
                << std::endl;                                           \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define checkCuBlasErrors(func)                                               \
  {                                                                           \
    cublasStatus_t e = (func);                                                \
    if (e != CUBLAS_STATUS_SUCCESS)                                           \
      printf("%s %d CuBlas: %s", __FILE__, __LINE__, _cuBlasGetErrorEnum(e)); \
  }

namespace cudabm {

// benchmark string helper
std::string strFormat(const char* format, ...);

template <typename T>
void genRandom(T* vec, unsigned long len);

template <typename T>
void Gemm(T* dA, T* dB, T* dC, int m, int n, int k);

template <typename T>
bool Equal(const unsigned int n, const T* x, const T* y, const float tolerance);

void genOnes(float* vec, unsigned long len);
void Print(float* vec, size_t len);
float Sum(float* vec, size_t len);

}  // namespace cudabm
